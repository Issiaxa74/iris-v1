//! Estrés de parámetros (FEEDBACK_SPEC.md §1) en su parte observable.
//!
//! UKJ v0.1 guarda un tensor Float32 por ruta, sin bloques de sub-parámetros
//! comprimidos, así que aún no hay nada que descomprimir. Cabrita implementa y
//! persiste la parte que sí existe hoy: el contador de activaciones por ruta,
//! su decaimiento por ciclo y el umbral (global o por ruta). Cuando llegue un
//! UKJ con sub-parámetros, la expansión temporal se conecta en
//! `engine::stats`, leyendo `is_stressed()`.
//!
//! Los contadores viven en un sidecar `<modelo>.stress.json` porque cabrita no
//! reescribe el UKJ (eso es trabajo del escritor F1a).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StressEntry {
    pub activations: u64,
    pub stressed_cycles: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StressFile {
    schema: u32,
    threshold: u64,
    decay: f32,
    routes: BTreeMap<String, StressEntry>,
    /// Umbrales por ruta fijados con `/stress <ruta> <n>`.
    /// `#[serde(default)]` para seguir leyendo sidecars antiguos.
    #[serde(default)]
    overrides: BTreeMap<String, u64>,
}

/// Una fila de `/stress`.
#[derive(Debug, Clone)]
pub struct StressReport {
    pub route_id: u32,
    pub activations: u64,
    pub threshold: u64,
    pub stressed: bool,
    pub stressed_cycles: u64,
}

#[derive(Debug)]
pub struct StressTable {
    path: PathBuf,
    threshold: u64,
    decay: f32,
    counts: BTreeMap<u32, StressEntry>,
    overrides: BTreeMap<u32, u64>,
}

impl StressTable {
    /// Carga el sidecar si existe; si no, empieza vacío con el umbral indicado.
    pub fn load(model_path: &Path, threshold: u64, decay: f32) -> Result<Self> {
        let path = sidecar_path(model_path);
        let mut table = StressTable {
            path: path.clone(),
            threshold: threshold.max(1),
            decay: clamp_decay(decay),
            counts: BTreeMap::new(),
            overrides: BTreeMap::new(),
        };
        if path.exists() {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("no se pudo leer {}", path.display()))?;
            let file: StressFile = serde_json::from_str(&raw)
                .with_context(|| format!("sidecar de estrés corrupto: {}", path.display()))?;
            table.counts = file
                .routes
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect();
            table.threshold = file.threshold.max(1);
            table.decay = clamp_decay(file.decay);
            table.overrides = file
                .overrides
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v.max(1))))
                .collect();
        }
        Ok(table)
    }

    /// Ajusta el umbral por ruta; se persiste en el sidecar junto a los contadores.
    pub fn set_route_threshold(&mut self, route_id: u32, threshold: u64) {
        self.overrides.insert(route_id, threshold.max(1));
    }

    pub fn threshold_for(&self, route_id: u32) -> u64 {
        self.overrides
            .get(&route_id)
            .copied()
            .unwrap_or(self.threshold)
    }

    pub fn threshold(&self) -> u64 {
        self.threshold
    }

    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Suma activaciones a una ruta (una por token procesado por defecto).
    pub fn register(&mut self, route_id: u32, activations: u64) {
        let threshold = self.threshold_for(route_id);
        let e = self.counts.entry(route_id).or_default();
        e.activations = e.activations.saturating_add(activations);
        if e.activations >= threshold {
            e.stressed_cycles += 1;
        }
    }

    /// Decae todos los contadores al cerrar un ciclo (un turno).
    pub fn decay_cycle(&mut self) {
        if (self.decay - 1.0).abs() < f32::EPSILON {
            return;
        }
        for e in self.counts.values_mut() {
            e.activations = (e.activations as f32 * self.decay) as u64;
        }
    }

    pub fn activations(&self, route_id: u32) -> u64 {
        self.counts
            .get(&route_id)
            .map(|e| e.activations)
            .unwrap_or(0)
    }

    pub fn is_stressed(&self, route_id: u32) -> bool {
        self.activations(route_id) >= self.threshold_for(route_id)
    }

    pub fn report(&self, route_ids: impl Iterator<Item = u32>) -> Vec<StressReport> {
        route_ids
            .map(|id| {
                let entry = self.counts.get(&id).cloned().unwrap_or_default();
                StressReport {
                    route_id: id,
                    activations: entry.activations,
                    threshold: self.threshold_for(id),
                    stressed: entry.activations >= self.threshold_for(id),
                    stressed_cycles: entry.stressed_cycles,
                }
            })
            .collect()
    }

    pub fn save(&self) -> Result<()> {
        let file = StressFile {
            schema: 1,
            threshold: self.threshold,
            decay: self.decay,
            routes: self
                .counts
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            overrides: self
                .overrides
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        };
        let json = serde_json::to_string_pretty(&file)?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&self.path, json)
            .with_context(|| format!("no se pudo guardar {}", self.path.display()))?;
        Ok(())
    }
}

fn clamp_decay(d: f32) -> f32 {
    if d.is_finite() && d > 0.0 {
        d.min(1.0)
    } else {
        1.0
    }
}

pub fn sidecar_path(model_path: &Path) -> PathBuf {
    let mut s = model_path.as_os_str().to_os_string();
    s.push(".stress.json");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acumula_y_marca_estres() {
        let dir = std::env::temp_dir().join("cabrita_stress");
        std::fs::create_dir_all(&dir).unwrap();
        let mut t = StressTable::load(&dir.join("m.ukj"), 10, 0.5).unwrap();
        t.register(1, 4);
        t.register(1, 6);
        assert_eq!(t.activations(1), 10);
        assert!(t.is_stressed(1));
        t.decay_cycle();
        assert_eq!(t.activations(1), 5);
        assert!(!t.is_stressed(1));
    }

    #[test]
    fn umbral_por_ruta_y_persistencia() {
        let dir = std::env::temp_dir().join("cabrita_stress_sidecar");
        let p = dir.join("m.ukj");
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(sidecar_path(&p));

        let mut t = StressTable::load(&p, 100, 0.9).unwrap();
        t.set_route_threshold(2, 3);
        t.register(2, 4);
        assert!(t.is_stressed(2));
        assert!(!t.is_stressed(1));
        t.save().unwrap();

        let t2 = StressTable::load(&p, 1, 1.0).unwrap();
        assert_eq!(t2.activations(2), 4);
        assert_eq!(t2.threshold(), 100);
        let _ = std::fs::remove_file(sidecar_path(&p));
    }
}
