//! TOJI: memoria viva por sectores, en archivos binarios `.mem`.
//!
//! Formato v0.1 implementado aquí (little-endian), con las decisiones que el
//! borrador TOJI_SPEC.md dejaba abiertas; todas documentadas en README.md:
//!
//! ```text
//! [0..8)    "TOJI\0\0\0\0" (el magic del spec, rellenado a 8 bytes como IRISTOK)
//! [8..12)   u32 version (=1)
//! [12..14)  u16 len + sector utf8
//! [..)      u32 n_neuronas, u32 n_anormales, u32 n_cables, u32 n_fronteras
//! neuronas: n × { u32 id, u16 len + id_str utf8, u32 n_satelites,
//!                 satelites × { u64 t_epoch, u8 tipo(0 evento|1 emoción),
//!                                f32 fuerza, u16 len + texto utf8 } }
//! anormales: n × { u32 id, u16 len + concepto utf8,
//!                  u32 n_hijos, hijos u32×n, u16 len + matiz utf8 }
//! cables:   n × { u32 a, u32 b, u8 tipo, f32 peso }
//! fronteras: n × { u16 len + sector_a, u16 len + sector_b, f32 peso }
//! footer:   u64 checksum = suma de bytes previos mód 2^64
//! ```
//!
//! * Las cadenas usan prefijo u16 (igual que los niveles de UKJ).
//! * Se añade `n_fronteras` porque el borrador lista fronteras sin contador.
//! * El archivo puede ir comprimido con zstd (se detecta por su magic
//!   `28 B5 2F FD`); al guardar, cabrita escribe plano salvo `--compress-mem`.
//! * El checksum sigue la convención del UKJ v0.1 (suma aditiva), no CRC64.

pub mod memory;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

pub use memory::{MemoryUpdater, MemoryView};

pub const MAGIC: &[u8; 8] = b"TOJI\0\0\0\0";
pub const VERSION_SUPPORTED: u32 = 1;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

pub const SAT_EVENT: u8 = 0;
pub const SAT_EMOTION: u8 = 1;

#[allow(dead_code)] // parte del formato .mem; TOJI aún no registra estos tipos
pub const CABLE_CAUSA: u8 = 0;
pub const CABLE_SIMILITUD: u8 = 1;
#[allow(dead_code)] // parte del formato .mem; TOJI aún no registra estos tipos
pub const CABLE_OPOSICION: u8 = 2;
#[allow(dead_code)] // parte del formato .mem; TOJI aún no registra estos tipos
pub const CABLE_SECUENCIA: u8 = 3;

#[derive(Debug, Clone)]
pub struct Satellite {
    pub t_epoch: u64,
    pub kind: u8,
    pub strength: f32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub id: u32,
    pub id_str: String,
    pub satellites: Vec<Satellite>,
}

impl Neuron {
    pub fn new(id: u32, id_str: impl Into<String>) -> Self {
        Neuron {
            id,
            id_str: id_str.into(),
            satellites: Vec::new(),
        }
    }

    /// Fuerza acumulada de la neurona (suma de satélites, saturada en 1.0).
    pub fn strength(&self) -> f32 {
        self.satellites
            .iter()
            .map(|s| s.strength)
            .fold(0.0f32, |a, b| a + b)
            .min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct Anormal {
    pub id: u32,
    pub concept: String,
    pub children: Vec<u32>,
    pub nuance: String,
}

#[derive(Debug, Clone)]
pub struct Cable {
    pub a: u32,
    pub b: u32,
    pub kind: u8,
    pub weight: f32,
}

impl Cable {
    pub fn same_pair(&self, a: u32, b: u32) -> bool {
        (self.a == a && self.b == b) || (self.a == b && self.b == a)
    }
}

#[derive(Debug, Clone)]
pub struct Border {
    pub sector_a: String,
    pub sector_b: String,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct Sector {
    pub name: String,
    pub neurons: Vec<Neuron>,
    pub anormals: Vec<Anormal>,
    pub cables: Vec<Cable>,
    pub borders: Vec<Border>,
}

impl Sector {
    pub fn new(name: impl Into<String>) -> Self {
        Sector {
            name: name.into(),
            neurons: Vec::new(),
            anormals: Vec::new(),
            cables: Vec::new(),
            borders: Vec::new(),
        }
    }

    pub fn next_id(&self) -> u32 {
        self.neurons
            .iter()
            .map(|n| n.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(1)
    }

    pub fn find_neuron(&self, id_str: &str) -> Option<usize> {
        self.neurons.iter().position(|n| n.id_str == id_str)
    }

    #[allow(dead_code)] // consulta puntual para el motor y la auditoría del .mem
    pub fn neuron(&self, id: u32) -> Option<&Neuron> {
        self.neurons.iter().find(|n| n.id == id)
    }

    pub fn to_bytes(&self, compress: bool) -> Result<Vec<u8>> {
        let mut w = Writer::new();
        w.bytes(MAGIC);
        w.u32(VERSION_SUPPORTED);
        w.lp_str(&self.name)?;
        w.u32(clamp_u32(self.neurons.len()));
        w.u32(clamp_u32(self.anormals.len()));
        w.u32(clamp_u32(self.cables.len()));
        w.u32(clamp_u32(self.borders.len()));

        for n in &self.neurons {
            w.u32(n.id);
            w.lp_str(&n.id_str)?;
            w.u32(clamp_u32(n.satellites.len()));
            for s in &n.satellites {
                w.u64(s.t_epoch);
                w.u8(s.kind);
                w.f32(s.strength);
                w.lp_str(&s.text)?;
            }
        }
        for a in &self.anormals {
            w.u32(a.id);
            w.lp_str(&a.concept)?;
            w.u32(clamp_u32(a.children.len()));
            for c in &a.children {
                w.u32(*c);
            }
            w.lp_str(&a.nuance)?;
        }
        for c in &self.cables {
            w.u32(c.a);
            w.u32(c.b);
            w.u8(c.kind);
            w.f32(c.weight);
        }
        for b in &self.borders {
            w.lp_str(&b.sector_a)?;
            w.lp_str(&b.sector_b)?;
            w.f32(b.weight);
        }
        let sum = w.sum();
        w.u64(sum);
        if compress {
            zstd::encode_all(&w.buf[..], 3).context("zstd falló al comprimir .mem")
        } else {
            Ok(w.buf)
        }
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        let data: Vec<u8>;
        let bytes: &[u8] = if raw.len() >= 4 && raw[..4] == ZSTD_MAGIC {
            data = zstd::decode_all(raw).context(".mem zstd corrupto")?;
            &data
        } else {
            raw
        };
        if bytes.len() < 8 + 4 + 2 + 16 + 8 {
            bail!("archivo .mem demasiado corto ({} B)", bytes.len());
        }
        if &bytes[..8] != MAGIC {
            bail!(
                "magic inválido: {:?} (se esperaba \"TOJI\\0\")",
                String::from_utf8_lossy(&bytes[..5.min(bytes.len())])
            );
        }
        // El checksum cubre todo menos los últimos 8 bytes.
        let body_len = bytes.len() - 8;
        let computed = bytes[..body_len]
            .iter()
            .fold(0u64, |a, b| a.wrapping_add(*b as u64));
        let mut cb = [0u8; 8];
        cb.copy_from_slice(&bytes[body_len..]);
        let stored = u64::from_le_bytes(cb);
        if stored != computed {
            bail!("checksum .mem inválido (guardado {stored}, calculado {computed})");
        }

        let mut r = Reader::new(&bytes[..body_len], 8);
        let version = r.u32()?;
        if version != VERSION_SUPPORTED {
            bail!(".mem versión {version} no soportada (v{VERSION_SUPPORTED})");
        }
        let name = r.lp_str()?;
        let n_neurons = r.u32()? as usize;
        let n_anormals = r.u32()? as usize;
        let n_cables = r.u32()? as usize;
        let n_borders = r.u32()? as usize;

        let mut neurons = Vec::with_capacity(n_neurons.min(1 << 20));
        for _ in 0..n_neurons {
            let id = r.u32()?;
            let id_str = r.lp_str()?;
            let n_sat = r.u32()? as usize;
            let mut satellites = Vec::with_capacity(n_sat.min(1 << 16));
            for _ in 0..n_sat {
                let t_epoch = r.u64()?;
                let kind = r.u8()?;
                let strength = r.f32()?;
                let text = r.lp_str()?;
                satellites.push(Satellite {
                    t_epoch,
                    kind,
                    strength,
                    text,
                });
            }
            neurons.push(Neuron {
                id,
                id_str,
                satellites,
            });
        }
        let mut anormals = Vec::with_capacity(n_anormals.min(1 << 20));
        for _ in 0..n_anormals {
            let id = r.u32()?;
            let concept = r.lp_str()?;
            let n_children = r.u32()? as usize;
            let mut children = Vec::with_capacity(n_children.min(1 << 16));
            for _ in 0..n_children {
                children.push(r.u32()?);
            }
            let nuance = r.lp_str()?;
            anormals.push(Anormal {
                id,
                concept,
                children,
                nuance,
            });
        }
        let mut cables = Vec::with_capacity(n_cables.min(1 << 22));
        for _ in 0..n_cables {
            let a = r.u32()?;
            let b = r.u32()?;
            let kind = r.u8()?;
            let weight = r.f32()?;
            cables.push(Cable { a, b, kind, weight });
        }
        let mut borders = Vec::with_capacity(n_borders.min(1 << 16));
        for _ in 0..n_borders {
            let sector_a = r.lp_str()?;
            let sector_b = r.lp_str()?;
            let weight = r.f32()?;
            borders.push(Border {
                sector_a,
                sector_b,
                weight,
            });
        }
        Ok(Sector {
            name,
            neurons,
            anormals,
            cables,
            borders,
        })
    }

    /// Fusiona otro sector del mismo nombre: une neuronas por palabra, suma
    /// fuerza de cables, conserva el mayor peso de frontera y remapea ids.
    pub fn merge(&mut self, other: &Sector) -> MergeStats {
        let mut stats = MergeStats::default();
        let mut remap: std::collections::HashMap<u32, u32> = Default::default();
        for inc in &other.neurons {
            match self.find_neuron(&inc.id_str) {
                Some(idx) => {
                    remap.insert(inc.id, self.neurons[idx].id);
                    for s in &inc.satellites {
                        let dup = self.neurons[idx]
                            .satellites
                            .iter()
                            .any(|e| e.text == s.text && e.t_epoch == s.t_epoch);
                        if !dup {
                            self.neurons[idx].satellites.push(s.clone());
                            stats.satellites_added += 1;
                        }
                    }
                }
                None => {
                    let id = self.next_id();
                    remap.insert(inc.id, id);
                    self.neurons.push(Neuron {
                        id,
                        id_str: inc.id_str.clone(),
                        satellites: inc.satellites.clone(),
                    });
                    stats.neurons_added += 1;
                }
            }
        }
        for inc in &other.cables {
            let (a, b) = match (remap.get(&inc.a), remap.get(&inc.b)) {
                (Some(a), Some(b)) => (*a, *b),
                _ => continue,
            };
            if let Some(existing) = self
                .cables
                .iter_mut()
                .find(|c| c.same_pair(a, b) && c.kind == inc.kind)
            {
                existing.weight = (existing.weight + inc.weight).min(1.0);
                stats.cables_merged += 1;
            } else {
                self.cables.push(Cable {
                    a,
                    b,
                    kind: inc.kind,
                    weight: inc.weight,
                });
                stats.cables_added += 1;
            }
        }
        for inc in &other.anormals {
            match self.anormals.iter_mut().find(|a| a.concept == inc.concept) {
                Some(existing) => {
                    for c in &inc.children {
                        if let Some(mapped) = remap.get(c) {
                            if !existing.children.contains(mapped) {
                                existing.children.push(*mapped);
                            }
                        }
                    }
                    if !inc.nuance.is_empty() && !existing.nuance.contains(&inc.nuance) {
                        if !existing.nuance.is_empty() {
                            existing.nuance.push_str(" | ");
                        }
                        existing.nuance.push_str(&inc.nuance);
                    }
                    stats.anormals_merged += 1;
                }
                None => {
                    let mut a = inc.clone();
                    a.id = self
                        .anormals
                        .iter()
                        .map(|x| x.id)
                        .max()
                        .map(|m| m + 1)
                        .unwrap_or(1);
                    a.children = inc
                        .children
                        .iter()
                        .filter_map(|c| remap.get(c).copied())
                        .collect();
                    self.anormals.push(a);
                    stats.anormals_added += 1;
                }
            }
        }
        for inc in &other.borders {
            let na = crate::util::sector_name(&inc.sector_a);
            let nb = crate::util::sector_name(&inc.sector_b);
            match self.borders.iter_mut().find(|b| {
                let a = crate::util::sector_name(&b.sector_a);
                let c = crate::util::sector_name(&b.sector_b);
                (a == na && c == nb) || (a == nb && c == na)
            }) {
                Some(b) => b.weight = b.weight.max(inc.weight),
                None => {
                    self.borders.push(Border {
                        sector_a: inc.sector_a.clone(),
                        sector_b: inc.sector_b.clone(),
                        weight: inc.weight,
                    });
                    stats.borders_added += 1;
                }
            }
        }
        stats
    }
}

#[derive(Debug, Default, Clone)]
pub struct MergeStats {
    pub neurons_added: usize,
    pub satellites_added: usize,
    pub cables_added: usize,
    pub cables_merged: usize,
    pub anormals_added: usize,
    pub anormals_merged: usize,
    pub borders_added: usize,
}

/// Almacén de sectores sobre un directorio.
#[derive(Debug, Clone)]
pub struct MemStore {
    dir: PathBuf,
    compress: bool,
}

impl MemStore {
    pub fn new(dir: impl Into<PathBuf>, compress: bool) -> Self {
        MemStore {
            dir: dir.into(),
            compress,
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path_for(&self, sector: &str) -> PathBuf {
        self.dir
            .join(format!("{}.mem", crate::util::sector_name(sector)))
    }

    /// Carga un sector; si no existe devuelve uno vacío (nunca falla por ausencia).
    pub fn load(&self, sector: &str) -> Result<Sector> {
        let path = self.path_for(sector);
        if !path.exists() {
            return Ok(Sector::new(crate::util::sector_name(sector)));
        }
        let raw =
            std::fs::read(&path).with_context(|| format!("no se pudo leer {}", path.display()))?;
        Sector::from_bytes(&raw)
            .with_context(|| format!("sector TOJI corrupto: {}", path.display()))
    }

    pub fn save(&self, sector: &Sector) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("no se pudo crear {}", self.dir.display()))?;
        let path = self.path_for(&sector.name);
        let bytes = sector.to_bytes(self.compress)?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("no se pudo escribir {}", path.display()))?;
        Ok(path)
    }

    pub fn list(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|e| e.to_str()) == Some("mem") {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        out.push(stem.to_string());
                    }
                }
            }
        }
        out.sort();
        out
    }
}

/// Escritor binario little-endian con suma de bytes acumulada.
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { buf: Vec::new() }
    }
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn lp_str(&mut self, s: &str) -> Result<()> {
        let b = s.as_bytes();
        if b.len() > u16::MAX as usize {
            bail!("cadena .mem de {} B supera el límite de 65535", b.len());
        }
        self.u16(b.len() as u16);
        self.bytes(b);
        Ok(())
    }
    fn sum(&self) -> u64 {
        self.buf.iter().fold(0u64, |a, b| a.wrapping_add(*b as u64))
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8], pos: usize) -> Self {
        Reader { buf, pos }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("longitud desbordada")?;
        if end > self.buf.len() {
            bail!(
                "archivo .mem truncado en offset {} (faltaban {n} B)",
                self.pos
            );
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn lp_str(&mut self) -> Result<String> {
        let len = self.u16()? as usize;
        let b = self.take(len)?;
        String::from_utf8(b.to_vec()).context("cadena .mem no es UTF-8 válido")
    }
}

fn clamp_u32(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sector() -> Sector {
        let mut s = Sector::new("programacion");
        let mut n = Neuron::new(1, "rust");
        n.satellites.push(Satellite {
            t_epoch: 1_700_000_000,
            kind: SAT_EVENT,
            strength: 0.7,
            text: "el usuario compiló cabrita".into(),
        });
        n.satellites.push(Satellite {
            t_epoch: 1_700_000_010,
            kind: SAT_EMOTION,
            strength: 0.4,
            text: "le gustó el resultado".into(),
        });
        s.neurons.push(n);
        s.neurons.push(Neuron::new(2, "compilador"));
        s.anormals.push(Anormal {
            id: 1,
            concept: "poseer".into(),
            children: vec![1, 2],
            nuance: "tener de forma estable".into(),
        });
        s.cables.push(Cable {
            a: 1,
            b: 2,
            kind: CABLE_SIMILITUD,
            weight: 0.5,
        });
        s.borders.push(Border {
            sector_a: "programacion".into(),
            sector_b: "general".into(),
            weight: 0.5,
        });
        s
    }

    #[test]
    fn roundtrip_binario() {
        let s = sample_sector();
        let bytes = s.to_bytes(false).unwrap();
        assert_eq!(&bytes[..5], b"TOJI\0");
        let back = Sector::from_bytes(&bytes).unwrap();
        assert_eq!(back.name, s.name);
        assert_eq!(back.neurons.len(), 2);
        assert_eq!(back.neurons[0].satellites.len(), 2);
        assert_eq!(back.neurons[0].satellites[1].text, "le gustó el resultado");
        assert_eq!(back.anormals[0].children, vec![1, 2]);
        assert_eq!(back.cables[0].weight, 0.5);
        assert_eq!(back.borders[0].sector_b, "general");
    }

    #[test]
    fn roundtrip_zstd() {
        let s = sample_sector();
        let bytes = s.to_bytes(true).unwrap();
        assert_eq!(bytes[..4], ZSTD_MAGIC);
        let back = Sector::from_bytes(&bytes).unwrap();
        assert_eq!(back.neurons.len(), 2);
    }

    #[test]
    fn detecta_corrupcion() {
        let s = sample_sector();
        let mut bytes = s.to_bytes(false).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let err = Sector::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("checksum"), "{err}");
    }

    #[test]
    fn merge_une_por_palabra() {
        let mut a = sample_sector();
        let mut b = Sector::new("programacion");
        let mut n = Neuron::new(7, "rust");
        n.satellites.push(Satellite {
            t_epoch: 1_700_000_020,
            kind: SAT_EVENT,
            strength: 0.6,
            text: "segunda conversación".into(),
        });
        b.neurons.push(n);
        b.neurons.push(Neuron::new(8, "ownership"));
        b.cables.push(Cable {
            a: 7,
            b: 8,
            kind: CABLE_SIMILITUD,
            weight: 0.4,
        });
        let stats = a.merge(&b);
        assert_eq!(stats.neurons_added, 1);
        assert_eq!(stats.satellites_added, 1);
        assert_eq!(a.neurons.len(), 3);
        assert_eq!(a.neurons[0].satellites.len(), 3);
        let added = a.find_neuron("ownership").unwrap();
        let mapped = a.neurons[added].id;
        assert!(a.cables.iter().any(|c| c.same_pair(1, mapped)));
    }

    #[test]
    fn store_guarda_y_lista() {
        let dir = std::env::temp_dir().join("cabrita_toji_store");
        let _ = std::fs::remove_dir_all(&dir);
        let store = MemStore::new(&dir, false);
        let mut s = store.load("Programación").unwrap();
        assert_eq!(s.name, "programacion");
        s.neurons.push(Neuron::new(1, "rust"));
        let p = store.save(&s).unwrap();
        assert!(p.exists());
        let back = store.load("programacion").unwrap();
        assert_eq!(back.neurons.len(), 1);
        assert_eq!(store.list(), vec!["programacion".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
