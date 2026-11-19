//! Motor de respuesta: contrato estable + implementaciones.
//!
//! Cabrita define aquí el punto de conexión:
//!
//! ```text
//! trait Engine { fn respond(&mut self, ctx: &EngineContext) -> Result<EngineAnswer> }
//! ```
//!
//! `ctx` lleva todo lo que un motor necesita: ruta activa, sus tensores en RAM,
//! memoria TOJI del turno, historial, tokens de entrada y el estado de estrés.
//! Hay tres motores registrados en `from_name`:
//!
//! * `transformer` — forward real del trunk desde el UKJ + manifiesto, con caché
//!   KV y muestreo. Carga sus propios tensores y no usa `ctx.tensors`.
//! * `stats` — diagnóstico: muestra ruta, memoria y estadísticas de la ruta.
//! * `template` — plantilla honesta que repite el mensaje; no inventa conocimiento.

mod transformer;

pub use transformer::TransformerEngine;

use std::path::Path;

use anyhow::Result;

use crate::router::HistoryTurn;
use crate::toji::MemoryView;
use crate::tok::Tokenizer;
use crate::ukj::{RouteInfo, RouteTensors};
use crate::util::human_bytes;

/// Parametros de generacion del motor real.
#[derive(Debug, Clone)]
pub struct GenOptions {
    pub max_new: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl Default for GenOptions {
    fn default() -> Self {
        GenOptions {
            max_new: 48,
            temperature: 0.9,
            top_k: 40,
            top_p: 0.95,
            seed: None,
        }
    }
}

/// Todo lo que el motor puede leer en un turno.
pub struct EngineContext<'a> {
    pub route: &'a RouteInfo,
    pub tensors: Option<&'a RouteTensors>,
    pub memory: &'a MemoryView,
    pub history: &'a [HistoryTurn],
    pub user_text: &'a str,
    pub tokens: &'a [u32],
    pub tok: &'a Tokenizer,
    pub tokens_in: usize,
    pub model_version: &'a str,
    pub route_bytes: u64,
    pub stressed: bool,
    pub stress_count: u64,
    pub stress_threshold: u64,
}

/// Respuesta completa antes de tokenizar. `notes` se muestra como metadatos.
#[derive(Debug, Clone)]
pub struct EngineAnswer {
    pub text: String,
    pub notes: Vec<String>,
}

pub trait Engine {
    fn name(&self) -> &str;

    /// `false` si el motor carga sus propios tensores (el transformer lee el
    /// trunk del UKJ al construirse y no necesita la copia de la ruta activa).
    fn uses_route_tensors(&self) -> bool {
        true
    }

    fn respond(&mut self, ctx: &EngineContext) -> Result<EngineAnswer>;
}

/// Construye el motor pedido por nombre.
///
/// `model_path` solo lo usan los motores que leen tensores (hoy el transformer);
/// los de plantilla lo ignoran.
pub fn from_name(name: &str, model_path: &Path, opts: GenOptions) -> Result<Box<dyn Engine>> {
    match name.to_ascii_lowercase().as_str() {
        "template" | "plantilla" => Ok(Box::new(TemplateEngine::default())),
        "stats" | "tensor" => Ok(Box::new(StatsEngine::default())),
        "transformer" | "real" | "modelo" => {
            Ok(Box::new(TransformerEngine::open(model_path, opts)?))
        }
        other => {
            anyhow::bail!("motor desconocido: {other} (disponibles: template, stats, transformer)")
        }
    }
}
fn echo(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        flat
    } else {
        let cut: String = flat.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

#[derive(Default)]
pub struct TemplateEngine;

impl Engine for TemplateEngine {
    fn name(&self) -> &str {
        "template"
    }

    fn respond(&mut self, ctx: &EngineContext) -> Result<EngineAnswer> {
        let norm = crate::util::normalize(ctx.user_text);
        let greeting = ["hola", "buenas", "buenos dias", "buenas tardes", "hey"]
            .iter()
            .any(|g| norm.starts_with(g));

        let mut text = String::new();
        if greeting {
            text.push_str(
                "Hola. Soy Iris en modo plantilla: el modelo aún no tiene pesos entrenados, \
                 así que no puedo responder con conocimiento real. Sí puedo mostrarte la ruta \
                 y la memoria que se activarían.\n",
            );
        } else {
            text.push_str(&format!(
                "(modo plantilla: sin pesos entrenados) Recibí «{}».\n",
                echo(ctx.user_text, 180)
            ));
        }
        text.push_str(&format!("Ruta activa: {}.\n", ctx.route.label()));
        text.push_str(&ctx.memory.summary());
        if !ctx.history.is_empty() {
            text.push_str(&format!("\nTurnos en contexto: {}.", ctx.history.len()));
        }

        Ok(EngineAnswer {
            text,
            notes: Vec::new(),
        })
    }
}

/// Igual que la plantilla, pero añade estadísticas reales de los tensores de la
/// ruta: sirve para verificar que el UKJ entregado se carga y se lee bien.
#[derive(Default)]
pub struct StatsEngine;

impl Engine for StatsEngine {
    fn name(&self) -> &str {
        "stats"
    }

    fn respond(&mut self, ctx: &EngineContext) -> Result<EngineAnswer> {
        let mut text = format!(
            "(modo plantilla+stats: sin pesos útiles todavía) Mensaje: «{}».\n",
            echo(ctx.user_text, 140)
        );
        text.push_str(&format!(
            "Ruta activa: {} (id {}).\n",
            ctx.route.label(),
            ctx.route.id
        ));
        text.push_str(&format!(
            "Modelo: {} · {} tok in · {} de tensores.\n",
            ctx.model_version,
            ctx.tokens_in,
            human_bytes(ctx.route_bytes)
        ));
        let mut notes = Vec::new();
        match ctx.tensors {
            Some(t) => {
                text.push_str(&format!(
                    "Tensores: dim={} · media={:.4} · energía={:.2} · huella={:016x}.\n",
                    t.dim,
                    t.mean(),
                    t.energy(),
                    t.checksum()
                ));
                notes.push(format!(
                    "ruta {} · dim {} · media {:.3} · {} en RAM",
                    t.route_id,
                    t.dim,
                    t.mean(),
                    human_bytes(t.bytes())
                ));
            }
            None => text.push_str("Tensores: la ruta no se cargó en este turno.\n"),
        }
        text.push_str(&ctx.memory.summary());
        if let Some(n) = ctx.memory.activated.first() {
            text.push_str(&format!(
                "\nNeurona más activa: #{} {} (fuerza {:.2}, {} satélites).",
                n.id, n.word, n.strength, n.satellites
            ));
        }
        text.push_str(&format!(
            "\nEstrés: {}/{} activaciones{}.",
            ctx.stress_count,
            ctx.stress_threshold,
            if ctx.stressed {
                " (umbral superado, ruta marcada)"
            } else {
                ""
            }
        ));
        Ok(EngineAnswer { text, notes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toji::MemoryUpdater;
    use crate::toji::Sector;

    fn tok<'a>() -> &'a Tokenizer {
        Box::leak(Box::new(
            Tokenizer::load(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/iris.tok")).unwrap(),
        ))
    }
    fn ctx<'a>(
        route: &'a RouteInfo,
        tensors: &'a RouteTensors,
        mem: &'a MemoryView,
        hist: &'a [HistoryTurn],
        text: &'a str,
    ) -> EngineContext<'a> {
        EngineContext {
            route,
            tensors: Some(tensors),
            memory: mem,
            history: hist,
            user_text: text,
            tokens: &[104, 111, 108, 97],
            tok: tok(),
            tokens_in: 5,
            model_version: "UKJ v1 (p4-one)",
            route_bytes: 256,
            stressed: false,
            stress_count: 3,
            stress_threshold: 12,
        }
    }

    #[test]
    fn plantilla_no_inventa_conocimiento() {
        let route = RouteInfo {
            id: 1,
            niveles: vec!["mundo".into(), "general".into()],
            offset: 0,
            nbytes: 0,
            dim: 0,
        };
        let t = RouteTensors {
            route_id: 1,
            dim: 2,
            data: vec![1.0, 2.0],
        };
        let mem = MemoryView::empty("mundo");
        let mut e = TemplateEngine::default();
        let ans = e
            .respond(&ctx(&route, &t, &mem, &[], "¿quién ganó el mundial?"))
            .unwrap();
        assert!(ans.text.contains("modo plantilla"));
        assert!(ans.text.contains("sin pesos entrenados"));
        assert!(ans.text.contains("mundo › general"));
        assert!(ans.notes.is_empty());
        assert_eq!(e.name(), "template");
    }

    #[test]
    fn stats_muestra_tensores() {
        let route = RouteInfo {
            id: 2,
            niveles: vec!["mundo".into(), "ciencia".into(), "matematicas".into()],
            offset: 0,
            nbytes: 0,
            dim: 4,
        };
        let t = RouteTensors {
            route_id: 2,
            dim: 4,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut s = Sector::new("ciencia");
        let mem = MemoryUpdater::default().absorb(&mut s, "integral de x");
        let mut e = StatsEngine::default();
        let ans = e
            .respond(&ctx(&route, &t, &mem, &[], "integral de x"))
            .unwrap();
        assert!(ans.text.contains("dim=4"));
        assert!(ans.text.contains("media=2.5000"));
        assert!(ans.text.contains("3/12"));
        assert!(!ans.notes.is_empty());
    }

    #[test]
    fn motor_desconocido() {
        let p = Path::new("assets/test.ukj");
        assert!(from_name("magia", p, GenOptions::default()).is_err());
        assert!(from_name("stats", p, GenOptions::default()).is_ok());
    }
}
