//! Router: interfaz estable + implementación provisional por contenido.
//!
//! El clasificador entrenado (otro proceso) sustituirá a `KeywordRouter`
//! implementando el mismo trait `Router`. La interfaz deliberadamente recibe
//! tokens *y* historial, y devuelve el id de ruta con su justificación, para
//! que el futuro clasificador no necesite cambios en el runtime.

pub mod keyword;

use crate::ukj::RouteInfo;

/// Un turno cerrado, tal como lo ve el router.
///
/// Se conserva completo (texto y tokens) a propósito: el clasificador
/// entrenado podrá mirar el turno entero, no solo el id de ruta.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct HistoryTurn {
    pub user: String,
    pub assistant: String,
    pub route_id: u32,
    pub route_label: String,
    pub tokens_in: usize,
    pub tokens_out: usize,
}

/// Entrada del router: texto, tokens del tokenizer y turnos previos.
pub struct RouteRequest<'a> {
    pub text: &'a str,
    pub tokens: &'a [u32],
    pub history: &'a [HistoryTurn],
    pub routes: &'a [RouteInfo],
}

/// Decisión del router, auditable.
#[derive(Debug, Clone)]
pub struct RouteChoice {
    pub route_id: u32,
    pub label: String,
    pub score: f32,
    pub reason: String,
    /// Puntuación de todas las rutas evaluadas (para /routes y depuración).
    pub candidates: Vec<(u32, f32)>,
    pub used_fallback: bool,
}

impl RouteChoice {
    pub fn fallback(routes: &[RouteInfo], reason: impl Into<String>) -> Self {
        let route = routes.first();
        match route {
            Some(r) => RouteChoice {
                route_id: r.id,
                label: r.label(),
                score: 0.0,
                reason: reason.into(),
                candidates: Vec::new(),
                used_fallback: true,
            },
            None => RouteChoice {
                route_id: 0,
                label: "(sin rutas)".to_string(),
                score: 0.0,
                reason: reason.into(),
                candidates: Vec::new(),
                used_fallback: true,
            },
        }
    }
}

/// Interfaz del router. Clavable: se puede reemplazar en caliente por el
/// clasificador entrenado sin tocar el REPL ni el motor.
pub trait Router {
    fn name(&self) -> &str;
    fn choose(&self, req: &RouteRequest) -> RouteChoice;
}

/// `true` cuando el texto parece continuación del turno anterior y conviene
/// mantener la ruta (evita saltos por frases cortas como "¿y por qué?").
pub fn is_followup(text: &str, n_tokens: usize) -> bool {
    let norm = crate::util::normalize(text);
    if norm.is_empty() {
        return true;
    }
    let starters = [
        "y ",
        "pero ",
        "entonces ",
        "por que",
        "porque",
        "como ",
        "otro",
        "otra",
        "mas ",
        "explicame",
        "explica",
        "continua",
        "sigue",
        "amplia",
        "detalla",
        "ejemplo",
        "y eso",
        "entonces",
        "ahora",
    ];
    if starters.iter().any(|s| norm.starts_with(s)) {
        return true;
    }
    n_tokens <= 4
}

#[cfg(test)]
pub(crate) fn test_routes() -> Vec<RouteInfo> {
    vec![
        RouteInfo {
            id: 1,
            niveles: vec!["mundo".into(), "tecnologia".into(), "programacion".into()],
            offset: 0,
            nbytes: 0,
            dim: 0,
        },
        RouteInfo {
            id: 2,
            niveles: vec!["mundo".into(), "ciencia".into(), "matematicas".into()],
            offset: 0,
            nbytes: 0,
            dim: 0,
        },
        RouteInfo {
            id: 3,
            niveles: vec!["mundo".into(), "geografia".into(), "estados_unidos".into()],
            offset: 0,
            nbytes: 0,
            dim: 0,
        },
        RouteInfo {
            id: 9,
            niveles: vec!["mundo".into(), "general".into()],
            offset: 0,
            nbytes: 0,
            dim: 0,
        },
    ]
}
