//! Router provisional por contenido (reemplazable).
//!
//! Reglas:
//! * puntúa coincidencias **de contenido** entre el texto y los niveles de la
//!   ruta más un vocabulario de alias por tema;
//! * desambigua con contexto, nunca por sonido: `estado` solo cuenta como país
//!   si aparecen términos de país en la misma frase (regla explícita, testeada
//!   con la frase trampa del FEEDBACK_SPEC: «¿en qué estado está el mundo?»);
//! * si el turno es una continuación («¿y por qué?», «otro ejemplo»), da
//!   inercia a la ruta anterior;
//! * si nada supera el umbral, cae a la ruta por defecto declarada en el modelo
//!   (`general`/`mundo`/`sft`, o el id más bajo).
//!
//! El clasificador entrenado sustituirá este módulo implementando `Router`.

use std::collections::BTreeSet;

use crate::router::{is_followup, RouteChoice, RouteRequest, Router};
use crate::ukj::RouteInfo;
use crate::util::{normalize, words};

/// Regla de desambiguación léxica.
struct AmbiguityRule {
    /// Palabra que dispara la duda (se compara por prefijo: `estado`/`estados`).
    trigger: &'static str,
    /// Alguna de estas cadenas debe aparecer para que la lectura ambigua valga.
    requires_any: &'static [&'static str],
    /// Rutas vetadas cuando falta el contexto (subcadena de la etiqueta).
    veto_labels: &'static [&'static str],
}

const RULES: &[AmbiguityRule] = &[AmbiguityRule {
    trigger: "estado",
    requires_any: &[
        "unidos",
        "eeuu",
        "ee.uu",
        "usa",
        "washington",
        "california",
        "texas",
        "florida",
        "norteamerica",
        "norteamericano",
    ],
    veto_labels: &[
        "estados unidos",
        "estados_unidos",
        "estado unido",
        "usa",
        "eeuu",
    ],
}];

/// Alias por tema. Vocabulario provisional del router, no conocimiento del modelo.
fn aliases_for(term: &str) -> &'static [&'static str] {
    match term {
        t if t.contains("programacion") || t.contains("informatica") => &[
            "programar",
            "programando",
            "codigo",
            "code",
            "rust",
            "julia",
            "python",
            "c",
            "backend",
            "frontend",
            "api",
            "base de datos",
            "bug",
            "compilar",
            "compilador",
            "funcion",
            "variable",
            "git",
            "servidor",
            "web",
            "sql",
            "algoritmo",
            "debug",
            "borrow",
            "memoria ram",
            "hilo",
            "concurrencia",
            "cabrita",
        ],
        t if t.contains("matematicas") || t.contains("calculo") => &[
            "derivada",
            "integral",
            "ecuacion",
            "algebra",
            "suma",
            "resta",
            "multiplicacion",
            "division",
            "numero",
            "probabilidad",
            "estadistica",
            "teorema",
            "funcion lineal",
            "limite",
            "matriz",
            "vector",
            "porcentaje",
        ],
        t if t.contains("estados unidos") || t.contains("estados_unidos") => &[
            "eeuu",
            "ee.uu",
            "usa",
            "unidos",
            "washington",
            "california",
            "texas",
            "florida",
            "gringo",
            "norteamerica",
            "norteamericano",
            "nueva york",
            "los angeles",
        ],
        t if t.contains("general") || t.contains("sft") => &[
            "hola",
            "buenas",
            "gracias",
            "quien eres",
            "que puedes hacer",
            "ayuda",
            "como estas",
            "buenos dias",
            "buenas tardes",
            "adios",
        ],
        _ => &[],
    }
}

pub struct KeywordRouter {
    min_score: f32,
    followup_boost: f32,
}

impl Default for KeywordRouter {
    fn default() -> Self {
        Self {
            min_score: 0.30,
            followup_boost: 0.60,
        }
    }
}

impl KeywordRouter {
    pub fn new(min_score: f32, followup_boost: f32) -> Self {
        Self {
            min_score,
            followup_boost,
        }
    }
}

impl Router for KeywordRouter {
    fn name(&self) -> &str {
        "keyword-v0-provisional"
    }

    fn choose(&self, req: &RouteRequest) -> RouteChoice {
        if req.routes.is_empty() {
            return RouteChoice::fallback(req.routes, "el modelo no declara rutas");
        }

        let norm = normalize(req.text);
        let input_words: BTreeSet<String> = words(req.text, 2).into_iter().collect();
        let history_route = req.history.last().map(|t| t.route_id);
        let followup = history_route.is_some() && is_followup(req.text, req.tokens.len());

        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(req.routes.len());
        let mut notes: Vec<String> = Vec::new();
        let mut best: Option<(u32, String, f32, Vec<String>)> = None;

        for route in req.routes {
            let mut score = 0.0f32;
            let mut hits: Vec<String> = Vec::new();
            let mut terms: BTreeSet<String> = BTreeSet::new();
            for t in route.terms() {
                terms.insert(normalize(&t));
            }
            for t in route.terms() {
                for a in aliases_for(&normalize(&t)) {
                    terms.insert(normalize(a));
                }
            }
            terms.remove("");

            for term in &terms {
                if input_words.contains(term) {
                    score += 1.0;
                    hits.push(term.clone());
                } else if term.contains(' ') && norm.contains(term.as_str()) {
                    score += 1.2;
                    hits.push(term.clone());
                } else if term.chars().count() >= 5 {
                    // Coincidencia de raíz: "programar" ↔ "programacion".
                    let root_hit = norm.split_whitespace().any(|w| {
                        let wl = w.chars().count();
                        wl >= 5
                            && wl.abs_diff(term.chars().count()) <= 3
                            && w.chars()
                                .zip(term.chars())
                                .take_while(|(a, b)| a == b)
                                .count()
                                >= 5
                    });
                    if root_hit {
                        score += 0.45;
                        hits.push(term.clone());
                    }
                }
            }

            if followup && Some(route.id) == history_route {
                score += self.followup_boost;
                hits.push("turno anterior".to_string());
            }

            // Desambiguación por contexto (después de la inercia, para que la
            // inercia no pueda devolver una lectura que el contenido descarta).
            for rule in RULES {
                let triggered = norm.split_whitespace().any(|w| w.starts_with(rule.trigger));
                if !triggered {
                    continue;
                }
                if rule.requires_any.iter().any(|needle| norm.contains(needle)) {
                    continue;
                }
                let label = normalize(&route.label());
                if rule
                    .veto_labels
                    .iter()
                    .any(|v| label.contains(&normalize(v)))
                {
                    if score > 0.0 {
                        notes.push(format!(
                            "desambiguación: «{}» sin contexto de país; se descarta «{}»",
                            rule.trigger,
                            route.label()
                        ));
                    }
                    score = 0.0;
                    hits.clear();
                }
            }

            candidates.push((route.id, score));
            if best.as_ref().map(|(_, _, s, _)| score > *s).unwrap_or(true) {
                best = Some((route.id, route.label(), score, hits));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((id, label, score, hits)) if score >= self.min_score => {
                let mut reason = if hits.is_empty() {
                    "coincidencia mínima de contenido".to_string()
                } else {
                    format!("contenido: {}", hits.join(", "))
                };
                for n in &notes {
                    reason.push_str(" · ");
                    reason.push_str(n);
                }
                RouteChoice {
                    route_id: id,
                    label,
                    score,
                    reason,
                    candidates,
                    used_fallback: false,
                }
            }
            Some((_, _, score, _)) => {
                let fb = default_route(req.routes);
                let mut reason = format!(
                    "sin coincidencia suficiente (máx {score:.2} < {:.2})",
                    self.min_score
                );
                for n in &notes {
                    reason.push_str(" · ");
                    reason.push_str(n);
                }
                RouteChoice {
                    route_id: fb.id,
                    label: fb.label(),
                    score,
                    reason,
                    candidates,
                    used_fallback: true,
                }
            }
            None => RouteChoice::fallback(req.routes, "no se pudo puntuar ninguna ruta"),
        }
    }
}

fn default_route(routes: &[RouteInfo]) -> RouteInfo {
    for needle in ["general", "mundo", "sft"] {
        if let Some(r) = routes
            .iter()
            .find(|r| r.niveles.iter().any(|n| normalize(n).contains(needle)))
        {
            return r.clone();
        }
    }
    routes
        .iter()
        .min_by_key(|r| r.id)
        .cloned()
        .unwrap_or_else(|| routes[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::test_routes;
    use crate::router::HistoryTurn;

    fn history(route_id: u32) -> Vec<HistoryTurn> {
        vec![HistoryTurn {
            user: "pregunta previa".into(),
            assistant: "respuesta previa".into(),
            route_id,
            route_label: "previa".into(),
            tokens_in: 3,
            tokens_out: 5,
        }]
    }

    fn choose(text: &str, routes: &[RouteInfo], hist: &[HistoryTurn]) -> RouteChoice {
        let router = KeywordRouter::default();
        let tokens: Vec<u32> = text.bytes().map(u32::from).collect();
        router.choose(&RouteRequest {
            text,
            tokens: &tokens,
            history: hist,
            routes,
        })
    }

    #[test]
    fn elige_programacion_por_contenido() {
        let routes = test_routes();
        let c = choose("quiero programar una API en Rust", &routes, &[]);
        assert_eq!(c.route_id, 1, "{}", c.reason);
        assert!(!c.used_fallback);
    }

    #[test]
    fn elige_matematicas() {
        let routes = test_routes();
        let c = choose("resuelve la integral de x al cuadrado", &routes, &[]);
        assert_eq!(c.route_id, 2, "{}", c.reason);
    }

    #[test]
    fn trampa_estado_del_mundo_no_cae_en_estados_unidos() {
        let routes = test_routes();
        // Incluso con inercia al tema país, el contenido manda.
        let hist = history(3);
        let c = choose("¿en qué estado está el mundo?", &routes, &hist);
        assert_ne!(c.route_id, 3, "cayó en estados_unidos: {}", c.reason);
        assert!(c.reason.contains("desambiguación"), "{}", c.reason);
    }

    #[test]
    fn contexto_de_pais_si_activa_estados_unidos() {
        let routes = test_routes();
        let c = choose("¿cuál es la capital de Estados Unidos?", &routes, &[]);
        assert_eq!(c.route_id, 3, "{}", c.reason);
    }

    #[test]
    fn continuacion_mantiene_ruta() {
        let routes = test_routes();
        let hist = history(2);
        let c = choose("¿y por qué?", &routes, &hist);
        assert_eq!(c.route_id, 2, "{}", c.reason);
    }

    #[test]
    fn sin_coincidencia_usa_ruta_por_defecto() {
        let routes = test_routes();
        let c = choose("xyzzy plugh quux", &routes, &[]);
        assert_eq!(c.route_id, 9);
        assert!(c.used_fallback);
    }

    #[test]
    fn acentos_y_mayusculas() {
        let routes = test_routes();
        let c = choose("PROGRAMACIÓN en RuSt", &routes, &[]);
        assert_eq!(c.route_id, 1);
    }

    #[test]
    fn sin_rutas_no_panica() {
        let c = choose("hola", &[], &[]);
        assert_eq!(c.route_id, 0);
    }
}
