//! Cola de feedback del usuario (FEEDBACK_SPEC.md §2).
//!
//! Cabrita **no entrena**: sólo escribe pares auditables en
//! `feedback_queue.jsonl` (una línea JSON por registro) para que el entrenador
//! de Julia los consuma en su próximo ciclo. Cada línea es independiente, así
//! que un consumo por streaming no necesita leer todo el archivo.
//!
//! Registros:
//! * `si` — respuesta correcta (con `correction` si el usuario dio la buena);
//! * `no` — respuesta equivocada;
//! * `up` — ruta correcta, pesos equivocados (refuerza router, penaliza pesos);
//! * `ent_my_pair` — par pregunta/respuesta de un `/ent-my`;
//! * `ent_my_meta` — resumen del archivo encolado (nº de pares y tamaño).
//!
//! Formato aceptado por `/ent-my` (decisiones documentadas, el spec lo
//! referenciaba pero no lo definía):
//!
//! ```text
//! Q: ¿qué es un cable en TOJI?
//! R: una conexión entre neuronas que cruza sectores.
//!
//! P: segunda pregunta
//! R: segunda respuesta
//! ```
//!
//! Marcadores válidos: `Q:`/`P:`/`PREGUNTA:` y `R:`/`A:`/`RESPUESTA:`
//! (sin distinguir mayúsculas ni acentos finales). También se acepta un archivo
//! `.json`/`.jsonl` con `{question,q,prompt,input}` y `{answer,a,response,output}`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::util::{iso8601, now_unix};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackKind {
    Si,
    No,
    Up,
    EntMyPair,
    EntMyMeta,
}

impl FeedbackKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            FeedbackKind::Si => "si",
            FeedbackKind::No => "no",
            FeedbackKind::Up => "up",
            FeedbackKind::EntMyPair => "ent_my_pair",
            FeedbackKind::EntMyMeta => "ent_my_meta",
        }
    }
}

/// Un registro de la cola. Campos estables y autodescriptivos.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRecord {
    pub schema: u32,
    pub kind: FeedbackKind,
    pub ts_unix: u64,
    pub ts_iso: String,
    pub route_id: u32,
    pub route_path: String,
    pub user_text: String,
    pub iris_text: String,
    pub correction: Option<String>,
    pub tokens_in: usize,
    pub tokens_out: usize,
    pub engine: String,
    pub model_version: String,
    pub source_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

/// Datos del turno que se adjuntan al feedback.
#[derive(Debug, Clone)]
pub struct TurnSnapshot {
    pub user_text: String,
    pub iris_text: String,
    pub route_id: u32,
    pub route_path: String,
    pub tokens_in: usize,
    pub tokens_out: usize,
    pub engine: String,
    pub model_version: String,
}

impl TurnSnapshot {
    pub fn record(&self, kind: FeedbackKind, correction: Option<String>) -> FeedbackRecord {
        FeedbackRecord {
            schema: 1,
            kind,
            ts_unix: now_unix(),
            ts_iso: iso8601(now_unix()),
            route_id: self.route_id,
            route_path: self.route_path.clone(),
            user_text: self.user_text.clone(),
            iris_text: self.iris_text.clone(),
            correction,
            tokens_in: self.tokens_in,
            tokens_out: self.tokens_out,
            engine: self.engine.clone(),
            model_version: self.model_version.clone(),
            source_file: None,
            extra: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeedbackQueue {
    path: PathBuf,
}

impl FeedbackQueue {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        FeedbackQueue { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Añade un registro al final. Crea el archivo si no existe.
    pub fn enqueue(&self, rec: &FeedbackRecord) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let mut line = serde_json::to_string(rec).context("no se pudo serializar el registro")?;
        line.push('\n');
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("no se pudo abrir la cola {}", self.path.display()))?;
        f.write_all(line.as_bytes())
            .with_context(|| format!("no se pudo escribir en {}", self.path.display()))?;
        Ok(())
    }

    /// Registros pendientes (líneas no vacías).
    pub fn pending(&self) -> usize {
        match std::fs::read_to_string(&self.path) {
            Ok(s) => s.lines().filter(|l| !l.trim().is_empty()).count(),
            Err(_) => 0,
        }
    }

    /// Lee todos los registros (para auditoría / tests).
    pub fn read_all(&self) -> Result<Vec<FeedbackRecord>> {
        let raw = std::fs::read_to_string(&self.path)
            .with_context(|| format!("no se pudo leer {}", self.path.display()))?;
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<FeedbackRecord>(l).context("registro de feedback corrupto"))
            .collect()
    }
}

/// Un par pregunta/respuesta listo para encolar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QaPair {
    pub question: String,
    pub answer: String,
    pub line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    Question,
    Answer,
}

fn marker_of(line: &str) -> Option<Marker> {
    let t = line.trim_start();
    let upper = t.to_uppercase();
    for q in ["Q:", "P:", "PREGUNTA:"] {
        if upper.starts_with(q) {
            return Some(Marker::Question);
        }
    }
    for a in ["R:", "A:", "RESPUESTA:"] {
        if upper.starts_with(a) {
            return Some(Marker::Answer);
        }
    }
    None
}

fn strip_marker(line: &str) -> String {
    let t = line.trim_start();
    let upper = t.to_uppercase();
    for q in ["Q:", "P:", "PREGUNTA:", "R:", "A:", "RESPUESTA:"] {
        if upper.starts_with(q) {
            return t[q.len()..].trim_start().to_string();
        }
    }
    t.to_string()
}

/// Parsea el archivo de `/ent-my`.
pub fn parse_ent_my(path: impl AsRef<Path>) -> Result<Vec<QaPair>> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("no se pudo leer {}", path.display()))?;
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(&raw).to_string();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "json" || ext == "jsonl" {
        return parse_json_pairs(&raw);
    }
    parse_text_pairs(&raw)
}

fn parse_text_pairs(raw: &str) -> Result<Vec<QaPair>> {
    let mut pairs: Vec<QaPair> = Vec::new();
    let mut question: Option<(String, usize)> = None;
    let mut answer: Option<String> = None;
    let mut current_is_answer = false;

    let flush = |pairs: &mut Vec<QaPair>,
                 question: &mut Option<(String, usize)>,
                 answer: &mut Option<String>|
     -> Result<()> {
        if let Some((q, line)) = question.take() {
            let a = answer.take().unwrap_or_default();
            if q.trim().is_empty() {
                bail!("pregunta vacía en la línea {line} de /ent-my");
            }
            if a.trim().is_empty() {
                bail!("la pregunta de la línea {line} no tiene respuesta (marcador R:)");
            }
            pairs.push(QaPair { question: q.trim().to_string(), answer: a.trim().to_string(), line });
        } else {
            answer.take();
        }
        Ok(())
    };

    for (idx, line) in raw.lines().enumerate() {
        let lineno = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "###" || trimmed.starts_with("//") {
            if trimmed.is_empty() {
                flush(&mut pairs, &mut question, &mut answer)?;
            }
            continue;
        }
        if trimmed.starts_with('#') && marker_of(trimmed).is_none() {
            continue; // comentario
        }
        match marker_of(line) {
            Some(Marker::Question) => {
                flush(&mut pairs, &mut question, &mut answer)?;
                question = Some((strip_marker(line), lineno));
                current_is_answer = false;
            }
            Some(Marker::Answer) => {
                if question.is_none() {
                    bail!("respuesta sin pregunta en la línea {lineno} de /ent-my");
                }
                answer = Some(strip_marker(line));
                current_is_answer = true;
            }
            None => {
                // Continuación multilínea del bloque actual.
                if current_is_answer {
                    if let Some(a) = answer.as_mut() {
                        a.push('\n');
                        a.push_str(trimmed);
                    }
                } else if let Some((q, _)) = question.as_mut() {
                    q.push('\n');
                    q.push_str(trimmed);
                } else {
                    bail!(
                        "línea {lineno} sin marcador Q:/R: (el archivo debe usar Q:/R: o ser .json/.jsonl)"
                    );
                }
            }
        }
    }
    flush(&mut pairs, &mut question, &mut answer)?;

    if pairs.is_empty() {
        bail!("no se encontró ningún par Q:/R: en el archivo");
    }
    Ok(pairs)
}

fn parse_json_pairs(raw: &str) -> Result<Vec<QaPair>> {
    let mut out = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("JSON inválido en la línea {}", idx + 1))?;
        let get = |keys: &[&str]| -> Option<String> {
            keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_str()).map(|s| s.to_string()))
        };
        let q = get(&["question", "q", "prompt", "input", "entrada"]);
        let a = get(&["answer", "a", "response", "output", "respuesta"]);
        match (q, a) {
            (Some(q), Some(a)) if !q.trim().is_empty() && !a.trim().is_empty() => {
                out.push(QaPair { question: q, answer: a, line: idx + 1 })
            }
            _ => bail!("línea {} no tiene pregunta y respuesta válidas", idx + 1),
        }
    }
    if out.is_empty() {
        bail!("el archivo JSON no contiene pares");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("cabrita_feedback");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn encola_jsonl_valido() {
        let path = tmp("queue.jsonl");
        let _ = std::fs::remove_file(&path);
        let q = FeedbackQueue::new(&path);
        let snap = TurnSnapshot {
            user_text: "2+2".into(),
            iris_text: "4".into(),
            route_id: 2,
            route_path: "mundo › ciencia › matematicas".into(),
            tokens_in: 3,
            tokens_out: 1,
            engine: "template".into(),
            model_version: "UKJ v1 (p4-one)".into(),
        };
        q.enqueue(&snap.record(FeedbackKind::Si, Some("cuatro".into()))).unwrap();
        q.enqueue(&snap.record(FeedbackKind::No, None)).unwrap();
        assert_eq!(q.pending(), 2);
        let recs = q.read_all().unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].kind, FeedbackKind::Si);
        assert_eq!(recs[0].correction.as_deref(), Some("cuatro"));
        assert_eq!(recs[1].kind, FeedbackKind::No);
        assert_eq!(recs[0].schema, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parsea_txt_estructurado() {
        let p = tmp("ent_my.txt");
        std::fs::write(
            &p,
            "\u{feff}# comentario\nQ: ¿qué es TOJI?\nR: la memoria viva de Iris,\n   por sectores.\n\nP: ¿y UKJ?\nA: el contenedor del conocimiento.\n",
        )
        .unwrap();
        let pairs = parse_ent_my(&p).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].question, "¿qué es TOJI?");
        assert!(pairs[0].answer.contains("por sectores"));
        assert_eq!(pairs[1].answer, "el contenedor del conocimiento.");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rechaza_archivo_sin_marcadores() {
        let p = tmp("plano.txt");
        std::fs::write(&p, "esto es texto suelto\nsin estructura\n").unwrap();
        let err = parse_ent_my(&p).unwrap_err().to_string();
        assert!(err.contains("sin marcador"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rechaza_pregunta_sin_respuesta() {
        let p = tmp("incompleto.txt");
        std::fs::write(&p, "Q: sola\n\n").unwrap();
        let err = parse_ent_my(&p).unwrap_err().to_string();
        assert!(err.contains("no tiene respuesta"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn parsea_jsonl() {
        let p = tmp("pares.jsonl");
        std::fs::write(
            &p,
            "{\"question\":\"q1\",\"answer\":\"a1\"}\n{\"prompt\":\"q2\",\"output\":\"a2\"}\n",
        )
        .unwrap();
        let pairs = parse_ent_my(&p).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[1].question, "q2");
        let _ = std::fs::remove_file(&p);
    }
}
