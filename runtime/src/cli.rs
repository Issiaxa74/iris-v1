//! Argumentos de línea de comandos. Sin dependencias: parser manual y explícito.

use std::path::PathBuf;

use anyhow::{bail, Result};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub struct Config {
    pub model: PathBuf,
    pub tok: PathBuf,
    pub mem_dir: PathBuf,
    pub queue: PathBuf,
    pub engine: String,
    pub min_score: f32,
    pub stress_threshold: u64,
    pub stress_decay: f32,
    pub compress_mem: bool,
    pub write_memory: bool,
    pub ask: Option<String>,
    pub help: bool,
    pub version: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: default_asset("test.ukj"),
            tok: default_asset("iris.tok"),
            mem_dir: PathBuf::from("memory"),
            queue: PathBuf::from("feedback_queue.jsonl"),
            engine: "template".to_string(),
            min_score: 0.30,
            stress_threshold: 12,
            stress_decay: 0.90,
            compress_mem: false,
            write_memory: true,
            ask: None,
            help: false,
            version: false,
        }
    }
}

/// `assets/x` si existe en el directorio actual; si no, junto al ejecutable.
fn default_asset(name: &str) -> PathBuf {
    let local = PathBuf::from("assets").join(name);
    if local.exists() {
        return local;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let near = dir.join("assets").join(name);
            if near.exists() {
                return near;
            }
        }
    }
    local
}

pub fn usage() -> String {
    format!(
        "CABRITA {VERSION} — runtime de Iris (UKJ + IRISTOK + TOJI)

USO
  cabrita [opciones]
  cabrita --ask \"texto\"     una sola pregunta y sale

MODELO Y MEMORIA
  --model <ruta.ukj>       modelo UKJ (por defecto assets/test.ukj)
  --tok <ruta.tok>         tokenizer IRISTOK (por defecto assets/iris.tok)
  --mem-dir <dir>          sectores TOJI .mem (por defecto ./memory)
  --queue <ruta.jsonl>     cola de feedback (por defecto ./feedback_queue.jsonl)
  --no-mem-write           no escribe .mem ni sidecar de estrés (modo lectura)
  --compress-mem           guarda los .mem con zstd

COMPORTAMIENTO
  --engine template|stats  motor de respuesta (por defecto template)
  --min-score <f>          umbral del router provisional (por defecto 0.30)
  --stress-threshold <n>   activaciones para marcar estrés (por defecto 12)
  --stress-decay <f>       decaimiento por turno, 0..1 (por defecto 0.90)
  --ask <texto>            responde una vez por stdout y termina
  -h, --help               esta ayuda
  -V, --version            versión

COMANDOS DEL REPL
  /si [texto]   última respuesta correcta (texto opcional = la correcta)
  /no           última respuesta equivocada
  /up           ruta correcta, respuesta equivocada
  /ent-my <f>   encola pares Q:/R: de un .txt (o .json/.jsonl)
  /info         ruta activa, RAM, tokens, versión UKJ, cola
  /routes       tabla de rutas del modelo
  /stress       contadores de estrés por ruta
  /stress <r> <n>  fija el umbral de estrés de la ruta r
  /mem          sectores TOJI y su tamaño
  /merge <a> <b>   fusiona dos sectores TOJI
  /tok <texto>  tokens IRISTOK y comprobación de ida y vuelta
  /queue [n]    últimos n registros de feedback
  /help         comandos
  /exit         salir (también Ctrl+D)

VARIABLES DE ENTORNO
  CABRITA_MODEL, CABRITA_TOK, CABRITA_MEM_DIR, CABRITA_QUEUE, CABRITA_ENGINE
"
    )
}

pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Config> {
    let mut cfg = Config::default();
    if let Ok(v) = std::env::var("CABRITA_MODEL") {
        cfg.model = PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("CABRITA_TOK") {
        cfg.tok = PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("CABRITA_MEM_DIR") {
        cfg.mem_dir = PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("CABRITA_QUEUE") {
        cfg.queue = PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("CABRITA_ENGINE") {
        cfg.engine = v;
    }

    let mut it = args.into_iter().peekable();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String> {
            it.next().ok_or_else(|| anyhow::anyhow!("falta el valor de {name}"))
        };
        match arg.as_str() {
            "-h" | "--help" => cfg.help = true,
            "-V" | "--version" => cfg.version = true,
            "--model" => cfg.model = PathBuf::from(value("--model")?),
            "--tok" => cfg.tok = PathBuf::from(value("--tok")?),
            "--mem-dir" => cfg.mem_dir = PathBuf::from(value("--mem-dir")?),
            "--queue" => cfg.queue = PathBuf::from(value("--queue")?),
            "--engine" => cfg.engine = value("--engine")?,
            "--min-score" => {
                cfg.min_score = value("--min-score")?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--min-score debe ser un número"))?
            }
            "--stress-threshold" => {
                cfg.stress_threshold = value("--stress-threshold")?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--stress-threshold debe ser un entero"))?
            }
            "--stress-decay" => {
                cfg.stress_decay = value("--stress-decay")?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--stress-decay debe ser un número 0..1"))?
            }
            "--compress-mem" => cfg.compress_mem = true,
            "--no-mem-write" | "--read-only" => cfg.write_memory = false,
            "--ask" => cfg.ask = Some(value("--ask")?),
            other if other.starts_with("--") => bail!("opción desconocida: {other} (usa --help)"),
            other => {
                // Texto posicional = pregunta única.
                if cfg.ask.is_none() {
                    cfg.ask = Some(other.to_string());
                } else {
                    bail!("argumento inesperado: {other}");
                }
            }
        }
    }
    if !(0.0..=1.0).contains(&cfg.stress_decay) {
        bail!("--stress-decay debe estar entre 0 y 1");
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_vec(v: &[&str]) -> Result<Config> {
        parse(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn valores_por_defecto() {
        let c = parse_vec(&[]).unwrap();
        assert_eq!(c.engine, "template");
        assert!(c.write_memory);
        assert!((c.min_score - 0.30).abs() < 1e-6);
    }

    #[test]
    fn opciones_y_pregunta() {
        let c = parse_vec(&["--engine", "stats", "--ask", "hola", "--no-mem-write"]).unwrap();
        assert_eq!(c.engine, "stats");
        assert_eq!(c.ask.as_deref(), Some("hola"));
        assert!(!c.write_memory);
    }

    #[test]
    fn rechaza_desconocida_y_valores_malos() {
        assert!(parse_vec(&["--nope"]).is_err());
        assert!(parse_vec(&["--min-score", "alto"]).is_err());
        assert!(parse_vec(&["--stress-decay", "2"]).is_err());
        assert!(parse_vec(&["--model"]).is_err());
    }
}
