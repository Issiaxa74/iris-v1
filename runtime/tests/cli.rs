//! Pruebas de integración del binario: CLI, REPL, cola de feedback y TOJI.
//!
//! Cada prueba trabaja sobre una copia de los assets en un directorio temporal,
//! de modo que el `test.ukj` del repositorio nunca se modifica (cabrita no
//! reescribe el modelo: el estrés vive en un sidecar aparte).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const BIN: &str = env!("CARGO_BIN_EXE_cabrita");

fn asset(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(name)
}

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("cabrita-it-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("crear directorio temporal");
    dir
}

struct Fixture {
    dir: PathBuf,
    model: PathBuf,
    tok: PathBuf,
    mem: PathBuf,
    queue: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let dir = temp_dir(tag);
        let model = dir.join("test.ukj");
        let tok = dir.join("iris.tok");
        std::fs::copy(asset("test.ukj"), &model).expect("copiar test.ukj");
        std::fs::copy(asset("iris.tok"), &tok).expect("copiar iris.tok");
        Fixture {
            mem: dir.join("memory"),
            queue: dir.join("feedback_queue.jsonl"),
            dir,
            model,
            tok,
        }
    }

    fn base_args(&self) -> Vec<String> {
        vec![
            "--model".into(),
            self.model.display().to_string(),
            "--tok".into(),
            self.tok.display().to_string(),
            "--mem-dir".into(),
            self.mem.display().to_string(),
            "--queue".into(),
            self.queue.display().to_string(),
            "--engine".into(),
            "stats".into(),
        ]
    }

    fn run(&self, extra: &[&str], stdin: Option<&str>) -> Output {
        let mut cmd = Command::new(BIN);
        cmd.args(self.base_args());
        cmd.args(extra);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        if stdin.is_some() {
            cmd.stdin(Stdio::piped());
        } else {
            cmd.stdin(Stdio::null());
        }
        let mut child = cmd.spawn().expect("arrancar cabrita");
        if let Some(script) = stdin {
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(script.as_bytes())
                .expect("escribir stdin");
        }
        child.wait_with_output().expect("esperar a cabrita")
    }

    fn sidecar(&self) -> PathBuf {
        PathBuf::from(format!("{}.stress.json", self.model.display()))
    }

    fn records(&self) -> Vec<serde_json::Value> {
        let raw = std::fs::read_to_string(&self.queue).unwrap_or_default();
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("registro JSONL válido"))
            .collect()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn pregunta_unica_enruta_y_deja_el_ukj_intacto() {
    let fx = Fixture::new("ask");
    let before = std::fs::read(&fx.model).expect("leer modelo");

    let out = fx.run(&["--ask", "quiero programar una API en Rust"], None);
    let stdout = text(&out.stdout);
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");

    assert!(
        stdout.contains("informatica › programacion"),
        "stdout: {stdout}"
    );
    assert!(
        stderr.contains("ruta: informatica › programacion"),
        "stderr: {stderr}"
    );
    assert!(stdout.contains("Tensores: dim=64"), "stdout: {stdout}");
    assert!(
        fx.mem.join("informatica.mem").exists(),
        "sector TOJI informatica"
    );
    assert!(fx.sidecar().exists(), "sidecar de estrés");
    assert!(
        !fx.queue.exists(),
        "una pregunta suelta no debe tocar la cola"
    );

    let after = std::fs::read(&fx.model).expect("releer modelo");
    assert_eq!(before, after, "cabrita nunca debe reescribir el UKJ");
}

#[test]
fn repl_comandos_escriben_cola_memoria_y_estres() {
    let fx = Fixture::new("repl");

    let pairs = fx.dir.join("ent_my.txt");
    std::fs::write(
        &pairs,
        "Q: ¿qué es una integral?\nR: Un acumulado continuo.\n\nQ: ¿qué es un límite?\nR: El valor al que tiende una sucesión.\n",
    )
    .expect("escribir pares");

    let script = format!(
        "/tok ¿en qué estado está el mundo?\n\
         explícame el cálculo integral\n\
         /si la integral acumula área\n\
         /no\n\
         /up\n\
         /ent-my {}\n\
         /stress 2 5\n\
         /merge informatica matematicas\n\
         /queue 10\n\
         /info\n\
         /routes\n\
         /mem\n\
         /exit\n",
        pairs.display()
    );

    let out = fx.run(&[], Some(&script));
    let stdout = text(&out.stdout);
    assert!(out.status.success(), "stdout: {stdout}");

    assert!(stdout.contains("ida y vuelta: OK"), "stdout: {stdout}");
    assert!(stdout.contains("umbral de estrés = 5"), "stdout: {stdout}");
    assert!(
        stdout.contains("fusionado «matematicas» → «informatica»"),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("candidatas :"), "stdout: {stdout}");

    let recs = fx.records();
    let kinds: Vec<&str> = recs
        .iter()
        .map(|r| r["kind"].as_str().expect("kind textual"))
        .collect();
    assert!(kinds.contains(&"si"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"no"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"up"), "kinds: {kinds:?}");
    assert_eq!(kinds.iter().filter(|k| **k == "ent_my_pair").count(), 2);
    assert_eq!(kinds.iter().filter(|k| **k == "ent_my_meta").count(), 1);

    let pair = recs
        .iter()
        .find(|r| r["kind"] == "ent_my_pair")
        .expect("par encolado");
    assert_eq!(pair["user_text"], "¿qué es una integral?");
    assert_eq!(pair["iris_text"], "Un acumulado continuo.");
    assert_eq!(pair["schema"], 1);

    assert!(fx.mem.join("informatica.mem").exists());
    assert!(fx.mem.join("matematicas.mem").exists());
    assert!(fx.sidecar().exists());

    let sidecar = std::fs::read_to_string(fx.sidecar()).expect("leer sidecar");
    let json: serde_json::Value = serde_json::from_str(&sidecar).expect("sidecar JSON");
    assert_eq!(
        json["overrides"]["2"], 5,
        "umbral por ruta persistido: {sidecar}"
    );
}

#[test]
fn no_reescribe_el_modelo_ni_sin_escribir_memoria() {
    let fx = Fixture::new("readonly");
    let before = std::fs::read(&fx.model).expect("leer modelo");

    let out = fx.run(
        &["--no-mem-write", "--ask", "resuelve la integral de x"],
        None,
    );
    assert!(out.status.success(), "{}", text(&out.stderr));

    assert!(
        !fx.mem.exists(),
        "sin --no-mem-write no se crea el directorio"
    );
    assert!(!fx.sidecar().exists(), "el sidecar tampoco se escribe");
    assert_eq!(
        before,
        std::fs::read(&fx.model).expect("releer"),
        "modelo intacto"
    );
}

#[test]
fn modelo_ausente_da_un_error_claro() {
    let fx = Fixture::new("missing");
    let missing = fx.dir.join("no_existe.ukj");
    let out = Command::new(BIN)
        .args([
            "--model",
            missing.to_str().unwrap(),
            "--tok",
            fx.tok.to_str().unwrap(),
            "--mem-dir",
            fx.mem.to_str().unwrap(),
            "--queue",
            fx.queue.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .output()
        .expect("arrancar cabrita");

    assert!(!out.status.success());
    let stderr = text(&out.stderr);
    assert!(
        stderr.contains("No se pudo abrir el modelo"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("--model"), "stderr: {stderr}");
}

#[test]
fn ayuda_lista_los_comandos_nuevos() {
    let out = Command::new(BIN)
        .arg("--help")
        .stdin(Stdio::null())
        .output()
        .expect("arrancar cabrita");
    let stdout = text(&out.stdout);
    assert!(out.status.success());
    for needle in ["/ent-my", "/merge", "/tok", "/queue", "--no-mem-write"] {
        assert!(stdout.contains(needle), "falta {needle} en --help");
    }
}

#[test]
fn motor_transformer_genera_texto_desde_el_trunk() {
    let fx = Fixture::new("transformer");
    let tiny = fx.dir.join("tiny.ukj");
    std::fs::copy(asset("tiny.ukj"), &tiny).expect("copiar tiny.ukj");
    std::fs::copy(asset("tiny.ukj.json"), fx.dir.join("tiny.ukj.json")).expect("copiar manifiesto");

    let out = Command::new(BIN)
        .args([
            "--model",
            tiny.to_str().unwrap(),
            "--tok",
            fx.tok.to_str().unwrap(),
            "--mem-dir",
            fx.mem.to_str().unwrap(),
            "--queue",
            fx.queue.to_str().unwrap(),
            "--engine",
            "transformer",
            "--max-new",
            "6",
            "--temp",
            "0",
            "--seed",
            "1",
            "--no-mem-write",
            "--ask",
            "hola mundo",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("arrancar cabrita");

    let stdout = text(&out.stdout);
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert!(stdout.contains("iris> hel"), "stdout: {stdout}");
    assert!(stderr.contains("transformer"), "stderr: {stderr}");
    assert!(stderr.contains("trunk 2 capas"), "stderr: {stderr}");
    assert!(!fx.mem.exists(), "--no-mem-write no debe crear memoria");
}

#[test]
fn transformer_sin_manifiesto_explica_el_export() {
    let fx = Fixture::new("sin-manifiesto");
    let out = fx.run(&["--engine", "transformer", "--ask", "hola"], None);
    assert!(!out.status.success());
    let stderr = text(&out.stderr);
    assert!(stderr.contains("manifiesto"), "stderr: {stderr}");
    assert!(stderr.contains("export_ukj.jl"), "stderr: {stderr}");
}
