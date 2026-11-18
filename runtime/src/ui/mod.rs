//! Reloj de terminal: REPL conversacional, streaming token a token y comandos.

use std::io::{BufRead, Write};
use std::time::Instant;

use anyhow::Result;

use crate::cli::{Config, VERSION};
use crate::engine::{self, Engine, EngineContext};
use crate::feedback::{parse_ent_my, FeedbackKind, FeedbackQueue, TurnSnapshot};
use crate::router::keyword::KeywordRouter;
use crate::router::{HistoryTurn, RouteChoice, RouteRequest, Router};
use crate::toji::{MemStore, MemoryUpdater, MemoryView, Sector};
use crate::tok::{StreamDecoder, Tokenizer};
use crate::ukj::stress::sidecar_path;
use crate::ukj::UkjModel;
use crate::util::{human_bytes, process_rss, sector_name};

const HISTORY_MAX: usize = 8;

pub struct App {
    cfg: Config,
    pub model: UkjModel,
    pub tok: Tokenizer,
    router: KeywordRouter,
    engine: Box<dyn Engine>,
    stress: crate::ukj::StressTable,
    store: MemStore,
    queue: FeedbackQueue,
    history: Vec<HistoryTurn>,
    last: Option<TurnSnapshot>,
    last_route: Option<u32>,
    last_choice: Option<RouteChoice>,
    updater: MemoryUpdater,
}

impl App {
    pub fn new(cfg: Config) -> Result<Self> {
        let model = UkjModel::open(&cfg.model).map_err(|e| {
            anyhow::anyhow!(
                "{e}\n\nNo se pudo abrir el modelo «{}».\n\
                 Copia un UKJ (por ejemplo assets/test.ukj) y pásalo con --model <ruta>, \
                 o define CABRITA_MODEL.",
                cfg.model.display()
            )
        })?;
        let tok = Tokenizer::load(&cfg.tok).map_err(|e| {
            anyhow::anyhow!(
                "{e}\n\nNo se pudo abrir el tokenizer «{}» (formato IRISTOK, iris.tok).",
                cfg.tok.display()
            )
        })?;
        let engine = engine::from_name(&cfg.engine)?;
        let stress = crate::ukj::StressTable::load(&cfg.model, cfg.stress_threshold, cfg.stress_decay)?;
        let store = MemStore::new(&cfg.mem_dir, cfg.compress_mem);
        let queue = FeedbackQueue::new(&cfg.queue);
        let router = KeywordRouter::new(cfg.min_score, 0.6);
        Ok(App {
            cfg,
            model,
            tok,
            router,
            engine,
            stress,
            store,
            queue,
            history: Vec::new(),
            last: None,
            last_route: None,
            last_choice: None,
            updater: MemoryUpdater::default(),
        })
    }

    pub fn model_version(&self) -> String {
        self.model.header.version_label()
    }

    pub fn banner(&self) {
        println!("CABRITA {VERSION} — runtime de Iris");
        println!(
            "modelo    : {} · {} rutas · {}",
            self.model.path().display(),
            self.model.header.n_rutas,
            human_bytes(self.model.size())
        );
        println!(
            "tokenizer : {} · {} tokens ({} fusiones)",
            self.cfg.tok.display(),
            self.tok.vocab_size(),
            self.tok.n_merges()
        );
        println!(
            "memoria   : {} · feedback: {}",
            self.store.dir().display(),
            self.queue.path().display()
        );
        println!("motor     : {} · router: {}", self.engine.name(), self.router.name());
        println!("escribe /help para los comandos, /exit para salir.\n");
    }

    pub fn run_interactive(&mut self) -> Result<()> {
        self.banner();
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            print!("tú> ");
            std::io::stdout().flush().ok();
            line.clear();
            let n = stdin.lock().read_line(&mut line)?;
            if n == 0 {
                println!();
                break;
            }
            let text = line.trim().to_string();
            if text.is_empty() {
                continue;
            }
            if text.starts_with('/') {
                if self.handle_command(&text)? {
                    break;
                }
            } else if let Err(e) = self.turn(&text, false) {
                eprintln!("error en el turno: {e:#}");
            }
        }
        Ok(())
    }

    pub fn ask_once(&mut self, text: &str) -> Result<()> {
        self.turn(text, true)
    }

    /// Un turno completo: tokenizar → router → ruta → TOJI → motor → streaming.
    pub fn turn(&mut self, user_text: &str, meta_to_stderr: bool) -> Result<()> {
        let started = Instant::now();
        let tokens = self.tok.encode(user_text);
        let choice = self.router.choose(&RouteRequest {
            text: user_text,
            tokens: &tokens,
            history: &self.history,
            routes: &self.model.routes,
        });

        let idx = self
            .model
            .route_index(choice.route_id)
            .ok_or_else(|| anyhow::anyhow!("el router devolvió una ruta inexistente ({})", choice.route_id))?;
        let route = self.model.route(idx)?.clone();

        let mut notes: Vec<String> = Vec::new();
        let tensors = match self.model.load_route(idx) {
            Ok(t) => Some(t),
            Err(e) => {
                notes.push(format!("aviso: no se pudieron cargar los tensores: {e}"));
                None
            }
        };

        self.stress.register(route.id, tokens.len().max(1) as u64);
        let stressed = self.stress.is_stressed(route.id);
        let stress_count = self.stress.activations(route.id);
        let stress_threshold = self.stress.threshold_for(route.id);

        let sector_key = sector_name(&route.sector());
        let mut sector: Sector = match self.store.load(&sector_key) {
            Ok(s) => s,
            Err(e) => {
                notes.push(format!("aviso: {} (se empieza un sector vacío)", e));
                Sector::new(sector_key.clone())
            }
        };
        let memory: MemoryView = self.updater.absorb(&mut sector, user_text);

        let ctx = EngineContext {
            route: &route,
            tensors: tensors.as_ref(),
            memory: &memory,
            history: &self.history,
            user_text,
            tokens_in: tokens.len(),
            model_version: &self.model_version(),
            route_bytes: self.model.route_bytes(&route),
            stressed,
            stress_count,
            stress_threshold,
        };
        let answer = self.engine.respond(&ctx)?;

        // Streaming token a token (no por palabras: por tokens IRISTOK).
        print!("iris> ");
        std::io::stdout().flush().ok();
        let tokens_out = self.stream(&answer.text)?;
        if !answer.text.ends_with('\n') {
            println!();
        }

        let elapsed = started.elapsed();
        let mut meta = format!(
            "  ⤷ ruta: {} · {:.2} · {} tok in / {} tok out · {} · {} ms{}",
            choice.label,
            choice.score,
            tokens.len(),
            tokens_out,
            self.engine.name(),
            elapsed.as_millis(),
            if choice.used_fallback { " · fallback" } else { "" }
        );
        if self.stress.is_stressed(route.id) {
            meta.push_str(" · ESTRÉS");
        }
        for n in &answer.notes {
            meta.push_str(" · ");
            meta.push_str(n);
        }
        if meta_to_stderr {
            eprintln!("{meta}");
        } else {
            println!("{meta}");
        }
        for n in &notes {
            if meta_to_stderr {
                eprintln!("  {n}");
            } else {
                println!("  {n}");
            }
        }

        let snapshot = TurnSnapshot {
            user_text: user_text.to_string(),
            iris_text: answer.text.clone(),
            route_id: route.id,
            route_path: route.label(),
            tokens_in: tokens.len(),
            tokens_out,
            engine: self.engine.name().to_string(),
            model_version: self.model_version(),
        };

        self.history.push(HistoryTurn {
            user: user_text.to_string(),
            assistant: answer.text.clone(),
            route_id: route.id,
            route_label: route.label(),
            tokens_in: tokens.len(),
            tokens_out,
        });
        if self.history.len() > HISTORY_MAX {
            self.history.remove(0);
        }
        self.last = Some(snapshot);
        self.last_route = Some(route.id);
        self.last_choice = Some(choice);
        self.stress.decay_cycle();

        if self.cfg.write_memory {
            if let Err(e) = self.store.save(&sector) {
                eprintln!("aviso: no se pudo guardar el sector TOJI: {e:#}");
            }
            if let Err(e) = self.stress.save() {
                eprintln!("aviso: no se pudo guardar el sidecar de estrés: {e:#}");
            }
        }
        Ok(())
    }

    /// Imprime el texto token a token; devuelve cuántos tokens se emitieron.
    fn stream(&mut self, text: &str) -> Result<usize> {
        let ids = self.tok.encode(text);
        let mut dec = StreamDecoder::new();
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        for id in &ids {
            let chunk = dec.push_token(&self.tok, *id)?;
            if !chunk.is_empty() {
                out.write_all(chunk.as_bytes())?;
                out.flush().ok();
            }
        }
        let rest = dec.flush();
        if !rest.is_empty() {
            out.write_all(rest.as_bytes())?;
        }
        out.flush().ok();
        Ok(ids.len())
    }

    /// Procesa un comando. Devuelve `true` si hay que salir.
    pub fn handle_command(&mut self, line: &str) -> Result<bool> {
        let (cmd, rest) = match line.split_once(char::is_whitespace) {
            Some((c, r)) => (c, r.trim()),
            None => (line, ""),
        };
        match cmd {
            "/exit" | "/salir" | "/quit" | "/q" => {
                println!("hasta luego.");
                return Ok(true);
            }
            "/help" | "/ayuda" | "/?" => self.print_help(),
            "/info" => self.print_info(),
            "/routes" | "/rutas" => self.print_routes(),
            "/stress" | "/estres" => {
                if rest.is_empty() {
                    self.print_stress();
                } else {
                    self.set_stress(rest)?;
                }
            }
            "/mem" | "/memoria" => self.print_mem(),
            "/merge" | "/fusionar" => self.cmd_merge(rest)?,
            "/tok" | "/tokens" => self.cmd_tok(rest)?,
            "/queue" | "/cola" => self.cmd_queue(rest),
            "/si" => self.feedback(FeedbackKind::Si, rest)?,
            "/no" => self.feedback(FeedbackKind::No, "")?,
            "/up" => self.feedback(FeedbackKind::Up, "")?,
            "/ent-my" | "/ent_my" => self.ent_my(rest)?,
            other => println!("comando desconocido: {other} (prueba /help)"),
        }
        Ok(false)
    }

    fn print_help(&self) {
        println!(
            "comandos:
  /si [texto]   la última respuesta fue correcta (texto = la respuesta correcta)
  /no           la última respuesta fue equivocada
  /up           la ruta era correcta pero la respuesta no
  /ent-my <f>   encola pares Q:/R: (o .json/.jsonl) para el entrenador
  /info         ruta activa, RAM, tokens, versión UKJ y cola de feedback
  /routes       rutas del modelo con su estrés
  /stress       contador de activaciones por ruta
  /stress <r> <n>  fija el umbral de estrés de la ruta r en n
  /mem          sectores TOJI en disco
  /merge <a> <b>   fusiona dos sectores TOJI distintos en el de menor nombre
  /tok <texto>  tokens IRISTOK del texto y verificación de ida y vuelta
  /queue [n]    últimos n registros de la cola de feedback (por defecto 5)
  /help         esta ayuda
  /exit         salir"
        );
    }

    fn feedback(&mut self, kind: FeedbackKind, correction: &str) -> Result<()> {
        let snapshot = match self.last.as_ref() {
            Some(s) => s.clone(),
            None => {
                println!("no hay un turno previo sobre el que dar feedback.");
                return Ok(());
            }
        };
        let correction = if correction.trim().is_empty() {
            None
        } else {
            Some(correction.trim().to_string())
        };
        let rec = snapshot.record(kind, correction);
        self.queue.enqueue(&rec)?;
        let extra = match kind {
            FeedbackKind::Si => "ruta y pesos reforzados",
            FeedbackKind::No => "ruta y pesos penalizados",
            FeedbackKind::Up => "router reforzado, pesos penalizados",
            _ => "",
        };
        println!(
            "anotado: {} ({extra}) → {} · {} pendientes",
            rec.kind.as_str(),
            self.queue.path().display(),
            self.queue.pending()
        );
        Ok(())
    }

    fn ent_my(&mut self, arg: &str) -> Result<()> {
        if arg.is_empty() {
            println!("uso: /ent-my <archivo.txt|archivo.jsonl>");
            return Ok(());
        }
        let path_raw = arg.trim().trim_matches(|c| c == '"' || c == '\'');
        let path = std::path::PathBuf::from(path_raw);
        let pairs = parse_ent_my(&path)?;
        let base = self.snapshot_or_empty();
        for p in &pairs {
            let mut rec = base.record(FeedbackKind::EntMyPair, None);
            rec.user_text = p.question.clone();
            rec.iris_text = p.answer.clone();
            rec.source_file = Some(path.display().to_string());
            rec.extra = Some(serde_json::json!({ "line": p.line }));
            self.queue.enqueue(&rec)?;
        }
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let mut meta = base.record(FeedbackKind::EntMyMeta, None);
        meta.source_file = Some(path.display().to_string());
        meta.extra = Some(serde_json::json!({ "pairs": pairs.len(), "bytes": size }));
        self.queue.enqueue(&meta)?;
        println!(
            "encolados {} pares de {} → {} · {} registros pendientes",
            pairs.len(),
            path.display(),
            self.queue.path().display(),
            self.queue.pending()
        );
        Ok(())
    }

    fn snapshot_or_empty(&self) -> TurnSnapshot {
        match self.last.clone() {
            Some(s) => s,
            None => TurnSnapshot {
                user_text: String::new(),
                iris_text: String::new(),
                route_id: 0,
                route_path: String::new(),
                tokens_in: 0,
                tokens_out: 0,
                engine: self.engine.name().to_string(),
                model_version: self.model_version(),
            },
        }
    }

    fn print_info(&mut self) {
        let active = self
            .last_route
            .and_then(|id| self.model.route_index(id))
            .map(|i| self.model.routes[i].clone());
        let integrity = self.model.integrity();
        let header = &self.model.header;

        println!("modelo     : {}", self.model.path().display());
        println!(
            "versión    : UKJ v{} ({}) · {} rutas · flags {}{}",
            header.version,
            header.quant,
            header.n_rutas,
            header.flags,
            if header.tokenizer_embedded() { " (tokenizer embebido)" } else { "" }
        );
        println!(
            "integridad : {} · checksum guardado {:#x} · calculado {:#x}",
            if integrity.ok { "OK" } else { "FALLO" },
            integrity.stored,
            integrity.computed
        );
        println!(
            "tabla      : {} · offset {} · cabecera+tabla {} de {}",
            self.model.header.quant,
            header.tabla_offset,
            human_bytes(self.model.table_bytes()),
            human_bytes(self.model.size())
        );
        println!(
            "tokenizer  : {} · {} tokens · {} fusiones",
            self.cfg.tok.display(),
            self.tok.vocab_size(),
            self.tok.n_merges()
        );
        println!(
            "motor      : {} · router: {} (umbral {:.2})",
            self.engine.name(),
            self.router.name(),
            self.cfg.min_score
        );
        match &active {
            Some(r) => println!(
                "ruta activa: {} (id {}) · dim {} · {} en RAM",
                r.label(),
                r.id,
                r.dim,
                human_bytes(r.nbytes)
            ),
            None => println!("ruta activa: (ninguna todavía)"),
        }
        if let Some(c) = &self.last_choice {
            println!(
                "decisión   : id {} · {:.2} · {}{}",
                c.route_id,
                c.score,
                c.reason,
                if c.used_fallback { " · fallback" } else { "" }
            );
            let mut cands = c.candidates.clone();
            cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = cands.iter().take(4).map(|(id, s)| format!("{id}:{s:.2}")).collect();
            if !top.is_empty() {
                println!("candidatas : {}", top.join("  "));
            }
        }
        let rss = process_rss().map(human_bytes).unwrap_or_else(|| "n/d".into());
        println!(
            "RAM        : proceso {rss} · ruta activa {} · modelo en disco {}",
            active.as_ref().map(|r| human_bytes(r.nbytes)).unwrap_or_else(|| "0 B".into()),
            human_bytes(self.model.size())
        );
        println!(
            "tokens     : historial {} turnos · sidecar estrés {}",
            self.history.len(),
            sidecar_path(self.model.path()).display()
        );
        if let Some(s) = &self.last {
            println!(
                "último     : ruta {} · {} tok in / {} tok out · motor {}",
                s.route_path, s.tokens_in, s.tokens_out, s.engine
            );
        }
        let sectors = self.store.list();
        println!(
            "memoria TOJI: {} · sectores: {}",
            self.store.dir().display(),
            if sectors.is_empty() { "(ninguno)".to_string() } else { sectors.join(", ") }
        );
        println!(
            "feedback   : {} · {} registros pendientes",
            self.queue.path().display(),
            self.queue.pending()
        );
        let reports = self.stress.report(self.model.routes.iter().map(|r| r.id));
        for r in reports.iter().filter(|r| r.activations > 0) {
            println!(
                "estrés     : ruta {} — {}/{} activaciones{}",
                r.route_id,
                r.activations,
                r.threshold,
                if r.stressed { " (ESTRESADA)" } else { "" }
            );
        }
    }

    fn print_routes(&self) {
        println!(
            "{:>3}  {:<48} {:>6} {:>9} {:>12}  {}",
            "id", "etiqueta", "dim", "bytes", "estrés", "última"
        );
        for r in &self.model.routes {
            let activations = self.stress.activations(r.id);
            let thr = self.stress.threshold_for(r.id);
            let mark = if self.last_route == Some(r.id) { "·" } else { "" };
            println!(
                "{:>3}  {:<48} {:>6} {:>9} {:>12}  {}",
                r.id,
                r.label(),
                r.dim,
                human_bytes(r.nbytes),
                format!("{activations}/{thr}"),
                mark
            );
        }
        if let Some(c) = &self.last_choice {
            println!("\núltima decisión: {} · {:.2} · {}", c.label, c.score, c.reason);
        }
    }

    fn print_stress(&self) {
        println!(
            "estrés (decaimiento {:.2} por turno, umbral por defecto {})",
            self.stress.decay(),
            self.stress.threshold()
        );
        for rep in self.stress.report(self.model.routes.iter().map(|r| r.id)) {
            println!(
                "  ruta {:>3}  {:>6}/{:<6}  {}{}",
                rep.route_id,
                rep.activations,
                rep.threshold,
                if rep.stressed { "ESTRESADA" } else { "normal" },
                if rep.stressed_cycles > 0 {
                    format!("  ({} ciclos bajo estrés)", rep.stressed_cycles)
                } else {
                    String::new()
                }
            );
        }
    }

    /// `/stress <ruta> <n>`: umbral por ruta, persistido en el sidecar.
    fn set_stress(&mut self, arg: &str) -> Result<()> {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 2 {
            println!("uso: /stress <ruta> <activaciones>");
            return Ok(());
        }
        let route_id: u32 = parts[0]
            .parse()
            .map_err(|_| anyhow::anyhow!("«{}» no es un id de ruta", parts[0]))?;
        let threshold: u64 = parts[1]
            .parse()
            .map_err(|_| anyhow::anyhow!("«{}» no es un número de activaciones", parts[1]))?;
        if self.model.route_index(route_id).is_none() {
            println!("no existe la ruta {route_id} en este modelo");
            return Ok(());
        }
        self.stress.set_route_threshold(route_id, threshold);
        println!(
            "ruta {route_id}: umbral de estrés = {} activaciones",
            self.stress.threshold_for(route_id)
        );
        if self.cfg.write_memory {
            if let Err(e) = self.stress.save() {
                eprintln!("aviso: no se pudo guardar el sidecar de estrés: {e:#}");
            }
        }
        Ok(())
    }

    fn print_mem(&self) {
        let sectors = self.store.list();
        if sectors.is_empty() {
            println!("sin sectores TOJI en {}", self.store.dir().display());
            return;
        }
        for s in sectors {
            let p = self.store.path_for(&s);
            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            match self.store.load(&s) {
                Ok(sec) => println!(
                    "  {:<20} {:>9}  neuronas {:>4} · cables {:>4} · anormales {:>2} · fronteras {:>2}",
                    s,
                    human_bytes(size),
                    sec.neurons.len(),
                    sec.cables.len(),
                    sec.anormals.len(),
                    sec.borders.len()
                ),
                Err(e) => println!("  {s:<20} {:<9}  ilegible: {e}", human_bytes(size)),
            }
        }
    }

    /// `/merge <a> <b>`: fusiona dos sectores en el de menor nombre y lo guarda.
    fn cmd_merge(&mut self, arg: &str) -> Result<()> {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 2 {
            println!("uso: /merge <sector_a> <sector_b>");
            return Ok(());
        }
        let (a, b) = (sector_name(parts[0]), sector_name(parts[1]));
        if a == b {
            println!("los dos nombres apuntan al mismo sector «{a}»");
            return Ok(());
        }
        let (dst, src) = if a < b { (a, b) } else { (b, a) };
        let mut left = self.store.load(&dst)?;
        let right = self.store.load(&src)?;
        if right.neurons.is_empty() && right.cables.is_empty() && right.anormals.is_empty() {
            println!("el sector «{src}» está vacío o no existe: nada que fusionar");
            return Ok(());
        }
        let stats = left.merge(&right);
        let path = if self.cfg.write_memory {
            self.store.save(&left)?
        } else {
            self.store.path_for(&dst)
        };
        println!(
            "fusionado «{src}» → «{dst}»{}",
            if self.cfg.write_memory { "" } else { " (sin escribir: --no-mem-write)" }
        );
        println!(
            "  añadidos: neuronas {} · satélites {} · cables {} · anormales {} · fronteras {}",
            stats.neurons_added,
            stats.satellites_added,
            stats.cables_added,
            stats.anormals_added,
            stats.borders_added
        );
        println!(
            "  fusionados: cables {} · anormales {}",
            stats.cables_merged,
            stats.anormals_merged
        );
        println!("  destino: {}", path.display());
        Ok(())
    }

    /// `/tok <texto>`: tokens IRISTOK y comprobación de ida y vuelta.
    fn cmd_tok(&self, arg: &str) -> Result<()> {
        if arg.is_empty() {
            println!("uso: /tok <texto>");
            return Ok(());
        }
        let ids = self.tok.encode(arg);
        let decoded = self.tok.decode(&ids)?;
        let bytes = self.tok.decode_bytes(&ids)?.len();
        println!("tokens: {} · bytes: {} · ida y vuelta: {}", ids.len(), bytes, if decoded == arg { "OK" } else { "FALLO" });
        let shown: Vec<String> = ids
            .iter()
            .take(16)
            .map(|id| format!("{id}:{}", self.tok.token_text(*id).unwrap_or_else(|| "?".into())))
            .collect();
        println!("  {}", shown.join("  "));
        if ids.len() > 16 {
            println!("  … {} tokens más", ids.len() - 16);
        }
        Ok(())
    }

    /// `/queue [n]`: últimos registros del JSONL de feedback.
    fn cmd_queue(&self, arg: &str) {
        let n: usize = if arg.is_empty() {
            5
        } else {
            match arg.parse() {
                Ok(v) => v,
                Err(_) => {
                    println!("uso: /queue [n]");
                    return;
                }
            }
        };
        if n == 0 {
            println!("n debe ser mayor que 0");
            return;
        }
        let recs = match self.queue.read_all() {
            Ok(r) => r,
            Err(_) => {
                println!("la cola {} aún no existe o está vacía", self.queue.path().display());
                return;
            }
        };
        if recs.is_empty() {
            println!("la cola {} está vacía", self.queue.path().display());
            return;
        }
        let start = recs.len().saturating_sub(n);
        println!("últimos {} de {} registros:", recs.len() - start, recs.len());
        for r in &recs[start..] {
            let user = truncate(&r.user_text, 42);
            let iris = truncate(&r.iris_text, 42);
            println!(
                "  {:<10} ruta {:>3}  {} → {}",
                r.kind.as_str(),
                r.route_id,
                if user.is_empty() { "—" } else { &user },
                if iris.is_empty() { "—" } else { &iris }
            );
        }
    }
}

fn truncate(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}…")
    }
}
