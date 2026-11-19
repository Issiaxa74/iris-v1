//! Motor real: forward del trunk sobre los tensores UKJ, con caché KV.
//!
//! Reconstruye el modelo que entrena `src/julia/model.jl` a partir de dos
//! archivos que produce `src/julia/export_ukj.jl`:
//!
//! * `modelo.ukj` — ruta `trunk` = `emb` + `t1..tN` + `fin.ln_g/ln_b`;
//! * `modelo.ukj.json` — manifiesto con la config y el offset de cada tensor
//!   dentro de la ruta (UKJ v0.1 no guarda nombres ni formas).
//!
//! El forward replica el de Julia: embeddings atados, senoidales absolutas,
//! pre-LN, GELU tanh, atención causal multi-cabeza y logits contra `emb`.
//! La caché KV guarda K y V por capa, así que cada token nuevo cuesta O(T)
//! en la atención y una pasada de pesos, no un forward completo.
//!
//! Los ids del tokenizador son 0-based y el modelo usa 1-based: aquí se suma
//! 1 al entrar y se enmascaran los ids que el tokenizador no puede decodificar
//! (el modelo reserva 16384 pero el BPE real tiene 1056).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::Deserialize;

use crate::engine::{Engine, EngineAnswer, EngineContext, GenOptions};
use crate::tok::{StreamDecoder, Tokenizer};
use crate::ukj::UkjModel;
use crate::util::human_bytes;

// ----------------------------- manifiesto ------------------------------

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    config: MConfig,
    rutas: Vec<MRuta>,
}

#[derive(Debug, Clone, Deserialize)]
struct MConfig {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize,
    vocab: usize,
    ctx: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct MRuta {
    id: u32,
    nivel: String,
    n_floats: usize,
    tensores: Vec<MTensor>,
}

#[derive(Debug, Clone, Deserialize)]
struct MTensor {
    nombre: String,
    offset: usize,
    n: usize,
}

/// Un tensor dentro del blob de la ruta.
#[derive(Debug, Clone, Copy)]
struct Span {
    off: usize,
    len: usize,
}

impl Span {
    fn get<'a>(&self, w: &'a [f32]) -> &'a [f32] {
        &w[self.off..self.off + self.len]
    }
}

#[derive(Debug, Clone)]
struct LayerW {
    ln1g: Span,
    ln1b: Span,
    wqkv: Span,
    wo: Span,
    ln2g: Span,
    ln2b: Span,
    w1: Span,
    w2: Span,
}

/// Pesos del trunk ya resueltos por nombre.
struct Trunk {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize,
    vocab: usize,
    ctx: usize,
    data: Vec<f32>,
    emb: Span,
    layers: Vec<LayerW>,
    fin_g: Span,
    fin_b: Span,
}

impl Trunk {
    fn bytes(&self) -> u64 {
        self.data.len() as u64 * 4
    }
}

fn span_de(
    map: &HashMap<String, MTensor>,
    nombre: &str,
    esperado: usize,
    total: usize,
) -> Result<Span> {
    let t = map.get(nombre).ok_or_else(|| {
        anyhow::anyhow!(
            "el manifiesto no declara el tensor «{nombre}» en la ruta trunk \
             (¿exportaste con export_ukj.jl?)"
        )
    })?;
    if t.n != esperado {
        bail!(
            "tensor «{nombre}»: el manifiesto dice {} floats, la arquitectura espera {esperado}",
            t.n
        );
    }
    if t.offset + t.n > total {
        bail!(
            "tensor «{nombre}»: offset {} + {} sale de la ruta ({} floats)",
            t.offset,
            t.n,
            total
        );
    }
    Ok(Span {
        off: t.offset,
        len: t.n,
    })
}

fn cargar_trunk(model_path: &Path) -> Result<Trunk> {
    let manifest_path = PathBuf::from(format!("{}.json", model_path.display()));
    let raw = fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "no existe el manifiesto {} (lo escribe `julia src/julia/export_ukj.jl`)",
            manifest_path.display()
        )
    })?;
    let man: Manifest = serde_json::from_str(&raw)
        .with_context(|| format!("manifiesto inválido: {}", manifest_path.display()))?;
    let cfg = &man.config;
    if cfg.dim == 0 || cfg.n_layers == 0 || cfg.n_heads == 0 || cfg.vocab == 0 {
        bail!("configuración del manifiesto incompleta: {cfg:?}");
    }
    if cfg.dim % cfg.n_heads != 0 {
        bail!(
            "dim={} no es divisible entre n_heads={}",
            cfg.dim,
            cfg.n_heads
        );
    }
    if cfg.ctx < 2 {
        bail!("ctx={} insuficiente (mínimo 2)", cfg.ctx);
    }

    let ruta = man
        .rutas
        .iter()
        .find(|r| r.nivel == "trunk")
        .or_else(|| man.rutas.iter().min_by_key(|r| r.id))
        .ok_or_else(|| anyhow::anyhow!("el manifiesto no declara ninguna ruta"))?;

    let modelo = UkjModel::open_no_verify(model_path)?;
    let idx = modelo.route_index(ruta.id).ok_or_else(|| {
        anyhow::anyhow!(
            "la ruta trunk id={} del manifiesto no está en {}",
            ruta.id,
            model_path.display()
        )
    })?;
    let tensores = modelo.load_route(idx)?;
    if tensores.data.len() != ruta.n_floats {
        bail!(
            "ruta trunk: {} floats en el UKJ, {} en el manifiesto",
            tensores.data.len(),
            ruta.n_floats
        );
    }

    let mut map: HashMap<String, MTensor> = HashMap::with_capacity(ruta.tensores.len());
    for t in &ruta.tensores {
        if map.insert(t.nombre.clone(), t.clone()).is_some() {
            bail!("el manifiesto repite el tensor «{}»", t.nombre);
        }
    }

    let d = cfg.dim;
    let f = cfg.ffn;
    let v = cfg.vocab;
    let total = tensores.data.len();
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for l in 1..=cfg.n_layers {
        let p = |s: &str| format!("t{l}.{s}");
        layers.push(LayerW {
            ln1g: span_de(&map, &p("ln1g"), d, total)?,
            ln1b: span_de(&map, &p("ln1b"), d, total)?,
            wqkv: span_de(&map, &p("wqkv"), d * 3 * d, total)?,
            wo: span_de(&map, &p("wo"), d * d, total)?,
            ln2g: span_de(&map, &p("ln2g"), d, total)?,
            ln2b: span_de(&map, &p("ln2b"), d, total)?,
            w1: span_de(&map, &p("w1"), d * f, total)?,
            w2: span_de(&map, &p("w2"), f * d, total)?,
        });
    }

    Ok(Trunk {
        dim: d,
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        ffn: f,
        vocab: v,
        ctx: cfg.ctx,
        emb: span_de(&map, "emb", d * v, total)?,
        layers,
        fin_g: span_de(&map, "fin.ln_g", d, total)?,
        fin_b: span_de(&map, "fin.ln_b", d, total)?,
        data: tensores.data,
    })
}

// ------------------------------ caché KV -------------------------------

struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct KvCache {
    cap: usize,
    layers: Vec<LayerCache>,
}

impl KvCache {
    fn new(n_layers: usize, cap: usize, d: usize) -> Self {
        KvCache {
            cap,
            layers: (0..n_layers)
                .map(|_| LayerCache {
                    k: vec![0.0; cap * d],
                    v: vec![0.0; cap * d],
                })
                .collect(),
        }
    }
}

// ------------------------------ aritmética -----------------------------

/// `out[o] = dot(x, W[o, :])` con W en orden column-major (formato de model.jl:
/// W es [in, out] y el elemento (i, o) vive en `i + o * in`).
fn matvec(x: &[f32], w: &[f32], out: &mut [f32], input: usize) {
    debug_assert_eq!(w.len(), out.len() * input);
    debug_assert_eq!(x.len(), input);
    out.par_iter_mut().enumerate().for_each(|(o, y)| {
        let row = &w[o * input..(o + 1) * input];
        let mut acc = [0.0f32; 4];
        let mut i = 0usize;
        while i + 4 <= input {
            acc[0] += x[i] * row[i];
            acc[1] += x[i + 1] * row[i + 1];
            acc[2] += x[i + 2] * row[i + 2];
            acc[3] += x[i + 3] * row[i + 3];
            i += 4;
        }
        let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        while i < input {
            s += x[i] * row[i];
            i += 1;
        }
        *y = s;
    });
}

/// Layernorm de una fila, idéntico a `_ln!` de model.jl (eps 1e-5).
fn layernorm(x: &[f32], g: &[f32], b: &[f32], out: &mut [f32]) {
    let d = x.len();
    let mut sum = 0.0f32;
    for v in x {
        sum += *v;
    }
    let mean = sum / d as f32;
    let mut var = 0.0f32;
    for v in x {
        let c = *v - mean;
        var += c * c;
    }
    let rstd = 1.0 / (var / d as f32 + 1.0e-5).sqrt();
    for k in 0..d {
        out[k] = (x[k] - mean) * rstd * g[k] + b[k];
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_56 * (x + 0.044_715 * x * x * x)).tanh())
}

/// Tabla de senoidales absolutas, igual que `sin_table` de model.jl.
fn sin_fila(pos: usize, d: usize, out: &mut [f32]) {
    for i in 0..d / 2 {
        let freq = (-(10000.0f64).ln() * (2 * i) as f64 / d as f64).exp();
        let ang = (pos as f64 * freq) as f32;
        out[2 * i] = ang.sin();
        out[2 * i + 1] = ang.cos();
    }
}

// ------------------------------ motor ----------------------------------

pub struct TransformerEngine {
    trunk: Trunk,
    opts: GenOptions,
    cache: Option<KvCache>,
    rng: u64,
}

impl TransformerEngine {
    pub fn open(model_path: &Path, opts: GenOptions) -> Result<Self> {
        let trunk = cargar_trunk(model_path)
            .with_context(|| format!("motor transformer: {}", model_path.display()))?;
        let seed = opts.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        });
        Ok(TransformerEngine {
            trunk,
            opts,
            cache: None,
            rng: seed | 1,
        })
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*: suficiente para muestrear y reproducible con --seed.
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Forward de un token en `pos`; escribe los logits (0-based, id de modelo
    /// = índice + 1) en `logits` y devuelve nada.
    fn forward_token(
        &mut self,
        token: u32,
        pos: usize,
        logits: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let t = &self.trunk;
        let d = t.dim;
        let h = t.n_heads;
        let hd = d / h;
        let f = t.ffn;
        let v = t.vocab;
        if pos >= self.cache.as_ref().map(|c| c.cap).unwrap_or(0) {
            bail!("posición {pos} fuera de la caché KV");
        }

        // x = emb[token-1] + senoidal(pos)
        let base = (token as usize).checked_sub(1).ok_or_else(|| {
            anyhow::anyhow!("token 0 en el modelo: los ids del BPE son 1-based aquí")
        })? * d;
        if base + d > t.emb.off + t.emb.len {
            bail!("token {token} fuera del vocabulario del modelo (vocab {v})");
        }
        scratch.x.copy_from_slice(&t.data[base..base + d]);
        sin_fila(pos, d, &mut scratch.seno);
        for k in 0..d {
            scratch.x[k] += scratch.seno[k];
        }

        let inv = 1.0 / (hd as f32).sqrt();
        let cache = self.cache.as_mut().expect("caché KV inicializada");

        for l in 0..t.n_layers {
            let lw = &t.layers[l];
            let lc = &mut cache.layers[l];

            // --- atención ---
            layernorm(
                &scratch.x,
                lw.ln1g.get(&t.data),
                lw.ln1b.get(&t.data),
                &mut scratch.n1,
            );
            matvec(&scratch.n1, lw.wqkv.get(&t.data), &mut scratch.qkv, d);

            // guarda K y V de este token (por cabeza, posiciones disjuntas)
            for head in 0..h {
                let src = d + head * hd;
                let vsrc = 2 * d + head * hd;
                let dst = pos * d + head * hd;
                lc.k[dst..dst + hd].copy_from_slice(&scratch.qkv[src..src + hd]);
                lc.v[dst..dst + hd].copy_from_slice(&scratch.qkv[vsrc..vsrc + hd]);
            }

            let scores_len = pos + 1;
            let mut scores = vec![0.0f32; h * scores_len];
            scores
                .par_chunks_mut(scores_len)
                .zip(scratch.ao.par_chunks_mut(hd))
                .enumerate()
                .for_each(|(head, (sc, ao_h))| {
                    let q = &scratch.qkv[head * hd..(head + 1) * hd];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..=pos {
                        let krow = &lc.k[j * d + head * hd..j * d + head * hd + hd];
                        let mut s = 0.0f32;
                        for k in 0..hd {
                            s += q[k] * krow[k];
                        }
                        s *= inv;
                        sc[j] = s;
                        if s > mx {
                            mx = s;
                        }
                    }
                    let mut sum = 0.0f32;
                    for j in 0..=pos {
                        let e = (sc[j] - mx).exp();
                        sc[j] = e;
                        sum += e;
                    }
                    let isum = 1.0 / sum;
                    for j in 0..=pos {
                        sc[j] *= isum;
                    }
                    for k in 0..hd {
                        ao_h[k] = 0.0;
                    }
                    for j in 0..=pos {
                        let p = sc[j];
                        let vrow = &lc.v[j * d + head * hd..j * d + head * hd + hd];
                        for k in 0..hd {
                            ao_h[k] += p * vrow[k];
                        }
                    }
                });

            matvec(&scratch.ao, lw.wo.get(&t.data), &mut scratch.mm, d);
            for k in 0..d {
                scratch.x2[k] = scratch.x[k] + scratch.mm[k];
            }

            // --- FFN ---
            layernorm(
                &scratch.x2,
                lw.ln2g.get(&t.data),
                lw.ln2b.get(&t.data),
                &mut scratch.n2,
            );
            matvec(&scratch.n2, lw.w1.get(&t.data), &mut scratch.pre, d);
            scratch
                .h
                .par_iter_mut()
                .zip(scratch.pre.par_iter())
                .for_each(|(hv, pv)| *hv = gelu(*pv));
            matvec(&scratch.h, lw.w2.get(&t.data), &mut scratch.ff, f);
            for k in 0..d {
                scratch.x[k] = scratch.x2[k] + scratch.ff[k];
            }
        }

        layernorm(
            &scratch.x,
            t.fin_g.get(&t.data),
            t.fin_b.get(&t.data),
            &mut scratch.hfin,
        );
        matvec(&scratch.hfin, t.emb.get(&t.data), logits, d);
        Ok(())
    }

    fn sample(&mut self, logits: &[f32], max_id: usize) -> u32 {
        let temp = self.opts.temperature;
        let top_k = self.opts.top_k;
        let top_p = self.opts.top_p;
        let n = logits.len().min(max_id);
        if n == 0 {
            return 0;
        }

        let mut idx: Vec<usize> = (0..n).collect();
        if temp <= 0.0 {
            let mut best = 0usize;
            for j in 1..n {
                if logits[j] > logits[best] {
                    best = j;
                }
            }
            return best as u32;
        }

        let scaled: Vec<f32> = logits[..n].iter().map(|v| v / temp).collect();
        idx.sort_by(|&a, &b| scaled[b].total_cmp(&scaled[a]));
        let keep = if top_k > 0 { top_k.min(n) } else { n };
        idx.truncate(keep);

        let mx = scaled[idx[0]];
        let mut pesos: Vec<f32> = idx.iter().map(|&i| (scaled[i] - mx).exp()).collect();
        let sum: f32 = pesos.iter().sum();
        for p in &mut pesos {
            *p /= sum;
        }

        if top_p < 1.0 {
            let mut acc = 0.0f32;
            let mut corte = pesos.len();
            for (j, p) in pesos.iter().enumerate() {
                acc += *p;
                if acc >= top_p {
                    corte = j + 1;
                    break;
                }
            }
            pesos.truncate(corte);
            idx.truncate(corte);
            let s: f32 = pesos.iter().sum();
            for p in &mut pesos {
                *p /= s;
            }
        }

        let u = (self.next_u64() as f64 / u64::MAX as f64) as f32;
        let mut acc = 0.0f32;
        for (j, p) in pesos.iter().enumerate() {
            acc += *p;
            if u <= acc {
                return idx[j] as u32;
            }
        }
        idx.last().copied().unwrap_or(0) as u32
    }

    /// Genera a partir de ids del tokenizador (0-based). Devuelve texto y los
    /// ids generados (0-based).
    fn generate(&mut self, prompt: &[u32], tok: &Tokenizer) -> Result<(String, Vec<u32>)> {
        let d = self.trunk.dim;
        let ctx = self.trunk.ctx;
        let max_new = self.opts.max_new;
        let prompt = if prompt.is_empty() {
            vec![b' ' as u32]
        } else {
            prompt.to_vec()
        };
        // El modelo necesita al menos un token de contexto y no puede pasar ctx.
        let max_prompt = ctx.saturating_sub(1).max(1);
        let prompt = if prompt.len() > max_prompt {
            prompt[prompt.len() - max_prompt..].to_vec()
        } else {
            prompt
        };

        let cap = (prompt.len() + max_new + 1).min(ctx).max(prompt.len() + 1);
        self.cache = Some(KvCache::new(self.trunk.n_layers, cap, d));
        let mut scratch = Scratch::new(d, self.trunk.ffn);
        let mut logits = vec![0.0f32; self.trunk.vocab];
        let max_id = tok.vocab_size() as usize;

        let mut pos = 0usize;
        for id in &prompt {
            let model_id = id + 1;
            self.forward_token(model_id, pos, &mut logits, &mut scratch)?;
            pos += 1;
        }

        let mut generated = Vec::with_capacity(max_new);
        let mut dec = StreamDecoder::new();
        let mut texto = String::new();
        for _ in 0..max_new {
            if pos >= cap {
                break;
            }
            let id = self.sample(&logits, max_id);
            generated.push(id);
            texto.push_str(&dec.push_token(tok, id)?);
            self.forward_token(id + 1, pos, &mut logits, &mut scratch)?;
            pos += 1;
        }
        texto.push_str(&dec.flush());
        Ok((texto, generated))
    }
}

/// Buffers del forward, reutilizados durante toda la generación.
struct Scratch {
    x: Vec<f32>,
    seno: Vec<f32>,
    n1: Vec<f32>,
    qkv: Vec<f32>,
    ao: Vec<f32>,
    mm: Vec<f32>,
    x2: Vec<f32>,
    n2: Vec<f32>,
    pre: Vec<f32>,
    h: Vec<f32>,
    ff: Vec<f32>,
    hfin: Vec<f32>,
}

impl Scratch {
    fn new(d: usize, f: usize) -> Self {
        Scratch {
            x: vec![0.0; d],
            seno: vec![0.0; d],
            n1: vec![0.0; d],
            qkv: vec![0.0; 3 * d],
            ao: vec![0.0; d],
            mm: vec![0.0; d],
            x2: vec![0.0; d],
            n2: vec![0.0; d],
            pre: vec![0.0; f],
            h: vec![0.0; f],
            ff: vec![0.0; d],
            hfin: vec![0.0; d],
        }
    }
}

impl Engine for TransformerEngine {
    fn name(&self) -> &str {
        "transformer"
    }

    fn uses_route_tensors(&self) -> bool {
        false
    }

    fn respond(&mut self, ctx: &EngineContext) -> Result<EngineAnswer> {
        let started = Instant::now();
        let (text, generated) = self.generate(ctx.tokens, ctx.tok)?;
        let elapsed = started.elapsed();
        let n = generated.len().max(1) as f64;
        let tok_s = n / elapsed.as_secs_f64().max(1.0e-9);

        let mut notes = vec![format!(
            "trunk {} capas · {} · {} tok/s",
            self.trunk.n_layers,
            human_bytes(self.trunk.bytes()),
            tok_s.round() as u64
        )];
        if ctx.route.label() != "trunk" {
            notes.push(format!(
                "ruta {} elegida por el router; el texto sale del trunk",
                ctx.route.label()
            ));
        }

        Ok(EngineAnswer { text, notes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Option<TransformerEngine> {
        let p = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/tiny.ukj"));
        if !p.exists() {
            return None;
        }
        Some(
            TransformerEngine::open(
                p,
                GenOptions {
                    max_new: 4,
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    seed: Some(7),
                },
            )
            .unwrap(),
        )
    }

    /// Valores de referencia calculados con `src/julia/model.jl` (forward_loss)
    /// sobre `tiny.irmdl`, el mismo checkpoint que produjo `tiny.ukj`.
    /// Prompt "hola mundo" -> ids [104, 323, 977, 568, 111].
    #[test]
    fn logits_coinciden_con_julia() {
        let Some(mut e) = fixture() else {
            eprintln!("sin assets/tiny.ukj: test omitido");
            return;
        };
        let tok = Tokenizer::load(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/iris.tok")).unwrap();
        let ids = tok.encode("hola mundo");
        assert_eq!(ids, vec![104, 323, 977, 568, 111]);

        let cap = ids.len() + 1;
        let d = e.trunk.dim;
        let f = e.trunk.ffn;
        e.cache = Some(KvCache::new(e.trunk.n_layers, cap, d));
        let mut scratch = Scratch::new(d, f);
        let mut logits = vec![0.0f32; e.trunk.vocab];
        for (pos, id) in ids.iter().enumerate() {
            e.forward_token(id + 1, pos, &mut logits, &mut scratch)
                .unwrap();
        }

        let esperado = [
            0.078764796f32,
            -0.21652567,
            0.1403784,
            0.005852605,
            -0.018697007,
            -0.1343285,
            0.11661493,
            0.0032750696,
        ];
        for (j, v) in esperado.iter().enumerate() {
            let dif = (logits[j] - v).abs();
            assert!(
                dif < 2.0e-2,
                "logit {j}: rust={} julia={v} dif={dif}",
                logits[j]
            );
        }
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 1044);
    }
    #[test]
    fn carga_el_fixture_y_genera() {
        let Some(mut e) = fixture() else {
            eprintln!("sin assets/tiny.ukj: test omitido");
            return;
        };
        let tok = Tokenizer::load(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/iris.tok")).unwrap();
        let ids = tok.encode("hola");
        let (texto, generados) = e.generate(&ids, &tok).unwrap();
        assert_eq!(generados.len(), 4);
        assert!(texto.len() < 4096);
        assert!(generados
            .iter()
            .all(|id| (*id as usize) < tok.vocab_size() as usize));
    }
}
