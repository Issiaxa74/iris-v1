//! Tokenizer BPE por bytes, formato binario `IRISTOK` (iris.tok).
//!
//! Estructura (little-endian), idéntica a `iris/julia/BPE.jl`:
//! ```text
//! [0..8)   "IRISTOK\0"
//! [8..12)  u32 version (=1)
//! [12..16) u32 n_merges
//! [16..)   n_merges × { u32 a, u32 b }
//! ```
//! Vocabulario: ids 0..=255 son bytes directos; la fusión i (base 0) produce el
//! id `256 + i`. `encode` aplica las fusiones **en orden de entrenamiento**
//! (no por rango), exactamente como el original; `decode` recompone bytes.
//!
//! `encode` se salta una fusión cuando ninguno de sus dos símbolos está
//! presente: es una poda segura (la fusión no podría aplicarse) que no cambia
//! el resultado, solo el tiempo.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub const MAGIC: &[u8; 8] = b"IRISTOK\0";
pub const VERSION_SUPPORTED: u32 = 1;
pub const BASE_VOCAB: u32 = 256;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    bytes_of: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read(path)
            .with_context(|| format!("no se pudo leer el tokenizer {}", path.display()))?;
        Self::from_bytes(&raw).with_context(|| format!("tokenizer inválido: {}", path.display()))
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        if raw.len() < 16 {
            bail!("archivo demasiado corto ({} B)", raw.len());
        }
        if &raw[..8] != MAGIC {
            bail!(
                "magic inválido: {:?} (se esperaba \"IRISTOK\\0\")",
                String::from_utf8_lossy(&raw[..8])
            );
        }
        let version = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
        if version != VERSION_SUPPORTED {
            bail!("versión {version} no soportada (cabrita entiende v{VERSION_SUPPORTED})");
        }
        let n_merges = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
        let expected = 16usize + n_merges as usize * 8;
        if raw.len() != expected {
            bail!(
                "tamaño {} B no coincide con n_merges={n_merges} (esperado {expected} B)",
                raw.len()
            );
        }

        let mut bytes_of: Vec<Vec<u8>> = (0u32..256).map(|b| vec![b as u8]).collect();
        let mut merges = Vec::with_capacity(n_merges as usize);
        for i in 0..n_merges as usize {
            let off = 16 + i * 8;
            let a = u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            let b = u32::from_le_bytes([raw[off + 4], raw[off + 5], raw[off + 6], raw[off + 7]]);
            let (ba, bb) = (bytes_of.get(a as usize), bytes_of.get(b as usize));
            let (ba, bb) = match (ba, bb) {
                (Some(x), Some(y)) => (x.clone(), y.clone()),
                _ => bail!(
                    "fusión #{i} referencia un id fuera del vocabulario previo (a={a}, b={b})"
                ),
            };
            let mut merged = ba;
            merged.extend_from_slice(&bb);
            bytes_of.push(merged);
            merges.push((a, b));
        }
        Ok(Tokenizer { merges, bytes_of })
    }

    pub fn vocab_size(&self) -> u32 {
        BASE_VOCAB + self.merges.len() as u32
    }

    pub fn n_merges(&self) -> usize {
        self.merges.len()
    }

    /// Bytes que representa un token (None si el id no existe).
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.bytes_of.get(id as usize).map(|v| v.as_slice())
    }

    /// Texto → tokens. Fiel al original: fusiones en orden de entrenamiento.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut seq: Vec<u32> = text.bytes().map(u32::from).collect();
        for (i, &(a, b)) in self.merges.iter().enumerate() {
            // Poda segura: si ni `a` ni `b` están presentes, la fusión no aplica.
            if !seq.contains(&a) || !seq.contains(&b) {
                continue;
            }
            let new_id = BASE_VOCAB + i as u32;
            let mut out = Vec::with_capacity(seq.len());
            let mut j = 0usize;
            while j < seq.len() {
                if j + 1 < seq.len() && seq[j] == a && seq[j + 1] == b {
                    out.push(new_id);
                    j += 2;
                } else {
                    out.push(seq[j]);
                    j += 1;
                }
            }
            seq = out;
        }
        seq
    }

    /// Tokens → bytes exactos (sin interpretar UTF-8).
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for id in ids {
            let bytes = self
                .token_bytes(*id)
                .ok_or_else(|| anyhow::anyhow!("token {id} fuera del vocabulario"))?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }

    /// Tokens → texto, exigiendo UTF-8 válido (roundtrip byte a byte).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).context("los tokens no forman UTF-8 válido")
    }

    /// Bytes de un token, como String si es UTF-8 completo (para depuración).
    pub fn token_text(&self, id: u32) -> Option<String> {
        self.token_bytes(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }
}

/// Decodificador incremental: un token puede cortar un carácter UTF-8 a la
/// mitad, así que se retiene el sufijo incompleto hasta el token siguiente.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_token(&mut self, tok: &Tokenizer, id: u32) -> Result<String> {
        let bytes = tok
            .token_bytes(id)
            .ok_or_else(|| anyhow::anyhow!("token {id} fuera del vocabulario"))?;
        self.pending.extend_from_slice(bytes);
        Ok(self.drain_valid())
    }

    fn drain_valid(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap_or(""));
                        self.pending.drain(..valid);
                    }
                    match e.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..bad);
                        }
                        None => break, // sufijo incompleto: esperar más tokens
                    }
                }
            }
        }
        out
    }

    /// Vacía el resto (solo debería quedar basura si el flujo no era UTF-8).
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        Tokenizer::load(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/iris.tok")).unwrap()
    }

    #[test]
    fn cabecera_del_tokenizer() {
        let t = tok();
        assert_eq!(t.n_merges(), 800);
        assert_eq!(t.vocab_size(), 1056);
        assert_eq!(t.token_bytes(0x65).unwrap(), b"e");
        assert_eq!(t.token_bytes(256 + 0), t.token_bytes(256 + 0));
    }

    #[test]
    fn roundtrip_exacto() {
        let t = tok();
        for s in [
            "",
            "a",
            "hola mundo",
            "Programación en Rust: ¿por qué el borrow checker?",
            "ñandú, corazón, pingüino",
            "emoji 🦀 y símbolos — “comillas” …",
            "¿en qué estado está el mundo?",
            "linea1\nlinea2\ttab",
        ] {
            let ids = t.encode(s);
            let back = t.decode(&ids).unwrap();
            assert_eq!(back, s, "roundtrip falló para {s:?}");
            assert_eq!(t.encode(s), ids, "encode no determinista");
        }
    }

    #[test]
    fn vacio_y_fuera_de_vocabulario() {
        let t = tok();
        assert!(t.encode("").is_empty());
        assert!(t.decode(&[]).unwrap().is_empty());
        assert!(t.decode(&[9999]).is_err());
        assert!(t.token_bytes(1056).is_none());
    }

    #[test]
    fn decodificador_incremental() {
        let t = tok();
        let text = "corazón 🦀 ñandú";
        let ids = t.encode(text);
        let mut sd = StreamDecoder::new();
        let mut out = String::new();
        for id in &ids {
            out.push_str(&sd.push_token(&t, *id).unwrap());
        }
        out.push_str(&sd.flush());
        assert_eq!(out, text);
    }

    #[test]
    fn rechaza_corruptos() {
        let raw = vec![0u8; 20];
        assert!(Tokenizer::from_bytes(&raw).is_err());
        let mut good =
            std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/iris.tok")).unwrap();
        good[8] = 7;
        assert!(Tokenizer::from_bytes(&good).is_err());
        good[8] = 1;
        good.truncate(good.len() - 1);
        assert!(Tokenizer::from_bytes(&good).is_err());
    }

    #[test]
    fn comprime_algo() {
        let t = tok();
        let s = "el usuario dijo que la programación en rust es interesante y que el préstamo de datos le cuesta";
        let ids = t.encode(s);
        assert!(
            ids.len() < s.chars().count(),
            "esperaba compresión, {} tokens vs {} chars",
            ids.len(),
            s.chars().count()
        );
    }
}
