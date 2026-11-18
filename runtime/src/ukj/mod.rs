//! Lector del contenedor UKJ v0.1 (compatible byte a byte con `iris/julia/UKJ.jl`).
//!
//! Formato en disco (little-endian):
//! ```text
//! [0..4)    "UKJ\0"
//! [4..8)    u32 version (=1)
//! [8..9)    u8  qlen
//! [9..9+q)  cuantización utf8 ("p4-one"…)
//! [..)      u32 n_rutas
//! [..)      u32 flags (bit0 = tokenizer dentro del archivo; v0.1 lo deja en 0)
//! tabla:    n_rutas × { u32 id, u16 n_niveles, n_niveles×(u16 len, utf8),
//!                       u64 offset, u64 nbytes, u32 dim }
//! datos:    tensores Float32 LE, uno por ruta (sin relleno de alineación)
//! footer:   u64 tabla_offset, u64 checksum = suma de bytes previos mód 2^64
//! ```
//!
//! Diferencias con el borrador UKJ_SPEC.md que este lector asume, porque el
//! escritor F1a de Julia es la fuente de verdad:
//! * no hay relleno de 64 bytes entre tensores;
//! * el footer es suma aditiva de bytes, no CRC64 (el spec lo anota como CRC64);
//! * `dim` es un solo u32 (nº de elementos) y `nbytes == dim * 4`.
//!
//! El archivo se mapea entero en memoria virtual (mmap de solo lectura) pero
//! solo se tocan las páginas de la cabecera, la tabla y la ruta pedida: la
//! residencia real es la de esa ruta, que es lo que mide `/info`.

pub mod stress;

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;

pub use stress::StressTable;

pub const MAGIC: &[u8; 4] = b"UKJ\0";
pub const VERSION_SUPPORTED: u32 = 1;
const FOOTER_LEN: u64 = 16;

/// Cabecera del archivo UKJ.
#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub quant: String,
    pub n_rutas: u32,
    pub flags: u32,
    pub tabla_offset: u64,
    pub file_size: u64,
}

impl Header {
    /// `true` si el tokenizer viaja dentro del UKJ (no soportado en v0.1).
    pub fn tokenizer_embedded(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn version_label(&self) -> String {
        format!("UKJ v{} ({})", self.version, self.quant)
    }
}

/// Una ruta de conocimiento tal como aparece en la tabla.
#[derive(Debug, Clone)]
pub struct RouteInfo {
    pub id: u32,
    pub niveles: Vec<String>,
    pub offset: u64,
    pub nbytes: u64,
    pub dim: u32,
}

impl RouteInfo {
    /// `mundo › sub › concepto`
    pub fn label(&self) -> String {
        if self.niveles.is_empty() {
            format!("ruta-{}", self.id)
        } else {
            self.niveles.join(" › ")
        }
    }

    /// Primer nivel de la jerarquía: se usa como sector TOJI.
    pub fn sector(&self) -> String {
        self.niveles
            .first()
            .cloned()
            .unwrap_or_else(|| "general".to_string())
    }

    /// Términos de la etiqueta, normalizados y sin vacíos.
    pub fn terms(&self) -> Vec<String> {
        self.niveles
            .iter()
            .flat_map(|n| crate::util::words(n, 2))
            .collect()
    }

    pub fn n_floats(&self) -> u64 {
        self.nbytes / 4
    }
}

/// Resultado de comprobar la integridad del archivo.
#[derive(Debug, Clone)]
pub struct Integrity {
    pub stored: u64,
    pub computed: u64,
    pub ok: bool,
}

/// Tensores de una ruta ya copiados a RAM (lo único que se materializa).
#[derive(Debug, Clone)]
pub struct RouteTensors {
    pub route_id: u32,
    pub dim: u32,
    pub data: Vec<f32>,
}

impl RouteTensors {
    pub fn bytes(&self) -> u64 {
        self.data.len() as u64 * 4
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().copied().sum::<f32>() / self.data.len() as f32
    }

    pub fn energy(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum::<f32>()
    }

    /// Suma Float32 exacta en el orden del archivo: sirve como huella estable
    /// para comparar contra el escritor Julia byte a byte.
    pub fn checksum(&self) -> u64 {
        self.data
            .iter()
            .fold(0u64, |acc, v| acc.wrapping_add(v.to_bits() as u64))
    }
}

/// Modelo UKJ abierto con mmap de solo lectura.
pub struct UkjModel {
    path: PathBuf,
    mmap: Mmap,
    pub header: Header,
    pub routes: Vec<RouteInfo>,
    integrity: Integrity,
}

impl UkjModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("no se pudo abrir el modelo {}", path.display()))?;
        let size = file
            .metadata()
            .with_context(|| format!("no se pudo leer el tamaño de {}", path.display()))?
            .len();
        if size < 32 {
            bail!("{} es demasiado pequeño para ser UKJ ({} B)", path.display(), size);
        }
        // SAFETY: archivo abierto en solo lectura y no se modifica mientras viva el mmap.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap falló sobre {}", path.display()))?;
        let (header, routes) = parse(&mmap, size)?;
        let mut model = UkjModel {
            path,
            mmap,
            header: Header {
                version: 1,
                quant: String::new(),
                n_rutas: 0,
                flags: 0,
                tabla_offset: 0,
                file_size: size,
            },
            routes,
            integrity: Integrity { stored: 0, computed: 0, ok: false },
        };
        model.header = header;
        model.integrity = model.compute_integrity();
        Ok(model)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn size(&self) -> u64 {
        self.header.file_size
    }

    pub fn integrity(&self) -> &Integrity {
        &self.integrity
    }

    /// Bytes de la tabla de rutas (cabecera incluida).
    pub fn table_bytes(&self) -> u64 {
        self.header
            .file_size
            .saturating_sub(FOOTER_LEN)
            .saturating_sub(self.routes.iter().map(|r| r.nbytes).sum::<u64>())
    }

    /// Bytes residentes de una ruta concreta (sin contar su mmap).
    pub fn route_bytes(&self, route: &RouteInfo) -> u64 {
        route.nbytes
    }

    fn compute_integrity(&self) -> Integrity {
        // Checksum v0.1: suma de todos los bytes salvo los 8 últimos (el propio checksum).
        let body_len = self.mmap.len().saturating_sub(8);
        let computed = self.mmap[..body_len]
            .iter()
            .fold(0u64, |acc, b| acc.wrapping_add(*b as u64));
        let stored = self.stored_checksum();
        Integrity { stored, computed, ok: stored == computed }
    }

    fn stored_checksum(&self) -> u64 {
        let len = self.mmap.len();
        if len < 8 {
            return 0;
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.mmap[len - 8..]);
        u64::from_le_bytes(b)
    }

    pub fn route_index(&self, route_id: u32) -> Option<usize> {
        self.routes.iter().position(|r| r.id == route_id)
    }

    pub fn route(&self, idx: usize) -> Result<&RouteInfo> {
        self.routes
            .get(idx)
            .ok_or_else(|| anyhow!("índice de ruta {idx} fuera de rango ({} rutas)", self.routes.len()))
    }

    /// Copia a RAM los tensores de una ruta. Es la única lectura de datos del modelo.
    pub fn load_route(&self, idx: usize) -> Result<RouteTensors> {
        let route = self.route(idx)?.clone();
        self.load_route_info(&route)
    }

    #[allow(dead_code)] // atajo cómodo para el motor y las herramientas externas
    pub fn load_route_by_id(&self, route_id: u32) -> Result<RouteTensors> {
        let idx = self
            .route_index(route_id)
            .ok_or_else(|| anyhow!("no existe la ruta {route_id} en {}", self.path.display()))?;
        self.load_route(idx)
    }

    fn load_route_info(&self, route: &RouteInfo) -> Result<RouteTensors> {
        let start = route.offset as usize;
        let end = start
            .checked_add(route.nbytes as usize)
            .ok_or_else(|| anyhow!("desbordamiento de offset en la ruta {}", route.id))?;
        if end > self.mmap.len() {
            bail!(
                "ruta {}: rango {}..{} fuera del archivo ({} B)",
                route.id,
                start,
                end,
                self.mmap.len()
            );
        }
        if route.nbytes % 4 != 0 {
            bail!("ruta {}: nbytes {} no es múltiplo de 4", route.id, route.nbytes);
        }
        let raw = &self.mmap[start..end];
        let mut data = Vec::with_capacity(route.n_floats() as usize);
        for chunk in raw.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if route.dim as u64 != data.len() as u64 {
            bail!(
                "ruta {}: dim={} no coincide con {} floats",
                route.id,
                route.dim,
                data.len()
            );
        }
        Ok(RouteTensors { route_id: route.id, dim: route.dim, data })
    }

    /// Ruta por defecto del router cuando no hay coincidencia: la primera que
    /// suene a "general"/"mundo"/"sft", o la de menor id.
    #[allow(dead_code)] // el router por contenido ya resuelve el fallback; se conserva la API
    pub fn fallback_route(&self) -> Option<&RouteInfo> {
        for needle in ["general", "mundo", "sft"] {
            if let Some(r) = self
                .routes
                .iter()
                .find(|r| r.niveles.iter().any(|n| crate::util::normalize(n).contains(needle)))
            {
                return Some(r);
            }
        }
        self.routes.iter().min_by_key(|r| r.id)
    }
}

impl std::fmt::Debug for UkjModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UkjModel")
            .field("path", &self.path)
            .field("header", &self.header)
            .field("rutas", &self.routes.len())
            .field("integrity", &self.integrity)
            .finish()
    }
}

fn parse(mmap: &[u8], size: u64) -> Result<(Header, Vec<RouteInfo>)> {
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> Result<&[u8]> {
        let end = p.checked_add(n).ok_or_else(|| anyhow!("longitud desbordada"))?;
        if end > mmap.len() {
            bail!("archivo truncado: se esperaban {n} bytes en offset {p}");
        }
        let s = &mmap[*p..end];
        *p = end;
        Ok(s)
    };

    let magic = take(&mut p, 4)?;
    if magic != MAGIC {
        bail!("magic inválido: {:?} (se esperaba \"UKJ\\0\")", String::from_utf8_lossy(magic));
    }
    let version = read_u32(take(&mut p, 4)?);
    if version != VERSION_SUPPORTED {
        bail!("versión UKJ {version} no soportada (cabrita entiende v{VERSION_SUPPORTED})");
    }
    let qlen = read_u8(take(&mut p, 1)?) as usize;
    if qlen == 0 {
        bail!("cabecera corrupta: cuantización vacía");
    }
    let quant = String::from_utf8(take(&mut p, qlen)?.to_vec())
        .context("cuantización no es UTF-8 válido")?;
    let n_rutas = read_u32(take(&mut p, 4)?);
    let flags = read_u32(take(&mut p, 4)?);
    let tabla_offset = p as u64;

    let mut routes = Vec::with_capacity(n_rutas as usize);
    for _ in 0..n_rutas {
        let id = read_u32(take(&mut p, 4)?);
        let n_niveles = read_u16(take(&mut p, 2)?);
        let mut niveles = Vec::with_capacity(n_niveles as usize);
        for _ in 0..n_niveles {
            let len = read_u16(take(&mut p, 2)?) as usize;
            let bytes = take(&mut p, len)?;
            niveles.push(
                String::from_utf8(bytes.to_vec())
                    .with_context(|| format!("ruta {id}: nivel no es UTF-8"))?,
            );
        }
        let offset = read_u64(take(&mut p, 8)?);
        let nbytes = read_u64(take(&mut p, 8)?);
        let dim = read_u32(take(&mut p, 4)?);
        routes.push(RouteInfo { id, niveles, offset, nbytes, dim });
    }
    let tabla_end = p as u64;

    if routes.is_empty() {
        bail!("el modelo no declara ninguna ruta");
    }
    if size < FOOTER_LEN {
        bail!("archivo sin footer ({} B)", size);
    }
    let data_start = size - FOOTER_LEN;
    let mut prev_end = tabla_end;
    for r in &routes {
        if r.nbytes % 4 != 0 {
            bail!("ruta {}: nbytes {} no es múltiplo de 4", r.id, r.nbytes);
        }
        if r.nbytes != r.dim as u64 * 4 {
            bail!(
                "ruta {}: nbytes={} contradice dim={} (esperado {})",
                r.id,
                r.nbytes,
                r.dim,
                r.dim as u64 * 4
            );
        }
        if r.offset < tabla_end || r.offset + r.nbytes > data_start {
            bail!(
                "ruta {}: offset/nbytes fuera de la zona de datos ({}..{}, datos {tabla_end}..{data_start})",
                r.id,
                r.offset,
                r.offset + r.nbytes
            );
        }
        if r.offset < prev_end {
            bail!("ruta {}: tensores solapados (offset {} < {})", r.id, r.offset, prev_end);
        }
        prev_end = r.offset + r.nbytes;
    }

    let footer = &mmap[data_start as usize..];
    let footer_tabla = read_u64(&footer[..8]);
    if footer_tabla != tabla_offset {
        bail!(
            "footer inconsistente: tabla_offset={footer_tabla} pero la tabla empieza en {tabla_offset}"
        );
    }
    if n_rutas as usize != routes.len() {
        bail!("n_rutas={n_rutas} pero se leyeron {}", routes.len());
    }
    if flags & 1 != 0 {
        // No es fatal: el modelo se puede cargar, pero el tokenizer embebido no.
        eprintln!("aviso: flags={flags} declara tokenizer embebido; cabrita usa iris.tok externo");
    }

    Ok((
        Header { version, quant, n_rutas, flags, tabla_offset, file_size: size },
        routes,
    ))
}

fn read_u8(b: &[u8]) -> u8 {
    b[0]
}

fn read_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn read_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_model() -> UkjModel {
        UkjModel::open(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.ukj")).unwrap()
    }

    #[test]
    fn cabecera_del_fixture() {
        let m = fixture_model();
        assert_eq!(m.header.version, 1);
        assert_eq!(m.header.quant, "p4-one");
        assert_eq!(m.header.n_rutas, 3);
        assert_eq!(m.header.flags, 0);
        assert!(m.header.tokenizer_embedded() == false);
        assert_eq!(m.routes.len(), 3);
        assert_eq!(m.size(), 939);
        assert!(m.integrity().ok, "checksum del fixture debe validar");
    }

    #[test]
    fn rutas_y_tensores_exactos() {
        let m = fixture_model();
        assert_eq!(m.routes[0].id, 1);
        assert_eq!(m.routes[0].niveles, vec!["informatica", "programacion"]);
        assert_eq!(m.routes[0].dim, 64);
        assert_eq!(m.routes[0].label(), "informatica › programacion");
        assert_eq!(m.routes[0].sector(), "informatica");

        let t = m.load_route(0).unwrap();
        assert_eq!(t.data.len(), 64);
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[63], 64.0);
        assert_eq!(t.mean(), 32.5);

        let t2 = m.load_route_by_id(2).unwrap();
        assert_eq!(t2.data[0], 100.0);
        assert_eq!(t2.data[63], 163.0);

        // Byte a byte: recomponer los 64 floats como bytes LE debe coincidir
        // con el fragmento del archivo.
        let start = m.routes[0].offset as usize;
        let raw = &m.mmap[start..start + 256];
        let mut rebuilt = Vec::with_capacity(256);
        for v in &t.data {
            rebuilt.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(raw, rebuilt.as_slice());
    }

    #[test]
    fn ruta_inexistente() {
        let m = fixture_model();
        assert!(m.load_route_by_id(99).is_err());
        assert!(m.load_route(7).is_err());
    }

    #[test]
    fn rechaza_magic_y_version() {
        let dir = std::env::temp_dir().join("cabrita_ukj_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.ukj")).unwrap();
        let bad_magic = dir.join("magic.ukj");
        bytes[0] = b'X';
        std::fs::write(&bad_magic, &bytes).unwrap();
        assert!(UkjModel::open(&bad_magic).is_err());

        bytes[0] = b'U';
        let bad_version = dir.join("version.ukj");
        bytes[4] = 9;
        std::fs::write(&bad_version, &bytes).unwrap();
        let err = UkjModel::open(&bad_version).unwrap_err().to_string();
        assert!(err.contains("versión"), "mensaje inesperado: {err}");
    }

    #[test]
    fn detecta_checksum_alterado() {
        let dir = std::env::temp_dir().join("cabrita_ukj_checksum");
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.ukj")).unwrap();
        bytes[200] ^= 0xFF;
        let p = dir.join("corrupto.ukj");
        std::fs::write(&p, &bytes).unwrap();
        let m = UkjModel::open(&p).unwrap();
        assert!(!m.integrity().ok);
        assert_ne!(m.integrity().stored, m.integrity().computed);
    }

    #[test]
    fn truncado_no_entra_en_panico() {
        let dir = std::env::temp_dir().join("cabrita_ukj_trunc");
        std::fs::create_dir_all(&dir).unwrap();
        let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.ukj")).unwrap();
        let p = dir.join("corto.ukj");
        std::fs::write(&p, &bytes[..40]).unwrap();
        let err = UkjModel::open(&p).unwrap_err().to_string();
        assert!(err.contains("truncado") || err.contains("footer") || err.contains("rango"), "{err}");
    }

    #[test]
    fn fallback_route() {
        let m = fixture_model();
        let f = m.fallback_route().unwrap();
        assert_eq!(f.id, 3);
    }
}
