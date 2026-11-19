//! Utilidades transversales: tiempo, tamaños, normalización de texto.

use std::time::{SystemTime, UNIX_EPOCH};

/// Segundos desde la época Unix (0 si el reloj está antes de 1970).
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// ISO-8601 UTC a partir de segundos Unix, sin dependencias externas.
pub fn iso8601(unix: u64) -> String {
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Algoritmo días-civiles de Howard Hinnant (1970-01-01 = día 0).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn human_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    let f = n as f64;
    if f < KB {
        format!("{n} B")
    } else if f < KB * KB {
        format!("{:.1} KB", f / KB)
    } else if f < KB * KB * KB {
        format!("{:.1} MB", f / (KB * KB))
    } else {
        format!("{:.2} GB", f / (KB * KB * KB))
    }
}

/// Minúsculas y sin acentos, conservando solo letras/dígitos y separando con espacios.
/// Se usa para comparar contenido, nunca para reescribir el texto del usuario.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = true;
    for ch in text.chars() {
        let low = ch.to_lowercase().next().unwrap_or(ch);
        let mapped = match low {
            'á' | 'à' | 'ä' | 'â' | 'ã' | 'å' => Some('a'),
            'é' | 'è' | 'ë' | 'ê' => Some('e'),
            'í' | 'ì' | 'ï' | 'î' => Some('i'),
            'ó' | 'ò' | 'ö' | 'ô' | 'õ' => Some('o'),
            'ú' | 'ù' | 'ü' | 'û' => Some('u'),
            'ñ' => Some('n'),
            'ç' => Some('c'),
            other => Some(other),
        };
        let c = mapped.unwrap_or(low);
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Palabras normalizadas (sin acentos, minúsculas) con longitud mínima.
pub fn words(text: &str, min_len: usize) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .filter(|w| w.chars().count() >= min_len)
        .map(|w| w.to_string())
        .collect()
}

/// Sector TOJI válido: minúsculas, sin acentos, `[a-z0-9_-]`.
pub fn sector_name(raw: &str) -> String {
    let mut s = String::new();
    for c in normalize(raw).chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            s.push(c);
        } else if c == ' ' && !s.ends_with('_') {
            s.push('_');
        }
    }
    let s = s.trim_matches('_').to_string();
    if s.is_empty() {
        "general".to_string()
    } else {
        s
    }
}

/// Uso de memoria residente del proceso, si el sistema lo permite.
/// En Windows vÃ­a `K32GetProcessMemoryInfo`; en otros, `/proc/self/statm`.
#[cfg(windows)]
pub fn process_rss() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: handle propio del proceso y estructura que inicializa el sistema.
    unsafe {
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let ok = K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);
        if ok != 0 {
            Some(counters.WorkingSetSize as u64)
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
pub fn process_rss() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fechas_iso() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(1_700_000_000), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn normaliza_acentos() {
        assert_eq!(
            normalize("¿Programación en Ñandú?"),
            "programacion en nandu"
        );
        assert_eq!(normalize("  ---  "), "");
    }

    #[test]
    fn sectores() {
        assert_eq!(sector_name("Programación"), "programacion");
        assert_eq!(sector_name("Estados Unidos"), "estados_unidos");
        assert_eq!(sector_name("///"), "general");
    }

    #[test]
    fn bytes_legibles() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024 * 1024 * 3 / 2), "1.5 MB");
    }
}
