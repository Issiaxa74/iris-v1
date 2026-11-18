// ukj_reader v0.1 — lector del formato UKJ (validación de cabecera y tabla)
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
struct Ruta {
    id: u32,
    niveles: Vec<String>,
    offset: u64,
    nbytes: u64,
    dim: u32,
}

fn read_u32(f: &mut File) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_u16(f: &mut File) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    f.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test.ukj".to_string());
    let mut f = File::open(&path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    assert_eq!(&magic, b"UKJ\0", "magic invalido");

    let version = read_u32(&mut f)?;
    let qlen = {
        let mut b = [0u8; 1];
        f.read_exact(&mut b)?;
        b[0] as usize
    };
    let mut qb = vec![0u8; qlen];
    f.read_exact(&mut qb)?;
    let quant = String::from_utf8(qb).expect("quant no es utf8");
    let n_rutas = read_u32(&mut f)?;
    let _flags = read_u32(&mut f)?;

    let mut rutas = Vec::new();
    for _ in 0..n_rutas {
        let id = read_u32(&mut f)?;
        let n_niv = read_u16(&mut f)?;
        let mut niveles = Vec::new();
        for _ in 0..n_niv {
            let l = read_u16(&mut f)? as usize;
            let mut s = vec![0u8; l];
            f.read_exact(&mut s)?;
            niveles.push(String::from_utf8(s).expect("nivel no es utf8"));
        }
        let offset = read_u64(&mut f)?;
        let nbytes = read_u64(&mut f)?;
        let dim = read_u32(&mut f)?;
        rutas.push(Ruta { id, niveles, offset, nbytes, dim });
    }

    println!("version={} quant={} rutas={}", version, quant, rutas.len());
    for r in &rutas {
        println!(
            "ruta {} {:?} offset={} nbytes={} dim={}",
            r.id, r.niveles, r.offset, r.nbytes, r.dim
        );
    }

    // muestra de los primeros floats de la primera ruta
    if let Some(r) = rutas.first() {
        f.seek(SeekFrom::Start(r.offset))?;
        let n = (r.nbytes as usize).min(16);
        let mut b = vec![0u8; n];
        f.read_exact(&mut b)?;
        let floats: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        println!("muestra de {:?}: {:?}", r.niveles, floats);
    }

    Ok(())
}
