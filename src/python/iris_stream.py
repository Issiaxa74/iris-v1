# -*- coding: utf-8 -*-
"""
IRIS F4b - Pipeline de ingesta streaming (solo DATOS; el entrenamiento lo conecta
Desarrollador de ia).

Principios:
- Lectura por HTTP Range sobre Parquet (row-groups), nunca descarga completa.
- Filtro idioma/calidad antes de escribir.
- Buffer filtrado en DISCO con tope (jsonl.gz troceado), FIFO por dataset.
- Cursor por dataset, reanudable tras reinicio (JSON atomico).
- Determinista: misma entrada -> mismos chunks (gzip mtime=0, orden fijo).

Rutas:
  iris/datos/buffer/<dataset>/<dataset>_<NNNNNN>.jsonl.gz   (chunks filtrados)
  iris/datos/cursors/<dataset>.cursor.json                  (cursor por dataset)

Interfaz para el tokenizador (iris/julia/BPE.jl, ids 1..vocab):
  consumir_buffer(dataset, desde_chunk) -> itera lineas JSON con campo "text".
  Contrato exacto en IRIS_F4B_PIPELINE.md.
"""
import gzip, io, json, os, sys, time, re
from pathlib import Path
import urllib.request
import fsspec
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parent
BUFFER = Path(os.environ.get("IRIS_BUFFER", RAIZ / "buffer"))
CURSORS = Path(os.environ.get("IRIS_CURSORS", RAIZ / "cursors"))
BUFFER.mkdir(exist_ok=True)
CURSORS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- datasets
CHUNK_FILAS = 20000          # filas por chunk jsonl.gz (~30-40 MB texto)
TOPE_GB = 0.5                # tope del buffer por dataset (comprimido, en disco)

DATASETS = {
    "wikipedia_es": dict(
        ds="wikimedia/wikipedia", cfg="20231101.es", split="train",
        cols=["id", "url", "title", "text"]),
    "wikipedia_en": dict(
        ds="wikimedia/wikipedia", cfg="20231101.en", split="train",
        cols=["id", "url", "title", "text"]),
    "fineweb_edu": dict(
        ds="HuggingFaceFW/fineweb-edu", cfg="sample-10BT", split="train",
        cols=["text", "id", "url", "language", "language_score", "int_score", "token_count"]),
    "c4_es": dict(
        ds="allenai/c4", cfg="es", split="train",
        cols=["text", "url", "timestamp"]),
    "oasst2": dict(
        ds="OpenAssistant/oasst2", cfg="default", split="train",
        cols=["message_id", "parent_id", "text", "role", "lang",
              "review_result", "deleted", "synthetic"]),
}
LISTA_URL = "https://huggingface.co/api/datasets/{ds}/parquet/{cfg}/{split}"

# ---------------------------------------------------------------- filtros
MIN_CHARS = 200
pat_web_ruido = re.compile(r"^(\s*\|.*\|)|(figura|imagen|archivo):", re.I)

def filtro_wikipedia(rec):
    t = rec.get("text", "")
    if len(t) < MIN_CHARS or len(t) > 200_000:
        return False
    if pat_web_ruido.match(t[:80]):
        return False
    return True

def filtro_fineweb(rec):
    if rec.get("language") != "en":
        return False
    if (rec.get("language_score") or 0) < 0.85:
        return False
    if (rec.get("int_score") or 0) < 3:      # calidad educativa >= 3/5
        return False
    t = rec.get("text", "")
    return MIN_CHARS <= len(t) <= 200_000

def filtro_c4(rec):
    t = rec.get("text", "")
    if len(t) < MIN_CHARS or len(t) > 200_000:
        return False
    if pat_web_ruido.match(t[:80]):
        return False
    return True

def filtro_oasst2(rec):
    if rec.get("deleted"):
        return False
    if rec.get("review_result") is False:
        return False
    if rec.get("lang") not in ("es", "en"):
        return False
    return len(rec.get("text", "")) >= 20

FILTROS = {
    "wikipedia_es": filtro_wikipedia,
    "wikipedia_en": filtro_wikipedia,
    "fineweb_edu": filtro_fineweb,
    "c4_es": filtro_c4,
    "oasst2": filtro_oasst2,
}

# ---------------------------------------------------------------- listado parquet
_cache_listas = {}

def lista_parquets(nombre):
    if nombre not in _cache_listas:
        d = DATASETS[nombre]
        url = LISTA_URL.format(ds=d["ds"], cfg=d["cfg"], split=d["split"])
        req = urllib.request.Request(url, headers={"User-Agent": "iris-f4b/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            _cache_listas[nombre] = json.load(r)
    return _cache_listas[nombre]

# ---------------------------------------------------------------- cursor
def ruta_cursor(nombre):
    return CURSORS / f"{nombre}.cursor.json"

def cargar_cursor(nombre):
    p = ruta_cursor(nombre)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"file_idx": 0, "row_group_idx": 0, "chunks_escritos": 0, "filas_utiles": 0,
            "filas_leidas": 0, "actualizado": ""}

def guardar_cursor(nombre, cur):
    cur["actualizado"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = ruta_cursor(nombre).with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ruta_cursor(nombre))   # atomico: escribir temp + rename

# ---------------------------------------------------------------- buffer
def ruta_chunk(nombre, n):
    d = BUFFER / nombre
    d.mkdir(exist_ok=True)
    return d / f"{nombre}_{n:06d}.jsonl.gz"

def bytes_dataset(nombre):
    d = BUFFER / nombre
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.glob("*.jsonl.gz"))

def tope_alcanzado(nombre):
    return bytes_dataset(nombre) >= TOPE_GB * 1e9

# ---------------------------------------------------------------- escritura de chunks
def escribir_chunk(nombre, filas, n_chunk):
    """filas: lista de dicts ya filtrados. Escribe jsonl.gz determinista (mtime=0)."""
    ruta = ruta_chunk(nombre, n_chunk)
    tmp = ruta.with_suffix(".tmp")
    raw = gzip.GzipFile(filename="", mode="wb", fileobj=open(str(tmp), "wb"),
                        compresslevel=6, mtime=0)
    with raw as gz, io.TextIOWrapper(gz, encoding="utf-8") as w:
        for rec in filas:
            w.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    tmp.replace(ruta)   # atomico

# ---------------------------------------------------------------- lectura streaming
def row_groups(nombre, file_idx, rg_inicio, limite_rg):
    """Generador: lee row-groups de un parquet por HTTP Range. NUNCA baja el archivo entero."""
    files = lista_parquets(nombre)
    if file_idx >= len(files):
        return
    fs = fsspec.filesystem("http")
    d = DATASETS[nombre]
    pf = pq.ParquetFile(fs.open(files[file_idx]))
    for rg in range(rg_inicio, min(pf.metadata.num_row_groups, rg_inicio + limite_rg)):
        tabla = pf.read_row_group(rg, columns=d["cols"])
        yield tabla.to_pylist()

_rg_cache = {}

def num_rg(nombre, file_idx):
    """Num row groups del parquet file_idx (footer por Range, cacheado)."""
    key = (nombre, file_idx)
    if key not in _rg_cache:
        files = lista_parquets(nombre)
        fs = fsspec.filesystem("http")
        pf = pq.ParquetFile(fs.open(files[file_idx]))
        _rg_cache[key] = pf.metadata.num_row_groups
    return _rg_cache[key]

# ---------------------------------------------------------------- ciclo de ingesta
def _flush(nombre, cur, pendiente, force):
    """Escribe pendiente a chunk si alcanza CHUNK_FILAS (o si force al cerrar).
    El cursor avanza SOLO a los row-groups cuyas filas ya estan persistidas:
    si el proceso muere, se re-lee desde el ultimo RG persistido (sin duplicados ni huecos)."""
    if not pendiente["filas"]:
        return 0
    if not (force or len(pendiente["filas"]) >= CHUNK_FILAS):
        return 0
    escribir_chunk(nombre, pendiente["filas"], cur["chunks_escritos"])
    cur["chunks_escritos"] += 1
    n = len(pendiente["filas"])
    cur["filas_utiles"] += n
    pendiente["filas"] = []
    cur["row_group_idx"] = pendiente["rg_fin"]        # primer RG no persistido
    guardar_cursor(nombre, cur)
    return n

def correr_ingesta(nombres, segundos, limite_rg_por_paso=30, log=print):
    """Lee de varios datasets en rotacion hasta agotar `segundos`. Reanudable."""
    fin = time.time() + segundos
    stats = {n: dict(rg=0, filas_leidas=0, utiles=0) for n in nombres}
    pendientes = {n: dict(filas=[], rg_fin=None) for n in nombres}
    orden = list(nombres)
    i = 0
    try:
        while time.time() < fin:
            nombre = orden[i % len(orden)]
            i += 1
            if tope_alcanzado(nombre):
                log(f"[{nombre}] buffer en tope ({bytes_dataset(nombre)/1e6:.0f} MB), skip")
                continue
            cur = cargar_cursor(nombre)
            files = lista_parquets(nombre)
            if cur["file_idx"] >= len(files):
                log(f"[{nombre}] sin mas parquets")
                continue
            pend = pendientes[nombre]
            pend["rg_fin"] = cur["row_group_idx"]
            filtro = FILTROS[nombre]
            while time.time() < fin:
                if cur["file_idx"] >= len(files):
                    break
                fin_rg_archivo = num_rg(nombre, cur["file_idx"])
                if cur["row_group_idx"] >= fin_rg_archivo:
                    cur["file_idx"] += 1
                    cur["row_group_idx"] = 0
                    pend["rg_fin"] = 0
                    guardar_cursor(nombre, cur)
                    log(f"[{nombre}] parquet {cur['file_idx']-1} completo")
                    continue
                n_leer = min(limite_rg_por_paso, fin_rg_archivo - cur["row_group_idx"])
                for rg_filas in row_groups(nombre, cur["file_idx"], cur["row_group_idx"], n_leer):
                    utiles = [r for r in rg_filas if filtro(r)]
                    pend["filas"].extend(utiles)
                    pend["rg_fin"] += 1
                    stats[nombre]["filas_leidas"] += len(rg_filas)
                    stats[nombre]["utiles"] += len(utiles)
                stats[nombre]["rg"] += n_leer
                escritas = _flush(nombre, cur, pend, force=False)
                if escritas:
                    log(f"[{nombre}] chunk {cur['chunks_escritos']-1}: {escritas} filas, "
                        f"buffer {bytes_dataset(nombre)/1e6:.0f} MB")
                if time.time() >= fin or tope_alcanzado(nombre):
                    break
    finally:
        # cierre: persistir lo pendiente de TODOS los datasets y avanzar cursores
        for nombre, pend in pendientes.items():
            cur = cargar_cursor(nombre)
            n = _flush(nombre, cur, pend, force=True)
            if n:
                log(f"[cierre {nombre}] flush {n} filas -> chunk {cur['chunks_escritos']-1}")
    return stats

# ---------------------------------------------------------------- consumo (interfaz tokenizador)
def consumir_buffer(nombre, desde_chunk=0, max_filas=None):
    """Itera las filas del buffer en orden FIFO de chunks.
    Uso: for rec in consumir_buffer('wikipedia_es', chunk_ya_consumido)."""
    d = BUFFER / nombre
    if not d.exists():
        return
    n = 0
    for f in sorted(d.glob("*.jsonl.gz")):
        num = int(f.name.rsplit("_", 1)[1].split(".")[0])
        if num < desde_chunk:
            continue
        with gzip.open(f, "rt", encoding="utf-8") as r:
            for line in r:
                yield json.loads(line)
                n += 1
                if max_filas and n >= max_filas:
                    return

if __name__ == "__main__":
    nombre = sys.argv[1] if len(sys.argv) > 1 else "wikipedia_es"
    seg = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    s = correr_ingesta([nombre], seg)
    print("stats:", s)
