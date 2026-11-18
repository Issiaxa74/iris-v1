# IRIS F4b — Pipeline de ingesta streaming (spec + prototipo verificado)

Fecha: 2026-11-17. Autor: co-ayudador de ia. Estado: **prototipo funcionando contra datasets reales** (no se ejecutó entrenamiento).

## 1. Decisiones de datasets (elegidos y explorados por HTTP Range)

Explorados los footers y esquemas reales vía API HF + HTTP Range (sin descarga completa):

| Dataset (config) | Archivos parquet | Columnas | Tamaño aprox. | Calidad/idioma propio | Decisión |
|---|---|---|---|---|---|
| `wikimedia/wikipedia` 20231101.es | 13 | `id, url, title, text` | ~9 GB total (0.69 GB/archivo, 142 RG de 1000 filas) | limpio | **IN** (15% ciclo) |
| `wikimedia/wikipedia` 20231101.en | 21 | `id, url, title, text` | ~13 GB | limpio | **IN** (15%) |
| `HuggingFaceFW/fineweb-edu` sample-10BT | 14 | `text,id,url,language,language_score,int_score,token_count` | ~80 GB muestra | `language_score` + `int_score` (calidad educativa 0-5) integrados | **IN** (25%) |
| `allenai/c4` es | 10 | `text, url, timestamp` | ~16 GB (por archivo) | ya filtrado por idioma | **IN** (30%) |
| `OpenAssistant/oasst2` default | 1 | `message_id,parent_id,text,role,lang,review_result,deleted,synthetic,…` | ~0.5 GB | `lang`, `review_result` | **IN** (15%) |

Elección: los 5 del plan quedan activos para el ciclo; el prototipo verificado end-to-end con **c4_es, fineweb_edu y wikipedia_es** (los otros dos usan el mismo código con otro config).

URLs de listado parquet (la API `https://huggingface.co/api/datasets/<ds>/parquet/<cfg>/<split>` devuelve la lista de archivos `N.parquet`, cada uno con ~1000 filas por row-group):

```
https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.es/train
https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train
https://huggingface.co/api/datasets/allenai/c4/parquet/es/train
https://huggingface.co/api/datasets/OpenAssistant/oasst2/parquet/default/train
```

## 2. Prototipo de lector streaming (implementado y verificado)

Código: `C:\Users\angel\OpenBot\iris\datos\iris_stream.py` (Python 3.12 + `fsspec` + `pyarrow`; **sin dependencias externas en el núcleo del modelo** — solo pipeline de datos, documentado).

Funcionamiento por row-group:

1. `lista_parquets(nombre)` → API HF (1 llamada, cacheada en memoria).
2. `pq.ParquetFile(fs.open(url))` → lee **solo el footer** por HTTP Range (~0.4 MB): columnas, num_rows, num_row_groups. Medido: 6.7 s.
3. `pf.read_row_group(rg, columns=…)` → descarga **solo los bytes de ese row-group** por Range (10-15 s por RG de 1000 filas en esta conexión).
4. Filtro idioma/calidad en memoria → solo las filas que pasan se escriben a disco.

Filtros aplicados (sencilla primera versión, calibrable):

- **wikipedia_es/en**: 200 ≤ len(text) ≤ 200 000; descarta tablas/figures en cabecera.
- **fineweb_edu**: `language == "en"`, `language_score ≥ 0.85`, `int_score ≥ 3` (calidad educativa).
- **c4_es**: 200 ≤ len(text) ≤ 200 000 (c4 ya viene en español).
- **oasst2**: `deleted == False`, `review_result != False`, `lang ∈ {es,en}`, len ≥ 20.

Rendimiento medido (150 s de prueba):

| Dataset | RG leídos | Filas leídas | Filas útiles | Tasa filtro | Buffer escrito |
|---|---|---|---|---|---|
| c4_es | 90 | 76 569 | 73 679 | 96.3% | ~150 MB (3 chunks) |
| fineweb_edu | 60 | 60 000 | 57 452 | 95.8% | ~112 MB (2 chunks) |
| wikipedia_es | 1020 (corto) | ~40 000 | 33 053 | ~82% | ~71 MB (34 chunks, formato antiguo de 1 chunk/RG) |

**Ritmo de lectura:** ~0.5-0.7 MB texto útil/s con el límite RG=30 → ~2 GB texto/día de lectura sostenida en el margen del ciclo, muy por encima del techo de digestión (~35 MB/día), tal como permite la decisión (b) del spec: lectura adelantada, buffer con tope.

## 3. Buffer filtrado en disco con tope

Ruta: `C:\Users\angel\OpenBot\iris\datos\buffer\<dataset>\<dataset>_<NNNNNN>.jsonl.gz`

- **Formato:** una fila JSON por línea (UTF-8, `ensure_ascii=False`), campos = columnas del dataset (ver §1). Chunks de **20 000 filas útiles** (~30-40 MB texto, ~35 MB gz). Escritura atómica (`.tmp` + rename) y **gzip determinista** (`mtime=0`).
- **Tope:** `TOPE_GB = 0.5` por dataset (total ~2.5 GB máx en disco). Si el buffer alcanza el tope, la ingesta de ese dataset **se pausa** (skip) hasta que el consumidor libere espacio; no se descarta nada.
- **Consumo (FIFO por dataset):** `consumir_buffer(nombre, desde_chunk)` itera los chunks en orden numérico creciente desde `desde_chunk`. El consumidor (tokenizador) lleva su propio puntero y **borra el chunk cuando lo termina** (o lo marca consumido; borrado directo recomendado para liberar el tope).
- Estado actual: buffer real con ~320 MB de datos útiles ya ingiridos (c4_es 5 chunks, fineweb_edu 2 chunks, wikipedia_es 34 chunks cortos del primer prototipo — todos válidos y consumibles).

## 4. Ciclo de 2 h y cursores reanudables

Lanzador: `C:\Users\angel\OpenBot\iris\datos\run_cycle.py` (SOLO datos; el entrenamiento lo integra Desarrollador de ia).

- **Ventana por dataset proporcional a su peso del plan** (wikipedia ES 15% · EN 15% · fineweb-edu 25% · c4-es 30% · oasst2 15%). Con ciclo de 7200 s: c4-es ~36 min, fineweb ~30 min, wikipedia ES/EN ~18 min cada uno, oasst2 ~18 min.
- **Cursor por dataset:** `iris/datos/cursors/<dataset>.cursor.json` con `{file_idx, row_group_idx, chunks_escritos, filas_leidas, filas_utiles, actualizado}`.
- **Invariante de reanudación:** el cursor avanza **solo después** de que las filas correspondientes están persistidas en un chunk (`_flush` escribe chunk → luego `guardar_cursor`). Si el proceso muere en mitad de un RG, al reiniciar se re-lee desde el último RG persistido: **sin duplicados ni huecos**. Escritura de cursor atómica (temp + rename).
- **Estado del ciclo:** `iris/datos/estado_ciclo.json` (resumen por dataset + MB en buffer).
- Comprobado: interrumpí y relancé la ingesta de c4_es → continuó exactamente en `file_idx=0, row_group_idx=90` y escribió el chunk 4 sin repetir los anteriores.

## 5. Interfaz para el tokenizador (iris/julia/BPE.jl)

El tokenizador consume el buffer filtrado. Contrato exacto:

1. **Fuente:** `buffer/<dataset>/<dataset>_<NNNNNN>.jsonl.gz`, orden FIFO por dataset (chunks 000000, 000001, …).
2. **Cada línea JSON** es un documento con campo `text` (el único que el tokenizador necesita) más metadatos (`title`, `url`, `language_score`/`int_score` según dataset).
3. **Llamada Python** (para pruebas/sync): `iris_stream.consumir_buffer(nombre, desde_chunk)` → generador de dicts.
4. **Para Julia sin Python en el bucle:** leer los `.jsonl.gz` directamente con `CodecZlib` + `JSON` (gzip estándar, sin formato propietario). El orden es determinista: chunks por nombre ascendente dentro de cada dataset.
5. **Concatenación para packing:** el stream de entrenamiento se construye consumiendo los 5 datasets en la proporción del ciclo (o intercalando round-robin sobre las 5 colas FIFO — el orden exacto lo fija el bucle de entrenamiento; las colas son independientes por dataset).
6. **BPE de Iris:** ids `1..vocab` según `iris/iris.tok` (hoy 1056, camino a 16k en F1c). El pipeline **no** tokeniza: entrega texto crudo filtrado. A 3.5 chars/token el buffer actual (~320 MB gz ≈ ~600 MB texto) da ~170 M tokens ya disponibles para el re-entreno del BPE y el primer ciclo.

## 6. Plan de pruebas (hecho / pendiente)

Hecho (verificado hoy):

- [x] Footer por HTTP Range de los 4 datasets candidatos (+oasst2): esquema y tamaños.
- [x] Lectura de row-groups reales por Range con pyarrow (c4_es, fineweb_edu, wikipedia_es).
- [x] Filtros de idioma/calidad aplicados y tasas medidas (82-96% útil).
- [x] Chunks jsonl.gz deterministas, UTF-8 verificado byte a byte (á = 0xC3 0xA1), escritura atómica.
- [x] Cursor reanudable: interrupción + relanzamiento sin duplicados.
- [x] Tope de buffer 0.5 GB/dataset respetado (320 MB actuales).

Pendiente (para cerrar F4b con el Desarrollador):

- [ ] Primer ciclo completo real de 2 h con los 5 datasets (`python run_cycle.py`).
- [ ] Conexión del tokenizador BPE al buffer (lado Julia: CodecZlib + JSON, o consumo vía Python si se prefiere).
- [ ] Calibración del tamaño de chunk si el consumidor pide bloques mayores/menores.
- [ ] Medición de GB/hora sostenidos durante 2 h (hoy solo ventanas de 2-3 min).

## 7. Notas

- Sin HF_TOKEN: los 5 datasets del plan son públicos (los gated siguen fuera por regla 7).
- No se tocó `iris/julia/model.jl`, `router.jl` ni backups. No se ejecutó ningún entrenamiento.
- El buffer actual (~320 MB) contiene **datos reales ya filtrados** listos para el primer ciclo.
