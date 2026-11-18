# Datos

Iris se entrena con next-token prediction sobre un stream de texto filtrado. No hay instrucciones, ni diálogos etiquetados, ni destilación: solo texto.

## 1. Fuentes

Todo el pipeline lee Parquet públicos de Hugging Face **sin descargar el dataset completo**: pide el listado de archivos, lee el footer de cada Parquet por HTTP Range (~0,4 MB), localiza los row-groups y descarga solo los que consume.

| Nombre interno | Dataset HF | Config | Split | Peso |
|---|---|---|---|---|
| `wikipedia_es` | `wikimedia/wikipedia` | `20231101.es` | train | 15 % |
| `wikipedia_en` | `wikimedia/wikipedia` | `20231101.en` | train | 15 % |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | train | 25 % |
| `c4_es` | `allenai/c4` | `es` | train | 30 % |
| `oasst2` | `OpenAssistant/oasst2` | `default` | train | 15 % |

Los row-groups son de 1000 filas en los cinco datasets, así que el coste por Range es pequeño y constante.

## 2. Filtros

Aplicados por el pipeline antes de escribir el buffer (`src/python/iris_stream.py`):

| Dataset | Filtro |
|---|---|
| Wikipedia ES/EN | texto de 200 a 200 000 caracteres; descarta tablas y pies de figura (`^\s*\|...\|`, `figura:`, `imagen:`, `archivo:`) |
| FineWeb-Edu | `language == "en"`, `language_score ≥ 0,85`, `int_score ≥ 3`, longitud 200–200 000 |
| C4 ES | longitud 200–200 000; mismo filtro de ruido web |
| oasst2 | no borrado, `review_result` no `False`, idioma es/en, ≥ 20 caracteres |

El texto se guarda tal cual en el campo `text` de un JSON por línea; el tokenizador es quien decide qué es un token.

## 3. Buffer

```
datos/buffer/<dataset>/<dataset>_<NNNNNN>.jsonl.gz
```

- Chunks de **20 000 filas** (~30–40 MB de texto, comprimido con gzip determinista `mtime=0`).
- Escritura **atómica**: se escribe a `.tmp` y se renombra; nunca hay chunks a medias.
- **Tope de 0,5 GB por dataset**. Al alcanzarlo, la ingesta de ese dataset se pausa hasta que el consumidor borre chunks.
- **FIFO**: el consumidor lee el chunk más antiguo; el entrenador lo borra **solo cuando lo termina**.

## 4. Cursores y reanudación

`datos/cursors/<dataset>.cursor.json` guarda:

| Campo | Significado |
|---|---|
| `chunk` | número de chunk crudo dentro del dataset |
| `line` | líneas crudas consumidas del chunk actual |
| `tokens` | tokens leídos históricos (contador) |

La invariante es que el cursor **solo avanza después de persistir las filas**. Un corte a mitad de chunk se reanuda en la línea exacta sin duplicar ni perder; se verificó relanzando c4 tras una interrupción (continuó en el row-group 90 sin duplicados).

En el entrenamiento, el cursor va hasta 5 minutos por delante del checkpoint: tras un corte, esos tokens se leen pero no se entrenan. No corrompe nada; solo se pierde ese tramo.

## 5. Ventanas de ingesta

`run_cycle.py` reparte la ventana del ciclo de forma proporcional al peso de cada dataset y rota hasta agotar el tiempo: c4 30 %, fineweb 25 %, wikipedia ES/EN 15+15 %, oasst2 15 %. Al terminar escribe `estado_ciclo.json` con estadísticas por fuente.

Rendimiento medido del lector: **0,5–0,7 MB de texto útil por segundo** (limitado por descarga y descompresión, no por CPU). En 2 h deja ~320 MB de buffer listo para entrenar.

## 6. Uso

```powershell
$env:IRIS_BUFFER="$PWD\datos\buffer"
$env:IRIS_CURSORS="$PWD\datos\cursors"

# ciclo completo de 2 h
python src/python/run_cycle.py 7200

# explorar esquemas y tamaños de un dataset sin ingerir
python src/python/hf_explore.py fineweb_edu
```

Variables: `IRIS_BUFFER` y `IRIS_CURSORS` cambian dónde se escribe. Si no se definen, se usan `src/python/buffer` y `src/python/cursors`.
