# AGENTS.md — instrucciones para agentes de IA

Este repositorio contiene **Iris v1**, un modelo de lenguaje escrito desde cero:
núcleo de tensores en C, entrenamiento en Julia, checkpoint propio (`IRISMDL`),
formato de modelo `UKJ` y runtime de inferencia en Rust (`cabrita`). No usa
PyTorch, transformers ni servicios externos.

Antes de ejecutar nada, lee:

1. `README.md` — qué es y cifras reales.
2. `docs/ESTADO.md` — qué funciona, qué no y por qué.
3. `docs/ENTRENAMIENTO.md` — procedimiento completo (datos → checkpoint → UKJ → texto).
4. `docs/AGENTES.md` — protocolo detallado, puertas de verificación y plantilla de informe.

## Estado honesto

- El modelo **no habla**: ~1,30 M de tokens, loss 5,77, salida `e e e`. Es el
  estado esperado, no un bug.
- 406 976 512 parámetros totales; 192 997 376 activos (trunk). Las 16 hojas y
  `sft` son peso muerto en el forward.
- El runtime ya tiene forward real con caché KV (`--engine transformer`).
- El checkpoint real (1,63 GB) **no está en el repo**; se regenera entrenando o se
  publica como release asset.

## Comandos de verificación

```bash
make -C src/c test                                  # 19/19 PASS
julia src/julia/model_test.jl                       # TODO OK (~2 min)
cd runtime && cargo test --release                  # 50 + 7, 0 fallos
```

Inferencia con el fixture diminuto incluido:

```bash
cd runtime
cargo run --release -- --engine transformer \
  --model assets/tiny.ukj --tok assets/iris.tok \
  --ask "hola mundo" --max-new 8 --temp 0 --seed 7
```

Entrenamiento (requiere buffer; ver `docs/ENTRENAMIENTO.md`):

```bash
export IRIS_BUFFER="$PWD/datos/buffer"
export IRIS_TOK="$PWD/assets/iris.tok"
julia --project=. src/julia/f4b_cycle.jl 120 models/model.irmdl src/julia/f4b_cursor.json
```

Exportación y runtime:

```bash
julia src/julia/export_ukj.jl models/model.irmdl models/model.ukj p4-one
cd runtime
cargo run --release -- --engine transformer --model ../models/model.ukj --tok ../assets/iris.tok --ask "hola"
```

## Reglas que no se negocian

- **No borres ni sobrescribas** checkpoints, buffers o datos que no hayas creado.
  Trabaja sobre una copia o rama.
- **No lances dos ciclos de entrenamiento a la vez**: comparten checkpoint.
- **No cambies el tokenizador ni `vocab`** y sigas entrenando el mismo `.irmdl`;
  los ids dejan de significar lo mismo.
- **No toques `model.jl` sin ejecutar `model_test.jl`**: el gradiente es manual.
- **No afirmes que algo funciona sin la salida del comando** que lo demuestra.
- **No inventes cifras**: loss, tokens, tok/s y tamaños salen de la salida real.
- **No subas archivos >100 MB** al repo; usa un release asset (máx. 2 GB).
- **No mates procesos por nombre** (`node`, `python`, `julia`) para liberar memoria.
- Pregunta antes de cualquier ejecución de más de 30 minutos.

## Mapa del repo

```
src/c/              núcleo de tensores + lector UKJ mínimo
src/julia/          modelo, entrenamiento, BPE, router, export UKJ
src/python/         ingesta de datos por HTTP Range y buffer
src/rust/ukj_reader lector UKJ de referencia
runtime/            cabrita: REPL, router, TOJI, motor transformer
assets/iris.tok     tokenizador entrenado (1056 tokens)
docs/               arquitectura, datos, UKJ, router, TOJI, estado, manual, agentes
models/             router entrenado (el checkpoint grande no se versiona)
```

## Puertas antes de dar una tarea por terminada

| Fase | Éxito |
|---|---|
| C | `19/19 PASS` |
| Modelo | `TODO OK` |
| Runtime | 50 unitarias + 7 integración, 0 fallos |
| Datos | chunks `.jsonl.gz` en `datos/buffer` |
| Humo | `OK f4b_cycle`, loss finita |
| Export | `verificacion: OK` |
| Inferencia | texto + `trunk N capas` |
| Paridad | Julia y Rust dan la misma secuencia con `--temp 0` |

El protocolo completo, los playbooks y la plantilla de informe están en
[`docs/AGENTES.md`](docs/AGENTES.md).
