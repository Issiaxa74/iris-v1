# Iris v1

Un modelo de lenguaje escrito **desde cero**: núcleo de tensores en C, entrenamiento en Julia, formato de modelo propio (UKJ) y runtime de inferencia en Rust. Sin PyTorch, sin transformers, sin dependencias de terceros en el camino del modelo.

Este repositorio contiene **todo el código del proyecto** y un manual para que cualquiera lo reproduzca y lo entrene en su propia máquina.

> **Estado honesto:** Iris **todavía no habla**. La infraestructura está completa y verificada (modelo, tokenizador, router, formato UKJ, runtime con forward real y caché KV), pero el modelo solo ha visto ~1,3 millones de tokens y su loss media acumulada es 5,77 (perplejidad ≈ 320). Producirá texto incoherente (`e e e e`). Los números y los límites están en [Estado real](#estado-real-y-por-qué-no-habla-todavía).

---

## Qué es Iris

Iris es una arquitectura pensada para crecer por **rutas de conocimiento** en lugar de un único bloque monolítico:

| Pieza | Qué hace | Estado |
|---|---|---|
| **Trunk** | 14 capas transformer (dim 1024, 16 cabezas, FFN 4096, ctx 2048, vocab 16384) | Entrena y funciona |
| **Hojas (16)** | Bloques expertos por ruta de conocimiento | Asignados en memoria, **sin usar en el forward** |
| **Router** | Clasificador ligero que elige ruta por contenido (bag-of-embeddings + MLP) | Entrenado: 22 rutas, τ=0,85 |
| **BPE** | Tokenizador propio por bytes, 1056 tokens | Funcional |
| **UKJ** | Formato contenedor propio (tipo GGUF) con tabla de rutas y mmap | Escritor Julia + lectores Rust y C |
| **TOJI** | Memoria viva por sectores (`.mem`) | Especificado y con implementación base en el runtime |
| **cabrita** | REPL/runtime en Rust que lee UKJ y conversa | Forward real con caché KV, 57 tests propios |

La idea de los sub-parámetros (`p4-one`: 4 sub-parámetros comprimidos por parámetro, que se liberan bajo estrés) está especificada en [docs/UKJ.md](docs/UKJ.md) y [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md); hoy el modelo entrena en fp32 plano y la cuantización solo existe al serializar.

## Estado real (y por qué no habla todavía)

Medido en un Ryzen 7 5825U (8 núcleos, sin GPU), Windows 11, Julia 1.13:

| Métrica | Valor |
|---|---|
| Parámetros en el checkpoint | 406.976.512 |
| Parámetros **activos** en el forward | **192.997.376** (trunk + embeddings + LN final) |
| Peso en disco (fp32) | 1,63 GB |
| Tokens entrenados | ~1,30 M |
| Loss | 5,77 nats → perplejidad ≈ 320 |
| Baseline uniforme (ln 16384) | 9,70 nats |
| Velocidad de entrenamiento | 108 tok/s sostenidos (B=2, T=256) |
| RAM de entrenamiento | 7,3 GB RSS |
| Forward solo | ~500 tok/s (T=256) |

**La cuenta que importa:** con 193 M de parámetros activos, la regla habitual (~20 tokens por parámetro) pide **3.860 M de tokens**. Con 1,3 M vas por el 0,03 %. A 108 tok/s eso son **~412 días de CPU continua**. Por eso el proyecto está publicado: en esta máquina no se llega, en una GPU sí.

Tres defectos conocidos que frenan el entrenamiento y están sin arreglar:

1. **Vocabulario sobredimensionado.** El tokenizador usa 1056 ids pero el modelo reserva 16384. El 94 % de la matriz de logits y de los embeddings es lastre: reducir `vocab` a 1056 ahorra ~15,7 M de parámetros y acelera el GEMM más grande del forward ~15×.
2. **Las 16 hojas y el bloque `sft` no se usan.** Ocupan 213.979.136 parámetros (53 % del checkpoint) que nunca entran en el forward ni reciben gradiente. El modelo que realmente corre es un transformer de 14 capas.
3. **El muestreador de Julia no tiene caché KV.** Cada token generado repite el forward completo (O(T²)); el runtime `cabrita` sí la tiene y genera en O(T).

La media acumulada de la loss terminó en 5,77. El ciclo imprime la media desde el inicio, no la loss por paso, así que no se puede afirmar que se estancara: falta instrumentar la loss por ventana. Si de verdad se estanca, los candidatos son learning rate, datos y calibración de Adam.

## Estructura del repositorio

```
src/c/                 núcleo de tensores + lector UKJ mínimo (C11)
src/julia/             modelo, entrenamiento, router, tokenizador, UKJ
src/python/            pipeline de datos: HF por HTTP Range, buffer, cursores
src/rust/ukj_reader/   lector UKJ de referencia en Rust
runtime/               cabrita: REPL, router, TOJI, UKJ con mmap (Rust)
assets/iris.tok        tokenizador entrenado (1056 tokens)
docs/                  arquitectura, formato UKJ, datos, router, TOJI, estado
```

## Requisitos

- **Julia 1.10+** (probado en 1.13.3) con el paquete `CodecZlib` (solo el pipeline de datos)
- **Python 3.10+** con `pyarrow` y `fsspec`
- **Rust 1.75+** (solo runtime y lector UKJ)
- **GCC/Clang o MSVC** (solo núcleo C)
- 8 GB de RAM libres para entrenar el modelo completo; ~2 GB para inferir

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
pip install -r requirements.txt
```

## Inicio rápido

```powershell
# 1. Núcleo C: compila y pasa sus tests
make -C src/c test          # 19/19 PASS

# 2. Modelo: test de gradiente, causalidad y save/load (~2 min)
julia src/julia/model_test.jl

# 3. Tokenizador + escritor UKJ
julia src/julia/test_writer.jl

# 4. Generar texto con un checkpoint (ver "Entrenar" si aún no tienes uno)
$env:IRIS_CKPT="models/model.irmdl"
julia src/julia/generate.jl --prompt "El cielo es" --tokens 40 --temp 0.9

# 5. O con el runtime (caché KV): exporta a UKJ y arranca cabrita
julia src/julia/export_ukj.jl models\model.irmdl models\model.ukj
cd runtime
cargo run --release -- --engine transformer --model ..\models\model.ukj --tok ..\assets\iris.tok --ask "El cielo es"
```

Si no tienes checkpoint, `generate.jl` falla con un mensaje claro: primero entrena (siguiente sección) o crea uno aleatorio con `julia src/julia/model_test.jl` y `save_model`.

## Manual: entrenar el modelo desde cero

El manual completo —requisitos, contrato del buffer, hiperparámetros, diagnóstico,
exportación, inferencia y recetas para escalar— está en
[**docs/ENTRENAMIENTO.md**](docs/ENTRENAMIENTO.md). Esta sección es el resumen
ejecutivo.

### 1. Entender el dato que come

El entrenamiento es **next-token prediction** puro sobre texto. No hay destilación: el proyecto rechaza aprender a hablar copiando las salidas de otro modelo. El stream se compone de cinco fuentes públicas:

| Dataset | Peso | Filtro |
|---|---|---|
| `c4` (es) | 30 % | longitud de texto |
| `fineweb-edu` (sample-10BT) | 25 % | `language_score ≥ 0.85`, `int_score ≥ 3` |
| `wikipedia` (es) | 15 % | 200–200 000 caracteres |
| `wikipedia` (en) | 15 % | ídem |
| `oasst2` | 15 % | `lang` es/en, `review_result` válido |

Todos son públicos y se leen **sin descargar el dataset entero**: el pipeline pide el footer del Parquet por HTTP Range (~0,4 MB), localiza los row-groups y descarga solo lo que consume (`src/python/iris_stream.py`).

### 2. Preparar el buffer de datos

```powershell
# Lee ~2 h de datos y deja chunks jsonl.gz filtrados en datos/buffer/<dataset>/
$env:IRIS_BUFFER="$PWD\datos\buffer"
$env:IRIS_CURSORS="$PWD\datos\cursors"
python src/python/run_cycle.py 7200      # el argumento son segundos, no minutos
```

- Chunks de 20 000 filas (~35 MB) comprimidos con gzip determinista.
- Tope de 0,5 GB por dataset; al llegar, la ingesta se pausa hasta que el entrenador borre chunks.
- Cursor por dataset en `datos/cursors/*.json`: solo avanza **después** de persistir las filas, así que un corte no duplica ni pierde datos.

### 3. Entrenar

```powershell
$env:IRIS_BUFFER="$PWD\datos\buffer"
$env:IRIS_TOK="$PWD\assets\iris.tok"
New-Item -ItemType Directory -Force models | Out-Null

# ciclo de 2 h; guarda checkpoint cada 15 min y cursor cada 5
julia --project=. src/julia/f4b_cycle.jl 120 models\model.irmdl src\julia\f4b_cursor.json
```

Qué hace cada paso: toma el dataset con menor `(tokens servidos / peso)`, tokeniza con BPE, empaqueta en secuencias de T=256, entrena micro-batches de B=2 y aplica AdamW **solo a los parámetros entrenables de la fase 1** (trunk + una hoja + `sft`). En la práctica solo el trunk recibe gradiente: las hojas y `sft` no entran en el forward. Al terminar un chunk lo borra y avanza el cursor.

Para reanudar: relanza el mismo comando; el checkpoint y el cursor mandan. Para ciclos encadenados, programa el comando cada 2 h (Programador de tareas en Windows, `cron` en Linux).

**Qué esperar:** ~108 tok/s, ~0,87 MB de texto por hora, loss bajando despacio. Con 1,3 M de tokens el modelo sigue en la fase de aprender frecuencias de palabras y bigramas.

### 4. Verificar que el entrenamiento es sano

```powershell
# test del modelo: gradiente analítico vs numérico, causalidad, save/load, Adam
julia src/julia/model_test.jl

# estado del buffer/cursores
julia src/julia/check_cycle.jl
```

`model_test.jl` es la red de seguridad: comprueba por grupos de parámetros que el gradiente analítico coincide con diferencias centrales (rel ≤ 1e-3), que el modelo es causal (dos secuencias que difieren solo en el futuro dan la misma loss pasada) y que `save_model`/`load_model` es bit-exacto. Si tocas `model.jl`, ejecútalo antes de gastar horas de CPU.

### 5. Exportar a UKJ

```powershell
julia src/julia/export_ukj.jl models\model.irmdl models\model.ukj p4-one
# escribe model.ukj + model.ukj.json (manifiesto: nombres, offsets y tamaños)
```

El script agrupa los tensores en 18 rutas (trunk, 16 hojas, sft), escribe la tabla UKJ y **relee el archivo para comprobar byte a byte** que cada ruta coincide con el checkpoint. UKJ v0.1 no guarda formas ni nombres: el manifiesto JSON es obligatorio para reconstruir el modelo. El formato completo está en [docs/UKJ.md](docs/UKJ.md).

### 6. Hablar con el modelo

```powershell
# muestreador de referencia en Julia (sin caché KV)
julia src/julia/generate.jl --prompt "Habia una vez" --tokens 60 --temp 0.8 --top-p 0.95

# runtime cabrita (Rust): forward real con caché KV y REPL con /si /no /up /ent-my
cd runtime
cargo run --release -- --engine transformer --model ..\models\model.ukj --tok ..\assets\iris.tok
```

## Entrenar el router

El router decide a qué ruta de conocimiento mandar una petición. Es independiente del modelo grande: bag-of-embeddings sobre el texto + MLP, 110 358 parámetros, formato `IRISRTR`.

```powershell
# 1. el TSV de 674 muestras ya está en data/router_data_f4c.tsv;
#    router_extractor.jl regenera desde Shared/IRIS_DESTILADO, que no se incluye

# 2. entrenar y evaluar: split 70/15/15, 10 semillas, selección por cobertura
#    val a tau=0.6 con precision >= 0.98 (no sobrescribe el artifact incluido)
julia src/julia/router_test.jl data\router_data_f4c.tsv models\router_v0_retrained.irrtr

# 3. barrido de tau sobre el artifact incluido, sin entrenar
julia src/julia/router_tau_sweep.jl data\router_data_f4c.tsv models\router_v0.irrtr
```

Resultado del modelo incluido (674 muestras, 22 rutas): argmax 56,2 % en test, y con **τ=0,85 grabado en el artifact**: 28/28 en validación y 29/29 en test, cobertura 29 %, texto sin sentido → ruta genérica. Detalles y límites en [docs/ROUTER.md](docs/ROUTER.md).

## Números medidos

| Medida | Valor | Nota |
|---|---|---|
| Forward T=256 | ~500 tok/s | solo forward |
| Entrenar, B=1 | ~105 tok/s | fwd+bwd+Adam |
| Entrenar, B=2 | 108–129 tok/s | punto dulce |
| Entrenar, B=4 | 104–125 tok/s | RSS 8,6 GB |
| Ciclo de 2 h | 781 312 tokens | 1,75 MB de texto |
| Núcleo C | 19/19 tests | GEMM, LN, atención, máscara causal |
| Router | 22 rutas, 0,42 MB | exacto a τ=0,85 |
| Inferencia cabrita | ~19 tok/s | checkpoint real, 14 capas, 736 MB en RAM |
| Tests runtime | 50 unitarias + 7 integración | 0 fallos |

## Limitaciones y trabajo pendiente

- [ ] Reducir `vocab` de 16384 a 1056 (ahorro ~16 M par y ~15× en el GEMM de logits)
- [ ] Conectar las hojas al forward (hoy son peso muerto) o eliminarlas del layout
- [x] Caché KV en el runtime para generación O(T) en vez de O(T²)
- [ ] Instrumentar la loss por ventana (la media acumulada no diagnostica)
- [ ] Diagnosticar la posible meseta de la loss (lr / datos / Adam)
- [ ] Entrenar de verdad: GPU o modelo mucho más pequeño (20–50 M)
- [ ] TOJI completo (sectores, cables, neuronas anómalas) en cabrita
- [ ] Sub-parámetros `p4-one` reales: hoy solo existen en la serialización

## Artefactos incluidos

Además del código, el repo trae lo que produjo el proyecto:

| Ruta | Qué es |
|---|---|
| `models/router_v0.irrtr` | Router entrenado (F4c, 22 rutas, τ=0,85, exacto en val y test) |
| `models/router_v0_f3_backup.irrtr` | Router F3 (16 rutas), conservado por trazabilidad |
| `models/router_v0_f4d_backup.irrtr` | Router F4d descartado (precisión 94,7 % a τ=0,85) |
| `data/router_data_f4c.tsv` | 674 muestras de entrenamiento del router (texto + ruta) |
| `data/router_data_f4d.tsv` | 825 muestras del intento F4d |
| `logs/` | Logs reales de los ciclos de entrenamiento |
| `docs/especificaciones/` | Especificaciones originales (diseño, modelo, UKJ, TOJI, feedback, plan, pipeline, esquema de rutas) |
| `runtime/assets/tiny.ukj` | Fixture de 199 808 parámetros para el test golden Julia ↔ Rust |

**No incluido por tamaño:** el checkpoint `model_f4b.irmdl` (1,63 GB). GitHub rechaza archivos de más de 100 MB en el repo y el almacenamiento LFS gratuito es de 1 GB. Se regenera entrenando o se puede publicar como *release asset* (límite 2 GB por archivo). Tampoco se incluye el buffer de datos (320 MB de texto público, regenerable con `run_cycle.py`).

## Documentación

- [Arquitectura](docs/ARQUITECTURA.md) — trunk, hojas, router, fases del proyecto
- [Formato UKJ](docs/UKJ.md) — binario, tabla de rutas, checksum, mmap
- [Datos](docs/DATOS.md) — datasets, filtros, buffer, contrato de cursores
- [Router](docs/ROUTER.md) — arquitectura, entrenamiento, calibración de τ
- [TOJI](docs/TOJI.md) — memoria viva por sectores
- [Estado y medidas](docs/ESTADO.md) — resultados completos de cada fase
- [**Manual de entrenamiento**](docs/ENTRENAMIENTO.md) — datos, entrenamiento, export UKJ e inferencia
- [**Guía para agentes de IA**](docs/AGENTES.md) — playbooks, reglas y puertas de verificación

## Licencia

MIT. Ver [LICENSE](LICENSE).
