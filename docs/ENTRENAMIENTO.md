# Manual completo de entrenamiento

Este documento es el procedimiento completo para reproducir Iris: descargar datos,
tokenizarlos, entrenar el transformer, guardar checkpoints, exportar el modelo a UKJ
y generar texto con él. Está escrito para que puedas seguirlo de principio a fin sin
leer el código, y para que un agente de IA pueda ejecutarlo por ti (la versión para
agentes está en [AGENTES.md](AGENTES.md)).

> **Antes de empezar, lee esto.** Iris v1 no va a hablar con los recursos de una CPU
> doméstica. El checkpoint incluido vio ~1,30 M de tokens y su loss es 5,77; genera
> `e e e e`. Este manual sirve para reproducir ese estado, verificar que toda la
> tubería funciona y, si tienes GPU o quieres un modelo más pequeño, entrenar algo
> que sí escriba frases. Los números reales están en [ESTADO.md](ESTADO.md).

---

## 0. Mapa del proceso

```
datos públicos (Parquet en HF)
        │  python src/python/run_cycle.py
        ▼
datos/buffer/<dataset>/*.jsonl.gz          (texto filtrado, chunks de 20k filas)
        │  julia src/julia/f4b_cycle.jl
        ▼
models/model.irmdl                          (checkpoint fp32, ~1,63 GB)
        │  julia src/julia/export_ukj.jl
        ▼
models/model.ukj + model.ukj.json           (contenedor + manifiesto)
        │  runtime/ (cabrita, Rust)
        ▼
texto generado
```

Cada flecha tiene su propio contrato. Si una falla, la siguiente no arranca: el
entrenador avisa si el buffer está vacío, el exportador avisa si el checkpoint no
existe y el runtime avisa si falta el manifiesto.

---

## 1. Requisitos

| Pieza | Versión probada | Para qué |
|---|---|---|
| Julia | 1.13.3 (1.10+) | modelo, entrenamiento, tokenizador, router, export UKJ |
| Python | 3.10+ | pipeline de datos (solo descarga y filtra) |
| Rust | 1.75+ | runtime cabrita y lector UKJ |
| GCC / Clang / MSVC | C11 | núcleo de tensores en C (opcional para entrenar) |
| RAM | 8 GB libres | 7,3 GB RSS medidos entrenando B=2 |
| Disco | ~4 GB | checkpoint 1,63 GB + UKJ 1,63 GB + buffer |
| GPU | no usada | todo el código es CPU; una GPU exigiría reescribir el backward |

Instalación:

```powershell
# Julia: dependencia del pipeline de datos
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Python
pip install -r requirements.txt

# Rust (solo si vas a usar cabrita)
cd runtime
cargo build --release
```

Comprobación rápida de que el entorno sirve:

```powershell
julia --version
python --version
cargo --version
julia src/julia/model_test.jl      # ~2 min; debe terminar con TODO OK
```

Si `model_test.jl` falla, no sigas: el problema está en la instalación de Julia o
en una modificación del código, no en los datos.

---

## 2. Los datos

### 2.1 Fuentes y mezcla

El modelo aprende **next-token prediction** sobre texto. No hay diálogos etiquetados,
ni instrucciones, ni destilación de otro modelo. Las cinco fuentes y sus pesos son:

| Nombre interno | Dataset de Hugging Face | Config | Peso |
|---|---|---|---|
| `c4_es` | `allenai/c4` | `es` | 30 % |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | 25 % |
| `wikipedia_es` | `wikimedia/wikipedia` | `20231101.es` | 15 % |
| `wikipedia_en` | `wikimedia/wikipedia` | `20231101.en` | 15 % |
| `oasst2` | `OpenAssistant/oasst2` | `default` | 15 % |

Los pesos están duplicados en dos sitios que deben coincidir:

- `src/python/run_cycle.py` → reparte el **tiempo de descarga**.
- `src/julia/f4b_cycle.jl` → reparte los **pasos de entrenamiento**.

Si cambias uno, cambia el otro. Un dataset que no esté en el diccionario `PESOS`
del entrenador se ignora aunque tenga chunks en el buffer.

### 2.2 Cómo se leen sin descargar el dataset entero

`src/python/iris_stream.py` pide el listado de archivos Parquet a Hugging Face, lee
el footer de cada archivo por **HTTP Range** (~0,4 MB), localiza los row-groups y
descarga solo los que va a consumir. Los cinco datasets usan row-groups de 1000
filas, así que el coste por lectura es pequeño y constante.

Filtros aplicados antes de escribir:

| Dataset | Filtro |
|---|---|
| Wikipedia ES/EN | 200–200 000 caracteres; descarta tablas y pies (`^\s*\|...\|`, `figura:`, `imagen:`, `archivo:`) |
| FineWeb-Edu | `language == "en"`, `language_score ≥ 0,85`, `int_score ≥ 3`, longitud 200–200 000 |
| C4 ES | longitud 200–200 000; mismo filtro de ruido web |
| oasst2 | no borrado, `review_result` distinto de `False`, idioma es/en, ≥ 20 caracteres |

### 2.3 Generar el buffer

```powershell
$env:IRIS_BUFFER="$PWD\datos\buffer"
$env:IRIS_CURSORS="$PWD\datos\cursors"
python src/python/run_cycle.py 7200
```

- El argumento es **segundos**, no minutos. `7200` = 2 h.
- Deja chunks de 20 000 filas en `datos/buffer/<dataset>/<dataset>_NNNNNN.jsonl.gz`
  (~30–40 MB de texto por chunk, gzip determinista).
- Tope de **0,5 GB por dataset**; al llegar, la ingesta de ese dataset se pausa
  hasta que el entrenador borre chunks.
- Cursores en `datos/cursors/*.json`; solo avanzan después de persistir las filas,
  así que un corte no duplica ni pierde datos.
- Rendimiento medido: 0,5–0,7 MB de texto útil por segundo. En 2 h deja ~320 MB.

`run_cycle.py` también escribe `src/python/estado_ciclo.json` con el reparto real
por fuente y los MB en buffer.

### 2.4 Usar datos propios

El contrato del buffer es deliberadamente simple:

1. Un archivo `.jsonl.gz` por chunk, con una línea JSON por documento y el texto en
   el campo `"text"`.
2. El nombre debe ser `<dataset>_<6 dígitos>.jsonl.gz` dentro de
   `datos/buffer/<dataset>/`.
3. `<dataset>` debe ser uno de los cinco nombres de la tabla (o añadirlo a `PESOS`
   en los dos archivos citados).

Ejemplo mínimo para probar la tubería con tus propios textos:

```python
import gzip, json
from pathlib import Path

destino = Path("datos/buffer/c4_es")
destino.mkdir(parents=True, exist_ok=True)
with gzip.open(destino / "c4_es_000000.jsonl.gz", "wt", encoding="utf-8") as f:
    for linea in Path("mis_textos.txt").read_text(encoding="utf-8").splitlines():
        if linea.strip():
            f.write(json.dumps({"text": linea}, ensure_ascii=False) + "\n")
```

El entrenador borra cada chunk cuando lo termina. Si quieres conservar tus datos,
trabaja sobre una copia.

---

## 3. El tokenizador

### 3.1 Qué es

BPE por bytes, sin dependencias: 256 tokens base (un byte cada uno) + 800 fusiones
= **1056 tokens**. Se entrenó con `BPE.train(texts, 800)` y se guardó en
`assets/iris.tok`.

Formato `IRISTOK`:

```
"IRISTOK\0" (8 bytes)
u32 version
u32 n_merges
n_merges × (u32 a, u32 b)
```

`encode` devuelve ids **0-based** (`0..1055`). El modelo indexa los embeddings con
`id + 1`, así que dentro del modelo los ids van de 1 a 1056. `decode` reconstruye
los bytes exactos; `StreamDecoder` (Rust) retiene sufijos UTF-8 incompletos cuando
un token corta un carácter por la mitad.

### 3.2 Reentrenarlo

```julia
include("src/julia/BPE.jl"); using .BPE
textos = readlines("mis_textos.txt")
t = BPE.train(textos, 800)
BPE.save("assets/mi_iris.tok", t)
```

Si cambias el número de fusiones cambia el vocabulario útil. El modelo reserva
`vocab=16384`, pero solo los primeros 1056 ids son decodificables; el runtime
enmascara el resto antes de muestrear. Puedes entrenar un modelo nuevo con
`vocab=1056` y ahorrar ~15,7 M de parámetros (ver §10).

**Un tokenizador nuevo invalida el checkpoint viejo.** No se puede cambiar el BPE
y seguir entrenando el mismo `.irmdl`: los ids ya no significan lo mismo.

---

## 4. El modelo

### 4.1 Configuración por defecto

`ModelConfig` en `src/julia/model.jl`:

| Campo | Valor | Notas |
|---|---|---|
| `dim` | 1024 | ancho del residual |
| `n_layers` | 14 | capas del trunk |
| `n_heads` | 16 | 64 dimensiones por cabeza |
| `ffn` | 4096 | capa intermedia (4×dim) |
| `vocab` | 16384 | solo 1056 se usan |
| `ctx` | 2048 | longitud máxima |
| `n_hojas` | 16 | bloques expertos, hoy no usados en el forward |
| `seed` | `0x49524953` | "IRIS" |

### 4.2 Layout del checkpoint

`layout(cfg)` asigna un vector plano `p::Vector{Float32}` con este orden:

```
emb                    dim × vocab
t1..t14                por capa: ln1g, ln1b, wqkv, wo, ln2g, ln2b, w1, w2
e1..e16                mismo bloque que una capa (hojas)
sft                    mismo bloque que una capa
fin.ln_g, fin.ln_b     dim cada uno
```

Por capa: `2·dim + dim·3·dim + dim·dim + 2·dim + dim·ffn + ffn·dim` =
`2·1024 + 3,145,728 + 1,048,576 + 2·1024 + 4,194,304 + 4,194,304` =
**12 587 008** parámetros.

| Parte | Parámetros |
|---|---|
| Trunk (emb + 14 capas + LN final) | **192 997 376** |
| 16 hojas + `sft` (17 bloques) | 213 979 136 |
| **Total del checkpoint** | **406 976 512** |

Solo el trunk entra en el forward y recibe gradiente real. Las hojas y `sft` se
reservan para las fases siguientes; hoy son peso muerto (53 % del archivo). El
manual los documenta en vez de ocultarlo porque afectan al tamaño y a la velocidad
de guardado.

### 4.3 Formato IRISMDL

```
"IRISMDL\0" (8 bytes)
u32 version = 1
u32 dim, n_layers, n_heads, ffn, vocab, ctx, n_hojas
u32 seed
i64 n_params
f32[n_params] little-endian
```

No guarda nombres ni offsets: se reconstruyen con `layout(cfg)`. Un checkpoint de
la configuración por defecto pesa 1 627 906 100 bytes (1,63 GB).

---

## 5. Entrenar

### 5.1 Comando

```powershell
$env:IRIS_BUFFER="$PWD\datos\buffer"
$env:IRIS_TOK="$PWD\assets\iris.tok"
New-Item -ItemType Directory -Force models | Out-Null

# 120 minutos; checkpoint cada 15 min, cursor cada 5
julia --project=. src/julia/f4b_cycle.jl 120 models\model.irmdl src\julia\f4b_cursor.json
```

Argumentos: `minutos`, `checkpoint`, `cursor`. Si el checkpoint no existe, se
inicializa uno nuevo con `ModelConfig()`; si existe, se reanuda desde él.

Para un ciclo corto de prueba (2 minutos):

```powershell
julia --project=. src/julia/f4b_cycle.jl 2 models\prueba.irmdl src\julia\prueba_cursor.json
```

### 5.2 Qué hace cada paso

1. Elige el dataset con menor cociente `tokens_servidos / peso` entre los que ya
   tienen al menos 512 tokens en memoria.
2. Si ninguno llega, pide 32 líneas al dataset más atrasado, las tokeniza y las
   añade a su buffer. Al agotar un chunk lo **borra** y avanza al siguiente.
3. Empaqueta dos secuencias de `T=256` (`B=2`) y ejecuta forward + backward.
4. Acumula gradientes y aplica AdamW **solo a los parámetros de la fase 1**
   (trunk + hoja 1 + `sft`; en la práctica solo el trunk recibe gradiente).
5. Cada 15 min guarda el checkpoint; cada 5 min guarda el cursor.

### 5.3 Hiperparámetros exactos

| Parámetro | Valor | Dónde |
|---|---|---|
| Secuencia `T` | 256 | `f4b_cycle.jl` |
| Micro-batch `B` | 2 | `f4b_cycle.jl` |
| Tokens por paso | 512 | `T × B` |
| Optimizador | AdamW | `model.jl` |
| Learning rate | 1e-3 | `f4b_cycle.jl` |
| `beta1` / `beta2` | 0,9 / 0,95 | `AdamOpt` |
| `eps` | 1e-8 | `AdamOpt` |
| Weight decay | 0,1 solo en matrices `*.w*` | `AdamOpt` |
| Grad clip | **ninguno** | — |
| Scheduler / warmup | **ninguno** | — |
| Precisión | fp32 | — |
| Hilos | los de Julia (`JULIA_NUM_THREADS`) | — |

No hay acumulación de gradientes entre pasos ni recorte por norma. Si entrenas un
modelo más grande o con lr más alto, añade clipping antes de subir el lr.

### 5.4 Reanudar y encadenar ciclos

El estado vive en dos archivos:

- **Checkpoint** (`.irmdl`): pesos. Se escribe cada 15 min y al terminar.
- **Cursor** (`f4b_cursor.json`): por dataset, `chunk`, `line` (líneas crudas) y
  `tokens` históricos. Se escribe cada 5 min y al terminar.

Relanzar el mismo comando continúa donde se quedó. El cursor puede ir hasta 5 min
por delante del checkpoint: tras un corte se pierden esos tokens (se leyeron pero
no se entrenaron), sin corromper nada.

Para encadenar ciclos sin intervención:

```powershell
# Windows: una tarea cada 2 h
schtasks /create /tn IrisF4bCycle /tr "julia --project=. src/julia/f4b_cycle.jl 120 models\model.irmdl src\julia\f4b_cursor.json" /sc hourly /mo 2
```

```bash
# Linux/macOS: cron cada 2 h
0 */2 * * * cd /ruta/iris && julia --project=. src/julia/f4b_cycle.jl 120 models/model.irmdl src/julia/f4b_cursor.json >> logs/cycle.log 2>&1
```

Antes de lanzar un ciclo largo, comprueba que no hay otro en marcha (dos procesos
escribiendo el mismo checkpoint se pisan).

### 5.5 Cuánto tarda y qué esperar

| Medida | Valor medido (Ryzen 7 5825U) |
|---|---|
| Velocidad | 108–129 tok/s (B=2) |
| Texto digerido | 0,87 MB/h |
| Ciclo de 2 h | ~781 000 tokens |
| Checkpoint | 1,63 GB cada 15 min |
| RSS | 7,3 GB |

La loss inicial de un modelo aleatorio es ≈ `ln(16384) = 9,70` (baseline uniforme).
Tras 1,30 M de tokens la media acumulada quedó en **5,77** (perplejidad ≈ 320).
Con esa cantidad de datos el modelo aprende frecuencias de palabras y algunos
bigramas; no forma frases.

**Cómo leer la loss del ciclo:** `f4b_cycle.jl` imprime dos valores: `loss acum`
(media desde el inicio del proceso) y `loss ventana` (media de los últimos 40 s).
La media acumulada se aplana por construcción; para diagnosticar usa la de
ventana. Los logs históricos solo tienen la acumulada, así que la meseta de 5,77
no demuestra que el modelo dejara de mejorar. Para una medida independiente,
evalúa el checkpoint con `model_test.jl` o `generate.jl`.

---

## 6. Verificar que el entrenamiento es sano

```powershell
# gradiente analítico vs numérico, causalidad, save/load, Adam
julia src/julia/model_test.jl

# estado del buffer y los cursores
julia src/julia/check_cycle.jl
```

`model_test.jl` comprueba, por grupos de parámetros, que el gradiente analítico
coincide con diferencias centrales (error relativo ≤ 1e-3), que el modelo es causal
(dos secuencias que difieren solo en el futuro dan la misma loss en el pasado) y que
`save_model`/`load_model` es bit-exacto. Si tocas `model.jl`, ejecútalo antes de
gastar horas de CPU.

Señales de que algo va mal:

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| Loss = `NaN` o `Inf` | lr demasiado alto, datos corruptos | baja lr a 3e-4, revisa el chunk |
| Loss no baja de ~9,7 | no aprende; gradientes muertos | comprueba `model_test.jl`, revisa el tokenizador |
| Loss baja y sube en dientes de sierra | lr alta sin clipping | añade recorte de norma o baja lr |
| `buffer agotado: N tokens < 512` | no hay datos suficientes | genera más buffer o sube el ciclo |
| El checkpoint no se escribe | disco lleno o ruta sin permisos | revisa espacio y ruta |

---

## 7. Exportar a UKJ

```powershell
julia src/julia/export_ukj.jl models\model.irmdl models\model.ukj p4-one
```

Salida:

- `models/model.ukj` — contenedor UKJ v0.1.
- `models/model.ukj.json` — manifiesto con la config y, por ruta, el offset y el
  tamaño de cada tensor.

Rutas que escribe el exportador:

| id | nivel | Contenido |
|---|---|---|
| 1 | `trunk` | `emb` + `fin.ln_g/b` + `t1..t14` |
| 2–17 | `hoja` | `e1..e16` |
| 18 | `sft` | bloque `sft` |

El script **relee el UKJ y compara los bytes de cada ruta con el checkpoint**. Si
la verificación falla, no uses el archivo. UKJ v0.1 no guarda nombres ni formas;
sin el manifiesto, el runtime no puede reconstruir el modelo.

En el modelo real el UKJ pesa 1 627,9 MB y el runtime solo carga la ruta `trunk`
(736,2 MB) en RAM.

---

## 8. Generar texto

### 8.1 Referencia en Julia (sin caché KV)

```powershell
$env:IRIS_CKPT="$PWD\models\model.irmdl"
julia src/julia/generate.jl --prompt "hola mundo" --tokens 20 --temp 0
```

- `--temp 0` elige siempre el token más probable (greedy). Es la forma de comparar
  implementaciones.
- `--temp 0.8 --top-k 40 --top-p 0.95 --seed 7` muestrea.
- Cada token nuevo repite el forward completo: es lento y solo sirve de referencia.
- Con el checkpoint real y `--temp 0`, la salida es `e e e e...`.

### 8.2 Runtime cabrita (Rust, con caché KV)

```powershell
cd runtime
cargo run --release -- --engine transformer `
  --model ..\models\model.ukj --tok ..\assets\iris.tok `
  --ask "hola mundo" --max-new 20 --temp 0 --seed 7
```

Sin `--ask` entra en el REPL interactivo. Flags del motor:

| Flag | Por defecto | Efecto |
|---|---|---|
| `--engine transformer` | `template` | activa el forward real |
| `--max-new` | 48 | tokens a generar |
| `--temp` | 0,9 | 0 = greedy |
| `--top-k` | 40 | 0 = sin recorte |
| `--top-p` | 0,95 | núcleo de probabilidad |
| `--seed` | reloj | semilla reproducible |

El motor enmascara los ids que el tokenizador no puede decodificar (el modelo
reserva 16384, el BPE real tiene 1056), así que nunca devuelve bytes inválidos.
Carga el trunk con mmap y no verifica el checksum completo al arrancar (leer 1,5 GB
en cada inicio no aporta nada); `UkjModel::verificar()` queda disponible para
herramientas.

Rendimiento medido con el checkpoint real: **~19 tok/s** en CPU (un matvec por
token sobre 736 MB; el entrenador es más rápido porque procesa 256 tokens por
pasada y amortiza la lectura de pesos).

### 8.3 Comparar Julia y Rust

Los dos motores deben dar los mismos logits. Hay una prueba golden en
`runtime/src/engine/transformer.rs` que compara los primeros 8 logits y el argmax
contra `model.jl` usando `runtime/assets/tiny.ukj`. Para el modelo real, genera con
`--temp 0` en ambos y compara el texto:

```powershell
julia src/julia/generate.jl --prompt "hola mundo" --tokens 12 --temp 0
cd runtime
cargo run --release -- --engine transformer --model ..\models\model.ukj --tok ..\assets\iris.tok --ask "hola mundo" --max-new 12 --temp 0
```

Ambos producen la misma secuencia de 12 tokens.

---

## 9. Reproducir el estado publicado

Con una CPU de 8 núcleos, el camino completo son ~7 h de reloj (2 h de ingesta +
3,5 h de entrenamiento + export + pruebas), pero solo obtendrás ~1,3 M de tokens, que
es el 0,03 % de lo que pide el modelo. Pasos:

1. `python src/python/run_cycle.py 7200`
2. `julia --project=. src/julia/f4b_cycle.jl 120 models\model.irmdl src\julia\f4b_cursor.json`
3. Repite el paso 2 tantas veces como quieras (cada ciclo añade ~780k tokens).
4. `julia src/julia/export_ukj.jl models\model.irmdl models\model.ukj`
5. `cd runtime && cargo run --release -- --engine transformer --model ..\models\model.ukj --tok ..\assets\iris.tok --ask "hola"`

No hace falta repetir la ingesta si el buffer aún tiene chunks; el entrenador los
consume y los borra. Cuando el buffer baje de 512 tokens por dataset, vuelve a
lanzar `run_cycle.py`.

---

## 10. Recetas para que llegue a hablar

La cuenta: 193 M de parámetros activos × ~20 tokens/parámetro ≈ **3 860 M de
tokens**. A 108 tok/s son ~414 días de CPU continua. Hay tres formas de cambiar
esa ecuación.

### Receta A — modelo pequeño (la más realista en CPU)

Un modelo de 20–50 M de parámetros aprende frases simples con ~1 000 M de tokens.
Con `dim=384, n_layers=6, n_heads=6, ffn=1536, vocab=1056, ctx=512` salen ~15 M
de parámetros y el entrenamiento es ~15× más rápido. Pasos:

```julia
include("src/julia/model.jl"); using .IrisModel
cfg = ModelConfig(dim=384, n_layers=6, n_heads=6, ffn=1536,
                  vocab=1056, ctx=512, n_hojas=1)
m = init_model(cfg)
save_model("models/small.irmdl", m)
```

Después entrena pasando ese checkpoint:

```powershell
julia --project=. src/julia/f4b_cycle.jl 120 models\small.irmdl src\julia\small_cursor.json
```

`n_hojas=1` es el mínimo que acepta `trainable_fase1(m, 1)`; la hoja sigue sin
usarse en el forward. Si reduces `vocab` a 1056, el tokenizador y el modelo quedan
alineados y desaparece el 94 % de la matriz de logits.

### Receta B — GPU

El código actual es CPU (Julia con `Threads.@threads` y GEMM de BLAS). Para GPU hay
que portar `forward_loss_batch`/`backward_batch!` a un framework con autograd o
escribir kernels propios. La arquitectura y el formato UKJ no cambian; el
exportador y el runtime siguen sirviendo. Sin ese port, una GPU no acelera nada.

### Receta C — reducir el modelo actual

Antes de entrenar más, arregla lo que sobra:

1. **`vocab` 16384 → 1056**: −15,7 M de parámetros y ~15× menos GEMM de logits.
2. **Quitar las 16 hojas y `sft`**: −214 M de parámetros y checkpoints de 1,63 GB →
   ~772 MB. El forward no los usa.
3. **Caché KV**: ya está en el runtime (generación O(T)); el entrenador no la
   necesita.

Cada cambio invalida el checkpoint actual: hay que reinicializar y entrenar de
nuevo. Son cambios de arquitectura, no parches.

### Hiperparámetros sugeridos para un modelo pequeño

| Parámetro | Valor inicial |
|---|---|
| lr | 3e-4 a 1e-3 |
| warmup | 1–2 % de los pasos |
| grad clip | norma 1,0 |
| `T` | 256–512 |
| `B` | el máximo que quepa en RAM |
| weight decay | 0,1 en matrices |
| tokens | ≥ 20 × parámetros activos |

---

## 11. Problemas frecuentes

| Error | Significado | Solución |
|---|---|---|
| `no existe el buffer` | `IRIS_BUFFER` mal o buffer vacío | define la variable y ejecuta `run_cycle.py` |
| `buffer sin chunks .jsonl.gz` | los chunks no tienen el nombre esperado | `<dataset>_<6 dígitos>.jsonl.gz` |
| `ningun dataset con peso del plan` | nombre de dataset desconocido | usa uno de los cinco o edita `PESOS` |
| `no existe el checkpoint` | ruta incorrecta en `generate.jl` | `IRIS_CKPT` o primer argumento de `export_ukj.jl` |
| `no existe el manifiesto` | falta `model.ukj.json` | vuelve a exportar con `export_ukj.jl` |
| `el manifiesto no declara el tensor ...` | UKJ y manifiesto de modelos distintos | exporta los dos juntos |
| `la ruta trunk id=... no está` | manifiesto de otro UKJ | borra el par y reexporta |
| `token ... fuera del vocabulario` | ids sin el `+1` o BPE distinto | usa `encode` y suma 1; no mezcles tokenizadores |
| `posición ... fuera de la caché KV` | `--max-new` + prompt > `ctx` | baja `--max-new` o acorta el prompt |
| El runtime genera `e e e` | el modelo está subentrenado | es el estado esperado con 1,3 M tokens |

---

## 12. Checklist final

- [ ] `julia src/julia/model_test.jl` termina en `TODO OK`.
- [ ] `make -C src/c test` pasa 19/19 (si compilas el núcleo C).
- [ ] `cd runtime && cargo test --release` pasa 50 unitarias + 7 de integración.
- [ ] El buffer tiene chunks y el entrenador avanza (`paso ... tok ... loss ...`).
- [ ] El checkpoint existe y pesa lo esperado (1,63 GB con la config por defecto).
- [ ] `export_ukj.jl` imprime `verificacion: OK`.
- [ ] `cabrita --engine transformer` genera texto y muestra `trunk 14 capas`.
- [ ] Julia y Rust dan la misma salida con `--temp 0`.
- [ ] Sabes qué límite te frena: tokens, RAM, CPU o arquitectura.

---

## 13. Glosario

- **Trunk**: las 14 capas transformer que sí ejecutan el forward.
- **Hoja**: bloque experto reservado para una ruta de conocimiento; hoy no se usa.
- **UKJ**: contenedor propio de solo lectura con tabla de rutas y mmap.
- **IRISMDL**: checkpoint de entrenamiento (config + vector plano de pesos).
- **IRISTOK**: tokenizador BPE por bytes (1056 tokens).
- **Manifiesto**: JSON que dice qué tensor vive en qué offset de la ruta UKJ.
- **Cursor**: posición de lectura por dataset, para reanudar sin duplicar.
- **Packing**: unir documentos en secuencias fijas de `T` tokens.
- **Loss**: entropía cruzada media en nats; `ln(16384) = 9,70` es el azar puro.
- **Perplejidad**: `exp(loss)`; 5,77 → ~320 tokens de incertidumbre por token.
- **Fase 1**: trunk + una hoja + `sft`; es lo que `trainable_fase1` marca.


---

## Apéndice A — entrenar el router

El router no participa en la generación de texto: elige a qué ruta mandar una
petición. Es un clasificador independiente (bag-of-embeddings + MLP, 110 358
parámetros, formato `IRISRTR`) y el artifact entrenado ya está en
`models/router_v0.irrtr`.

```powershell
# barrido de tau sobre el artifact incluido (no entrena, tarda segundos)
julia src/julia/router_tau_sweep.jl data\router_data_f4c.tsv models\router_v0.irrtr

# reentrenar desde el TSV incluido (10 semillas, split 70/15/15)
julia src/julia/router_test.jl data\router_data_f4c.tsv models\router_v0_retrained.irrtr
```

El TSV tiene 674 muestras y 22 rutas. El resultado publicado es 56,2 % de argmax
en test y, con `τ=0,85` grabado en el artifact, 28/28 en validación y 29/29 en test
con cobertura del 29 %. Los detalles están en [ROUTER.md](ROUTER.md).

`src/julia/router_extractor.jl` regenera el TSV desde `Shared/IRIS_DESTILADO`,
que **no se incluye en el repositorio**. Para usar tus propios datos, el contrato
es simple: un `.tsv` con `texto<TAB>ruta` por línea; las rutas deben ser las mismas
que espera el runtime o el artifact no sabrá etiquetarlas.