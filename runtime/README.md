# CABRITA

Runtime propio de **Iris** en Rust. Abre un modelo **UKJ** con mmap parcial (solo la
ruta activa entra en RAM), tokeniza con el BPE propio **IRISTOK**, elige ruta,
actualiza la memoria **TOJI** del sector y deja el feedback del usuario en una cola
JSONL que consume el entrenador. No entrena nada dentro del REPL y **nunca reescribe
el UKJ**.

```
tú> quiero programar una API en Rust
iris> (modo plantilla+stats: sin pesos útiles todavía) Mensaje: «quiero programar una API en Rust».
Ruta activa: informatica › programacion (id 1).
Modelo: UKJ v1 (p4-one) · 11 tok in · 256 B de tensores.
Tensores: dim=64 · media=32.5000 · energía=89440.00 · huella=0000001072400000.
TOJI[informatica]: 5 neuronas, 6 cables, activas: quiero, programar, una, api
Neurona más activa: #1 quiero (fuerza 0.40, 1 satélites).
Estrés: 11/12 activaciones.
  ⤷ ruta: informatica › programacion · 3.90 · 11 tok in / 256 tok out · stats · 4 ms · ruta 1 · dim 64 · media 32.500 · 256 B en RAM
```

## Estado

| Pieza | Estado |
|---|---|
| Lector UKJ con mmap parcial + checksum | Funcional, verificado contra `iris/julia/UKJ.jl` |
| Tokenizer IRISTOK (`iris.tok`) | Funcional, `encode`/`decode` byte a byte |
| Router | Provisional por contenido; interfaz estable para el clasificador entrenado |
| Memoria TOJI (`.mem` por sector) | Crear, cargar, guardar, fusionar, zstd |
| Estrés por ruta | Contadores + umbral por ruta persistidos en sidecar |
| Cola de feedback JSONL | `/si`, `/no`, `/up`, `/ent-my` con esquema v1 |
| Motor de respuesta | **Placeholder honesto** (`template`, `stats`); el forward pass real no existe todavía |

## Compilar y ejecutar

Requisitos: Rust 1.75 o superior. Sin GPU, sin servicios externos.

```bash
cargo build --release
cargo run --release                       # REPL con assets/test.ukj
cargo run --release -- --ask "¿qué es una integral?" --engine stats
```

Dependencias: `memmap2`, `zstd`, `serde`, `serde_json`, `anyhow` y, solo en Windows,
`windows-sys` para leer el RSS del proceso.

Por defecto busca `assets/test.ukj` y `assets/iris.tok` en el directorio actual y,
si no están, junto al ejecutable. También acepta variables de entorno:

| Variable | Campo |
|---|---|
| `CABRITA_MODEL` | ruta del `.ukj` |
| `CABRITA_TOK` | ruta del `iris.tok` |
| `CABRITA_MEM_DIR` | directorio de sectores TOJI |
| `CABRITA_QUEUE` | archivo JSONL de feedback |
| `CABRITA_ENGINE` | `template` o `stats` |

## Estructura

```
cabrita/
├── Cargo.toml
├── assets/
│   ├── test.ukj          # fixture de 3 rutas (dim 64) escrito por UKJ.jl
│   └── iris.tok          # tokenizer IRISTOK: 256 bytes + 800 fusiones
├── src/
│   ├── main.rs           # arranque, --help/--version, modo lectura
│   ├── cli.rs            # parser manual de argumentos (sin clap)
│   ├── util.rs           # normalización, sectores, ISO-8601, RSS
│   ├── ukj/
│   │   ├── mod.rs        # mmap, cabecera, tabla, checksum, tensores
│   │   └── stress.rs     # contadores y umbrales por ruta (sidecar JSON)
│   ├── tok/mod.rs        # IRISTOK: merges secuenciales, StreamDecoder
│   ├── router/
│   │   ├── mod.rs        # trait Router, RouteRequest, RouteChoice
│   │   └── keyword.rs    # router provisional por contenido (v0)
│   ├── toji/
│   │   ├── mod.rs        # formato .mem, MemStore, Sector::merge
│   │   └── memory.rs     # activación de neuronas/satélites por turno
│   ├── feedback/mod.rs   # registros JSONL, /ent-my (txt y json/jsonl)
│   ├── engine/mod.rs     # trait Engine + motores template y stats
│   └── ui/mod.rs         # REPL, comandos, streaming token a token
└── tests/cli.rs          # integración del binario (5 pruebas)
```

## Formatos en disco

### UKJ v0.1 (lo que realmente escribe `UKJ.jl`)

Little-endian. Cabecera de 23 bytes + tabla + tensores + footer de 16.

```
[0..4)    "UKJ\0"
[4..8)    u32 version = 1
[8..9)    u8  qlen
[9..9+q)  utf8 cuantización ("p4-one")
[..)      u32 n_rutas
[..)      u32 flags            (bit0 = tokenizer embebido; v0.1 = 0)
tabla:    n_rutas × { u32 id, u16 n_niveles, n_niveles × (u16 len + utf8),
                      u64 offset, u64 nbytes, u32 dim }
datos:    Float32 LE por ruta, sin relleno de alineación
footer:   u64 tabla_offset, u64 checksum
```

Decisiones tomadas frente al borrador de `UKJ_SPEC.md`, porque el escritor de Julia
es la fuente de verdad:

- **Sin relleno de 64 bytes** entre tensores.
- **Checksum aditivo** de todos los bytes anteriores al propio checksum (mód 2^64),
  no CRC64 como decía el borrador.
- **`dim` es un único `u32`** (número de elementos) y se exige `nbytes == dim * 4`.
- El footer guarda primero `tabla_offset` y después el checksum.

El archivo se mapea completo en memoria virtual, pero solo se leen la cabecera, la
tabla y la ruta pedida; `/info` informa del RSS real del proceso y de los bytes de la
ruta activa.

### IRISTOK (`iris.tok`)

```
"IRISTOK\0"  (8 bytes)
u32 version
u32 n_merges
n_merges × (u32 a, u32 b)
```

El vocabulario son los 256 bytes base más un token por fusión (1056 tokens con 800
fusiones). `encode` aplica las fusiones **en el orden de entrenamiento**, igual que
`BPE.jl`; `decode` reconstruye los bytes exactos y `StreamDecoder` retiene el sufijo
UTF-8 incompleto cuando un token corta un carácter por la mitad.

### TOJI `.mem`

Un archivo por sector (`<sector>.mem`), binario little-endian:

```
"TOJI\0\0\0\0"  (8 bytes)
u32 version = 1
utf8 sector
u32 n_neuronas, n_anormales, n_cables, n_fronteras
neuronas:  { u32 id, utf8 id_str, u32 n_satelites,
             satelites { u64 t_epoch, u8 tipo (0 evento | 1 emoción), f32 fuerza, utf8 texto } }
anormales: { u32 id, utf8 concepto, u32 n_hijos, u32 hijos[], utf8 matiz }
cables:    { u32 a, u32 b, u8 tipo, f32 peso }
fronteras: { utf8 sector_a, utf8 sector_b, f32 peso }
u64 checksum aditivo
```

Las cadenas usan longitud `u16`. Al leer se detecta automáticamente la cabecera zstd
(`28 B5 2F FD`), así que `--compress-mem` produce archivos que el mismo lector abre
sin configuración. `Sector::merge` fusiona por `id_str` de neurona, remapea los hijos
de las anormales y une cables y fronteras por par, quedándose con el peso máximo.

Decisiones: el spec no fija el tipo de `tipo` ni el checksum; se usan `u8` y suma
aditiva para seguir la convención del UKJ v0.1. `n_fronteras` se añade explícito
(el borrador solo lo insinuaba).

### Sidecar de estrés

El UKJ es de solo lectura, así que contadores y umbrales viven en
`<modelo>.stress.json`:

```json
{
  "schema": 1,
  "threshold": 12,
  "decay": 0.9,
  "routes": { "1": { "activations": 3, "stressed_cycles": 0 } },
  "overrides": { "2": 5 }
}
```

- Cada turno suma una activación por token de entrada y luego decae (`--stress-decay`).
- `--stress-threshold` fija el umbral global; `/stress <ruta> <n>` fija uno por ruta.
- Al superar el umbral la ruta se marca `ESTRESADA` y el motor lo recibe en
  `EngineContext { stressed, stress_count, stress_threshold }`.
- La liberación real de sub-parámetros del `p4-one` **todavía no se ejecuta**: el
  contador y la señal están cableados, pero el UKJ actual no guarda sub-parámetros
  comprimidos. Es el punto de enganche para F3.

## Comandos del REPL

| Comando | Efecto |
|---|---|
| `/si [texto]` | Marca la última respuesta como correcta (texto opcional = la correcta) |
| `/no` | Marca la última respuesta como equivocada |
| `/up` | La ruta era correcta y la respuesta no (refuerza router, penaliza pesos) |
| `/ent-my <archivo>` | Encola pares Q:/R: de un `.txt` o `.json`/`.jsonl` |
| `/info` | Ruta activa, decisión del router, RAM, integridad, cola, sectores |
| `/routes` | Tabla de rutas con dim, bytes, estrés y última decisión |
| `/stress` | Contadores de estrés por ruta |
| `/stress <ruta> <n>` | Fija el umbral de esa ruta (se persiste en el sidecar) |
| `/mem` | Sectores TOJI en disco con sus recuentos |
| `/merge <a> <b>` | Fusiona dos sectores en el de menor nombre y lo guarda |
| `/tok <texto>` | Tokens IRISTOK del texto y comprobación de ida y vuelta |
| `/queue [n]` | Últimos `n` registros de la cola de feedback |
| `/help`, `/exit` | Ayuda y salida (también `/salir`, `/quit`, Ctrl+D) |

`/ent-my` acepta tres formas:

1. Bloques `Q:` / `R:` (también `P:`/`PREGUNTA:` y `A:`/`RESPUESTA:`) separados por
   línea en blanco.
2. `.json` con una lista de objetos `{ "question"|"prompt"|"input", "answer"|"response"|"output" }`.
3. `.jsonl` con un objeto por línea.

## Cola de feedback

Un registro JSON por línea en `feedback_queue.jsonl`, esquema `1`:

```json
{
  "schema": 1,
  "kind": "si",
  "ts_unix": 1763330000,
  "ts_iso": "2026-11-17T10:13:20Z",
  "route_id": 1,
  "route_path": "informatica › programacion",
  "user_text": "quiero programar una API en Rust",
  "iris_text": "…",
  "correction": "la respuesta correcta era…",
  "tokens_in": 9,
  "tokens_out": 61,
  "engine": "stats",
  "model_version": "UKJ v1 (p4-one)",
  "source_file": null,
  "extra": null
}
```

`kind` es uno de `si`, `no`, `up`, `ent_my_pair`, `ent_my_meta`. `/ent-my` escribe un
registro por par y un `ent_my_meta` con `pairs` y `bytes` del archivo. Cabrita solo
**encola**: el entrenador lee el JSONL y decide. Nada de esto modifica el UKJ ni los
`.mem`.

## Router

`trait Router` recibe `RouteRequest { text, tokens, history, routes }` y devuelve
`RouteChoice { route_id, label, score, reason, candidates, used_fallback }`. El
clasificador entrenado se enchufa implementando el mismo trait; el REPL, el motor,
TOJI y el feedback no cambian.

La implementación actual (`keyword-v0-provisional`) puntúa **contenido**: términos de
los niveles de cada ruta más una tabla de alias (p. ej. `api`/`rust`/`código` →
`informatica`). Reglas explícitas:

- **Nunca por sonido**: la tabla de alias no contiene palabras sueltas ambiguas y
  `estado` está vetado como señal de `estados_unidos`. La frase
  «¿en qué estado está el mundo?» cae en `mundo`/`general`, nunca en
  `estados_unidos`, y la razón lo dice (`desambiguación`). Hay prueba unitaria.
- **Inercia de seguimiento**: frases cortas de continuación («y por qué», «otro
  ejemplo», ≤4 tokens) mantienen la ruta del turno anterior.
- **Fallback**: si nada supera `--min-score`, elige `general`/`mundo`/`sft` o la ruta
  de menor id, y lo marca con `used_fallback`.
- Umbral configurable con `--min-score` (por defecto `0.30`).

La decisión y sus candidatas se muestran en `/info`, `/routes` y en la línea de
metadatos de cada turno.

## Tests

```bash
cargo test
```

- **47 pruebas unitarias**: cabecera/tabla/checksum del UKJ, truncados y corrupción,
  roundtrip del tokenizer, rutas y trampa de `estado`, formato `.mem` (con y sin
  zstd), fusión de sectores, parser `/ent-my`, cola JSONL y sidecar de estrés.
- **5 pruebas de integración** (`tests/cli.rs`): pregunta única por CLI, sesión de
  REPL con todos los comandos, modo `--no-mem-write`, error claro con modelo
  ausente y `--help`. Cada una trabaja sobre una copia temporal de los assets y
  comprueba que el `test.ukj` queda byte a byte igual.

Resultado actual: `47 passed; 0 failed` + `5 passed; 0 failed`, sin warnings de
compilación.

## Cómo conectar el modelo entrenado

1. Copia el `.ukj` nuevo y su `iris.tok` al lado de cabrita (o pásalos con
   `--model`/`--tok`).
2. Arranca: `cargo run --release -- --model iris.ukj --tok iris.tok`.
3. El UKJ debe cumplir el formato de arriba y el `dim` de cada ruta debe ser
   `nbytes / 4`. La tabla puede tener cualquier número de rutas y niveles; no hay
   límite codificado.
4. El motor real se implementa sobre `EngineContext::tensors` en
   `src/engine/mod.rs` y se registra en `engine::from_name`. Todo lo demás
   (router, TOJI, estrés, feedback, streaming) ya está conectado.

## Decisiones y ambigüedades documentadas

- **Alineación y checksum UKJ**: se sigue al escritor Julia, no al borrador del spec
  (sin relleno, suma aditiva, `dim` único).
- **Estrés fuera del UKJ**: el modelo es inmutable; los contadores van en un sidecar
  y las pruebas verifican que el `.ukj` no cambia.
- **Sub-parámetros `p4-one`**: el runtime expone la señal de estrés, pero no
  descomprime nada porque el contenedor actual no los guarda. Se documenta como
  pendiente de F3 en lugar de simularlo.
- **`/merge` sin pérdida**: fusiona por identidad (`id_str` de neurona, par de
  extremos en cables/fronteras) y conserva el peso máximo; los ids se remapean.
- **Fallback del router**: `general`/`mundo`/`sft` antes que la primera ruta, para
  que el modelo no responda con una ruta temática arbitraria.
- **Modo lectura**: `--no-mem-write` impide escribir `.mem` y el sidecar, útil para
  inspeccionar un modelo sin dejar rastro.

## Limitaciones conocidas

- El motor es una plantilla: no inventa conocimiento, avisa de que no hay pesos
  útiles y muestra ruta, memoria y tensores para verificar la carga.
- El router es provisional; su calidad depende de los nombres de nivel del UKJ.
- El TOJI implementado es un subconjunto fiel del spec: neuronas, satélites,
  anormales, cables y fronteras, pero la creación de memoria usa reglas simples
  (palabras nuevas y emociones) hasta que exista el pensamiento profundo.
- Solo se mide RSS en Windows; en otros sistemas `/info` muestra `n/d`.
