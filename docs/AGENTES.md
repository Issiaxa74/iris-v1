# Guía para agentes de IA

Este documento está escrito para que un agente (Claude Code, Codex, OpenCode,
Hermes, Cursor o cualquier otro con acceso a shell) pueda **instalar, verificar,
entrenar y configurar Iris** sin romper el proyecto ni inventar resultados.

La regla de oro: **no hay prisa y no se afirma nada sin una salida que lo demuestre.**
Iris es un proyecto de sistemas, no una demo; la mayoría de los errores se detectan
con una comprobación barata antes de gastar horas de CPU.

---

## 1. Qué hay que conseguir

Según lo que pida la persona, el trabajo es uno de estos:

| Petición | Resultado esperado | Tiempo |
|---|---|---|
| «Compruébame que funciona» | tests en verde y una generación de prueba | 5–10 min |
| «Entrénalo un rato» | buffer + ciclo corto + checkpoint | 20–60 min |
| «Haz que hable» | modelo más pequeño o GPU; ver §7 | horas/días |
| «Publica / actualiza el repo» | commits, push, verificación por API | 5 min |
| «Cambia la arquitectura» | plan + tests; no tocar el checkpoint real | variable |

Si la petición es ambigua, pregunta **una** cosa: ¿quieres reproducir el estado
publicado (1,3 M tokens, no habla) o entrenar algo que escriba frases? La respuesta
cambia por completo el trabajo.

---

## 2. Antes de tocar nada

### 2.1 Aislar el trabajo

Nunca trabajes directamente sobre un checkpoint que no hayas creado tú. Si el
proyecto ya tiene `model_f4b.irmdl` o un buffer con datos:

```bash
git status
git switch -c agente/<fecha>-<tarea>     # si es un repo git
```

Si no es un repo, copia el proyecto o al menos `models/` y `datos/` antes de
empezar. El entrenador **borra los chunks que consume**; el buffer no es sagrado.

### 2.2 Leer en este orden

1. `README.md` — qué es y cifras reales.
2. `docs/ESTADO.md` — qué funciona y qué no.
3. `docs/ENTRENAMIENTO.md` — el procedimiento completo.
4. `docs/DATOS.md` — contrato del buffer y cursores.
5. `docs/UKJ.md` — formato del modelo exportado.
6. `runtime/README.md` — runtime y flags de inferencia.

No hace falta leer todo el código para operar; sí para modificarlo.

### 2.3 Auditoría del entorno

```bash
julia --version
python --version
cargo --version
free -h          # Linux; en Windows: Get-CimInstance Win32_OperatingSystem
df -h            # espacio libre
```

Mínimos: Julia 1.10+, Python 3.10+, Rust 1.75+, 8 GB de RAM libres, 4 GB de disco.
Sin GPU no pasa nada: el proyecto es CPU por diseño.

---

## 3. Reglas duras

Estas reglas existen por errores reales. No las saltes.

1. **No borres ni sobrescribas el checkpoint original** (`model_f4b.irmdl` o
   cualquier `.irmdl` que no hayas creado). Si necesitas espacio, copia y trabaja
   sobre la copia.
2. **No lances dos ciclos de entrenamiento a la vez.** Dos procesos escribiendo el
   mismo checkpoint lo corrompen.
3. **No cambies el tokenizador y sigas entrenando el mismo checkpoint.** Un BPE
   nuevo invalida los ids; hay que reinicializar el modelo.
4. **No reduzcas `vocab` ni quites hojas «para probar» sobre un checkpoint ya
   entrenado.** Son cambios de arquitectura: checkpoint nuevo desde cero.
5. **No afirmes que el modelo «ya habla»** porque genera texto. Con 1,3 M tokens
   produce `e e e`; eso es el estado esperado, no un fallo.
6. **No inventes cifras.** Loss, tokens, tok/s y tamaños salen de la salida de los
   comandos o de `docs/ESTADO.md`.
7. **No mates procesos por nombre** (`node`, `python`, `julia`) para «liberar
   memoria». Si un ciclo se cuelga, identifica el PID concreto y pregúntalo.
8. **No subas checkpoints grandes al repo.** GitHub rechaza archivos >100 MB. Un
   release asset admite hasta 2 GB.
9. **No modifiques `model.jl` sin ejecutar `model_test.jl`.** Es la red de
   seguridad del gradiente manual.
10. **No toques el proyecto local del usuario** si trabajas sobre una copia: el
    objetivo es publicar, no reemplazar su trabajo.

---

## 4. Playbook: verificar que todo funciona

Puerta de entrada para cualquier tarea. No entrenes hasta que esto pase.

```bash
# 1. Núcleo C (opcional, pero es rápido)
make -C src/c test

# 2. Modelo: gradiente, causalidad, save/load
julia src/julia/model_test.jl

# 3. Runtime
cd runtime && cargo test --release
```

Salidas esperadas:

| Comando | Éxito |
|---|---|
| `make -C src/c test` | `19/19 PASS` |
| `julia src/julia/model_test.jl` | termina con `TODO OK` |
| `cargo test --release` | `50 passed` + `7 passed`, 0 fallos |

Después, una generación con el fixture diminuto (no necesita el checkpoint real):

```bash
cd runtime
cargo run --release -- --engine transformer \
  --model assets/tiny.ukj --tok assets/iris.tok \
  --ask "hola mundo" --max-new 8 --temp 0 --seed 7
```

Debe imprimir `iris> helhelaj...` y una línea de metadatos con `transformer`,
`trunk 2 capas`. Si falta `assets/tiny.ukj`, el test golden se omite: el fixture
está en el repo para que no se omita.

---

## 5. Playbook: entrenar un ciclo corto

### 5.1 Datos

```bash
export IRIS_BUFFER="$PWD/datos/buffer"
export IRIS_CURSORS="$PWD/datos/cursors"
python src/python/run_cycle.py 600      # 10 minutos de ingesta
```

En Windows PowerShell:

```powershell
$env:IRIS_BUFFER="$PWD\datos\buffer"
$env:IRIS_CURSORS="$PWD\datos\cursors"
python src/python/run_cycle.py 600
```

Comprueba que hay chunks:

```bash
find datos/buffer -name '*.jsonl.gz' | head
du -sh datos/buffer
```

### 5.2 Humo de 2 minutos

```bash
julia --project=. src/julia/f4b_cycle.jl 2 models/smoke.irmdl src/julia/smoke_cursor.json
```

Criterio de éxito: imprime pasos, `loss` finita (no `NaN`) y termina con
`OK f4b_cycle`. El checkpoint `smoke.irmdl` debe existir y pesar lo que corresponda
a la config (1,63 GB con la default; menos si es un modelo pequeño).

### 5.3 Ciclo real

Solo si el humo pasó y la persona lo autorizó:

```bash
julia --project=. src/julia/f4b_cycle.jl 120 models/model.irmdl src/julia/f4b_cursor.json
```

Antes de lanzarlo, confirma:

- [ ] Hay al menos 512 tokens por dataset con peso (o el entrenador se quedará sin
      datos y terminará).
- [ ] Hay ~2 GB libres para el checkpoint temporal.
- [ ] No hay otro ciclo en marcha.
- [ ] Sabes si el checkpoint existe y si se reanuda o se crea nuevo.

Durante el ciclo, no lances comandos que lean/escriban el mismo checkpoint. Puedes
observar el log o el uso de CPU.

---

## 6. Playbook: exportar y hablar con el modelo

```bash
julia src/julia/export_ukj.jl models/model.irmdl models/model.ukj p4-one
```

Criterio de éxito: `verificacion: OK (18 rutas, bytes identicos al checkpoint)`.
Se escriben `model.ukj` y `model.ukj.json`; los dos son necesarios.

```bash
cd runtime
cargo run --release -- --engine transformer \
  --model ../models/model.ukj --tok ../assets/iris.tok \
  --ask "hola mundo" --max-new 20 --temp 0 --seed 7
```

Criterio de éxito: texto + `transformer` + `trunk 14 capas` + `tok/s`. Con el
checkpoint real la salida es `e e e...`; con el fixture diminuto, `helhelaj...`.

Comparación con la referencia Julia:

```bash
IRIS_CKPT=models/model.irmdl julia src/julia/generate.jl --prompt "hola mundo" --tokens 12 --temp 0
```

Ambos deben generar la misma secuencia con `--temp 0`. Si no coincide, el problema
está en el manifiesto, en el orden de los tensores o en el `+1` de los ids; no en el
muestreo.

---

## 7. Playbook: hacer que escriba frases

No prometas que «va a hablar» entrenando el modelo actual en CPU. La cuenta es
3 860 M de tokens para 193 M de parámetros; a 108 tok/s son ~414 días. Presenta las
opciones con números y deja elegir:

| Opción | Qué implica | Coste realista |
|---|---|---|
| Modelo pequeño 15–50 M | `ModelConfig` nuevo, `vocab=1056`, `n_hojas=1`; entrenar desde cero | días de CPU o horas de GPU |
| GPU | portar forward/backward; UKJ y runtime no cambian | trabajo de ingeniería + hardware |
| Seguir con el actual | solo sirve para verificar la tubería | no llega a frases |

Receta de modelo pequeño (crear el checkpoint antes de entrenar):

```julia
include("src/julia/model.jl"); using .IrisModel
cfg = ModelConfig(dim=384, n_layers=6, n_heads=6, ffn=1536,
                  vocab=1056, ctx=512, n_hojas=1)
save_model("models/small.irmdl", init_model(cfg))
```

```bash
julia --project=. src/julia/f4b_cycle.jl 120 models/small.irmdl src/julia/small_cursor.json
```

`n_hojas=1` es el mínimo que acepta `trainable_fase1(m, 1)`. La hoja no entra en el
forward; es un requisito del layout actual. Si el objetivo es un modelo limpio, hay
que adaptar `trainable_fase1` para aceptar 0 hojas; no lo hagas sin tests.

Antes de entrenar horas, calcula y enseña el presupuesto:

```
tokens_necesarios ≈ 20 × parámetros_activos
horas = tokens_necesarios / (tok_s × 3600)
```

Si el resultado no cabe en el tiempo disponible, dilo y propone reducir el modelo.

---

## 8. Puertas de verificación

Ninguna fase se da por buena sin su puerta. Copia la salida real en el informe.

| Fase | Comando | Puerta |
|---|---|---|
| Entorno | `julia src/julia/model_test.jl` | `TODO OK` |
| C | `make -C src/c test` | `19/19 PASS` |
| Runtime | `cargo test --release` | 50 + 7, 0 fallos |
| Datos | `python src/python/run_cycle.py 600` | chunks en buffer, MB > 0 |
| Humo | `f4b_cycle.jl 2 ...` | loss finita, `OK f4b_cycle` |
| Export | `export_ukj.jl ...` | `verificacion: OK` |
| Inferencia | `cabrita --engine transformer` | texto + `trunk N capas` |
| Paridad | Julia vs Rust `--temp 0` | misma secuencia |

Si una puerta falla, **para**. No continúes a la siguiente fase «para ver si
funciona»: los errores se propagan y el diagnóstico se vuelve más caro.

---

## 9. Cómo modificar código sin romper el modelo

Antes de tocar `src/julia/model.jl`, `export_ukj.jl` o
`runtime/src/engine/transformer.rs`:

1. Guarda una copia del archivo y del checkpoint de prueba (`tiny.ukj` + `.json`).
2. Ejecuta `model_test.jl` y `cargo test --release` para tener una línea base.
3. Haz el cambio mínimo.
4. Vuelve a ejecutar las dos suites.
5. Compara logits contra el golden de `transformer.rs` (primeros 8 logits y
   argmax 1044 con `tiny.ukj`).
6. Si cambias el layout o el orden de tensores, actualiza **a la vez**
   `layout()` en Julia, `export_ukj.jl`, el manifiesto y `cargar_trunk()` en Rust.
   Un desajuste de offsets no da error hasta que el texto sale mal.

Prohibido en un cambio «pequeño»: cambiar `vocab`, quitar hojas, alterar el orden
de `wqkv`, cambiar la tabla senoidal o el `eps` del LayerNorm. Todo eso invalida
checkpoints y golden tests.

---

## 10. Plantilla de informe

Cuando termines, entrega esto (sin adornos):

```
Tarea: <una línea>
Entorno: <SO, CPU, RAM, Julia/Python/Rust>
Qué se ejecutó:
  - <comando> -> <resultado literal>
  - <comando> -> <resultado literal>
Estado: <verificado / bloqueado / parcial>
Artefactos: <rutas y tamaños>
Limitaciones: <lo que no se pudo probar y por qué>
Siguiente paso sugerido: <uno solo>
```

Nunca escribas «funciona» sin la línea de salida que lo respalde. Si algo quedó
sin probar, dilo.

---

## 11. Problemas y respuestas

| Síntoma | Diagnóstico | Acción |
|---|---|---|
| `model_test.jl` falla | código o Julia mal instalado | no entrenes; revisa el stacktrace |
| `buffer sin chunks` | ingesta no corrió o el nombre no cuadra | revisa `IRIS_BUFFER` y el patrón |
| `loss NaN` | lr alta o chunk corrupto | baja lr, prueba otro chunk |
| `cargo test` falla por `test.ukj` | falta el fixture | `git checkout -- runtime/assets/test.ukj` o regenéralo con `UKJ.jl` |
| `no existe el manifiesto` | exportaste solo el UKJ | vuelve a ejecutar `export_ukj.jl` |
| El runtime dice `template` | no pasaste `--engine transformer` | añade el flag |
| El texto es `e e e` | modelo subentrenado | esperado; no es un bug |
| El puerto/prompt no cabe | `--max-new` + prompt > `ctx` | baja `--max-new` |
| El repo rechaza el push por tamaño | checkpoint >100 MB | no lo subas; usa release asset |

---

## 12. Prompt listo para pegar

Si la persona solo quiere darte la tarea, este bloque contiene lo esencial:

```text
Trabajas en el repositorio Iris v1 (modelo de lenguaje desde cero en C/Julia/Rust).
Lee README.md, docs/ESTADO.md, docs/ENTRENAMIENTO.md y docs/AGENTES.md antes de actuar.

Objetivo: <reproducir el estado publicado | entrenar un ciclo | hacer que escriba frases |
cambiar X>.

Reglas:
- No borres ni sobrescribas checkpoints que no hayas creado.
- No lances dos ciclos a la vez.
- No cambies vocab/tokenizador y sigas con el mismo checkpoint.
- No afirmes nada sin la salida del comando que lo demuestra.
- Pregunta antes de cualquier ejecución de más de 30 minutos.
- Trabaja sobre una copia o rama; el proyecto original no se toca.

Empieza auditando el entorno y ejecutando las puertas de verificación de §8.
Luego propone un plan de 3 pasos con tiempos y espera confirmación.
```

---

## 13. Lo que un agente no debe prometer

- «Ya habla» con 1,3 M de tokens: no.
- «Lo entreno en una hora en CPU»: no, ni de lejos.
- «La GPU lo acelera sin cambiar código»: no, hay que portar el backward.
- «El checkpoint está en el repo»: no, pesa 1,63 GB y va aparte.
- «Los tests cubren la calidad del texto»: no, cubren la corrección del código.

La honestidad es parte del proyecto: el README dice que no habla y por qué. Un
agente que lo repite con números es más útil que uno que promete una demo.
