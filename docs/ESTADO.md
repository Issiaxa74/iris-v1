# Estado y medidas

Registro de lo que se construyó, lo que se verificó y lo que falta. Los números salen de ejecuciones reales en un Ryzen 7 5825U (8 núcleos / 16 hilos, Windows 11, 16 GB RAM, sin GPU), Julia 1.13.3.

## Resumen

| Fase | Qué se hizo | Verificación |
|---|---|---|
| F1 | UKJ v0.1: escritor Julia, lectores Rust y C | 3 rutas de prueba, bytes idénticos |
| F1b | BPE por bytes, 1056 tokens | roundtrip y vocabulario comprobados |
| F2 | Núcleo de tensores en C | 19/19 tests; GEMM 2,0 % de error, 0,18 % bajo estrés |
| F2b | Modelo Julia 0.407 B | gradiente = numérico (rel ≤ 1e-3), causal, save/load exacto |
| F3 | Router v0 (16 rutas) | forward independiente rel 2,5e-8, gradiente OK |
| F4a | Fusión de kernels, hilos, micro-batch | 68 → 108–129 tok/s; test batch OK |
| F4b | Pipeline de datos + ciclos de 2 h | 2 ciclos completos, cursores reanudables |
| F4c | Router v1 (22 rutas) | exacto a τ=0,85 en val y test |
| F6 | Runtime cabrita | 57 tests (50 + 7) |
| F7 | Motor transformer + export UKJ real | paridad de logits con Julia, misma generación greedy, 19 tok/s |

## Entrenamiento real

Dos ciclos de 2 h, con reinicios y reanudación por cursor:

| Ciclo | Pasos | Tokens | Velocidad | Loss (media acumulada) | Texto |
|---|---|---|---|---|---|
| 1 (79 min, cortado por reinicio) | ~1 000 | 514 048 | 108,9 tok/s | 7,80 → 5,88 | ~1,15 MB |
| 2 (120 min) | 1 526 | 781 312 | 108,4 tok/s | 5,95 → 5,77 | 1,75 MB |

- Reparto del ciclo 2: c4 29,8 %, fineweb 36,3 %, wikipedia 33,9 %. RSS 7,34 GB.
- En los ciclos publicados solo se registró `lossac / lossn`: la media
  **acumulada desde el inicio del proceso**, no la loss del paso. La meseta de
  5,77 es la media estabilizada; no se puede afirmar que el modelo dejara de
  mejorar. La versión actual del ciclo imprime además la media de la ventana de
  40 s.
- Total: ~1,30 M tokens sobre 193 M de parámetros activos → 0,007 tokens/parámetro.

## Rendimiento del entrenador

| Medida | Valor |
|---|---|
| Forward T=256 | ~500 tok/s |
| Fwd+bwd+Adam, B=1 | ~105 tok/s |
| Fwd+bwd+Adam, B=2 | 108–129 tok/s |
| Fwd+bwd+Adam, B=4 | 104–125 tok/s (RSS 8,6 GB) |
| Checkpoint (fp32) | 1,55 GB cada 15 min |
| Texto digerido | 0,87 MB/h |

Cuello de botella: los GEMM de pesos (~6·N FLOPs por token) contra los ~270 GFLOP/s efectivos del chip, más el estado de Adam. El micro-batch amortiza Adam; el resto es matemática del hardware.

## Router

| Versión | Rutas | Muestras | Argmax test | τ | Cobertura | Precisión |
|---|---|---|---|---|---|---|
| F3 | 16 | 248 | 45,4 % | 0,6 | 8–16 % | 100 % (n=3) |
| F4c (vigente) | 22 | 674 | 56,2 % | 0,85 | 29 % | 100 % (28/28 val, 29/29 test) |
| F4d (descartado) | 22 | 825 | — | 0,85 | — | 94,7 % test |

F4d usó extractos de 300 caracteres y quedó peor en todos los t; el artifact operativo volvió a F4c. El backup F4d no se publica: solo se incluye `models/router_v0.irrtr` (F4c).

## Runtime e inferencia

- 50 tests unitarios + 7 de integración en verde.
- El motor `transformer` ejecuta el forward real del trunk desde el UKJ + manifiesto, con caché KV, atención causal, GELU/LN y muestreo temp/top-k/top-p/seed. Solo carga la ruta `trunk` (736,2 MB del modelo real).
- Paridad con Julia: la prueba golden compara 8 logits y el argmax; con el checkpoint real, Julia y Rust generan la misma secuencia de 12 tokens con `--temp 0`.
- Rendimiento: ~19 tok/s en CPU con el checkpoint real (un matvec por token); el entrenador procesa secuencias completas y por eso es más rápido.
- `template` y `stats` siguen disponibles como diagnóstico, pero ya no son el motor para generar texto.

## Lo que falta

| Tarea | Por qué importa |
|---|---|
| Reducir `vocab` 16384 → 1056 | ~15,7 M parámetros y ~15× menos GEMM de logits |
| Conectar o quitar las 16 hojas | 53 % del checkpoint es peso muerto |

| Validación held-out | no hay perplejidad de validación, solo loss de entrenamiento |
| Diagnosticar la posible meseta | puede ser lr, datos o Adam |
| Publicar el checkpoint | 1,63 GB no cabe en el repo; release asset de hasta 2 GB |
| Entrenar a escala | 3 860 M tokens para 193 M parámetros; CPU no llega |
| TOJI completo | creación automática de memoria en el pensamiento profundo |
| Sub-parámetros reales | `p4-one` hoy es solo una etiqueta |

## Comparación honesta

Con 1,3 M de tokens, Iris está en la fase en que un modelo aprende la frecuencia de palabras y algunos bigramas. Un transformer pequeño bien entrenado (10 M de parámetros, 1 000 M de tokens) escribe frases simples. Para eso hacen falta ~1 000× más tokens de los que ha visto este proyecto, o un modelo ~20× más pequeño con los mismos datos.
