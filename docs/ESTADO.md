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
| F6 | Runtime cabrita | 52 tests (47 + 5) |

## Entrenamiento real

Dos ciclos de 2 h, con reinicios y reanudación por cursor:

| Ciclo | Pasos | Tokens | Velocidad | Loss | Texto |
|---|---|---|---|---|---|
| 1 (79 min, cortado por reinicio) | ~1 000 | 514 048 | 108,9 tok/s | 7,80 → 5,88 | ~1,15 MB |
| 2 (120 min) | 1 526 | 781 312 | 108,4 tok/s | 5,95 → 5,77 | 1,75 MB |

- Reparto del ciclo 2: c4 29,8 %, fineweb 36,3 %, wikipedia 33,9 %. RSS 7,34 GB.
- La loss se quedó **plana en 5,77 la segunda hora**. Sin diagnosticar.
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

F4d usó extractos de 300 caracteres y quedó peor en todos los τ; se conserva como `router_v0_f4d_backup.irrtr` y el artifact operativo volvió a F4c.

## Runtime

- 47 tests unitarios + 5 de integración en verde.
- El **motor de inferencia es una plantilla**: carga UKJ, resuelve rutas, gestiona TOJI y feedback, pero no ejecuta el transformer. La generación real, hoy, es `src/julia/generate.jl`.

## Lo que falta

| Tarea | Por qué importa |
|---|---|
| Reducir `vocab` 16384 → 1056 | ~15,7 M parámetros y ~15× menos GEMM de logits |
| Conectar o quitar las 16 hojas | 53 % del checkpoint es peso muerto |
| Caché KV | generación O(T) en vez de O(T²) |
| Diagnosticar la loss plana | puede ser lr, datos o Adam |
| Entrenar a escala | 3 860 M tokens para 193 M parámetros; CPU no llega |
| TOJI completo | creación automática de memoria en el pensamiento profundo |
| Sub-parámetros reales | `p4-one` hoy es solo una etiqueta |

## Comparación honesta

Con 1,3 M de tokens, Iris está en la fase en que un modelo aprende la frecuencia de palabras y algunos bigramas. Un transformer pequeño bien entrenado (10 M de parámetros, 1 000 M de tokens) escribe frases simples. Para eso hacen falta ~1 000× más tokens de los que ha visto este proyecto, o un modelo ~20× más pequeño con los mismos datos.
