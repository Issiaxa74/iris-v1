# Arquitectura

Iris se construyó por fases, con cada pieza verificada antes de pasar a la siguiente. Este documento describe lo que hay, no lo que se planea.

## 1. Modelo

Transformer decoder-only, implementado a mano en Julia (`src/julia/model.jl`), sin frameworks:

| Campo | Valor |
|---|---|
| `dim` | 1024 |
| `n_layers` | 14 |
| `n_heads` | 16 (cabeza de 64) |
| `ffn` | 4096 |
| `vocab` | 16384 |
| `ctx` | 2048 |
| `n_hojas` | 16 |

Layout de parámetros (un solo vector fp32 `p`, con rangos por nombre):

```
emb [d x vocab]
por bloque (t1..t14, e1..e16, sft):
  ln1g, ln1b, wqkv [d x 3d], wo [d x d], ln2g, ln2b, w1 [d x f], w2 [f x d]
fin: ln_g, ln_b
```

Total: 406 976 512 parámetros. **El forward usa solo `emb`, `t1..t14` y `fin`** (192 997 376 parámetros). Las 16 hojas y el bloque `sft` están asignados y se guardan, pero no participan en el cálculo ni reciben gradiente: son el hueco para el enrutado por conocimiento, todavía sin conectar.

Decisiones de implementación que importan:

- **Pre-LN** con residual, GELU (tanh) en la FFN, atención causal con escala `1/√(d/H)` fusionada en el softmax.
- **Sin dependencias** en el núcleo: solo `Random` y `LinearAlgebra` de la stdlib. Los GEMM usan BLAS.
- **Paralelismo por filas/cabezas** con hilos de Julia; el resultado es idéntico a 1 hilo (los tests lo comprueban).
- **Micro-batch** empaquetando secuencias de igual longitud por filas; sin cross-atención entre secuencias del batch.
- **Gradiente verificado** contra diferencias centrales por grupo de parámetros (rel ≤ 1e-3), causalidad probada con pares de secuencias y save/load bit-exacto.

## 2. Tokenizador

BPE por bytes (`src/julia/BPE.jl`), formato `IRISTOK` v1:

- 256 tokens base (bytes) + 800 fusiones = **1056 ids**.
- `encode` devuelve ids 0-based; el modelo indexa `emb` con `id-1`, así que el pipeline suma 1.
- El vocabulario del modelo (16384) es mayor que el del tokenizador (1056): los ids muertos nunca se generan al entrenar, pero sí pueden salir al muestrear si no se enmascaran. `generate.jl` los enmascara con `-Inf`.

## 3. Router

Clasificador independiente del modelo grande (`src/julia/router.jl`):

- Bolsa de embeddings propia (tabla de 1056 × 32) sobre los tokens del contenido + MLP (32→64→n_rutas, GELU).
- 110 358 parámetros, 0,42 MB, formato `IRISRTR` v1 con los nombres de ruta embebidos.
- Salida: distribución sobre rutas + umbral τ. Si `max p < τ`, la petición va a la ruta genérica.
- Se entrena con pares (texto, ruta) y contraejemplos de desambiguación; nunca por palabras-sonido.

## 4. Formato UKJ

Contenedor binario propio equivalente a GGUF: cabecera, tabla de rutas, tensores Float32 alineados y checksum. Permite abrir un archivo de varios GB y mapear solo la ruta pedida. Detalle completo en [UKJ.md](UKJ.md).

## 5. TOJI

Memoria viva de la conversación (no conocimiento): neuronas, satélites, neuronas anómalas, sectores y cables, persistidos en archivos `.mem` por sector. Detalle en [TOJI.md](TOJI.md).

## 6. Fases del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| F1 | UKJ v0.1: escritor Julia, lectores Rust y C con checksum | Hecho |
| F1b | Tokenizador BPE propio | Hecho |
| F2 | Núcleo de tensores en C (`tensor.c`) | Hecho, 19/19 |
| F2b | Modelo Julia + gradiente verificado | Hecho |
| F3 | Router clasificador | Hecho |
| F4a | Optimización de núcleos y micro-batch | Hecho |
| F4b | Pipeline de datos + ciclos de 2 h con checkpoint | Hecho |
| F4c/F4d | Router con más datos y calibración de τ | Hecho (F4c vigente) |
| F4e | Entrenamiento real a escala | **Pendiente** |
| F5 | TOJI v0.1 | Pendiente |
| F6 | Runtime cabrita | Hecho |
| F7 | Visión | Pendiente |

## 7. Por qué no usa PyTorch

El objetivo del proyecto es entender y controlar cada operación: kernels, gradientes, formato de pesos, memoria y enrutado. Un framework habría ocultado justo las partes que se querían construir. El coste es la velocidad: sin CUDA, este entrenador rinde ~108 tok/s en CPU, tres órdenes de magnitud por debajo de lo que necesita el modelo.
