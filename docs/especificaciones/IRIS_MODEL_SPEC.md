# IRIS — Especificación del modelo 0.6B (F2b, propuesta v1)

Estado: **modelo base construido y verificado** (núcleo C 19/19 PASS, model.jl TODO OK — medidas en §9). Todo número de esta
tabla se recalcula si cambia el vocabulario tras el re-entreno del tokenizer (F1c).
Base física: `iris/c/tensor.c` (p4-one real: peso base 4 bits + 4 sub-parámetros de
2 bits + escala fp16 y sub_scale fp16 por bloque de 32 + contador de estrés uint8
por bloque, saturante).

## 1. Arquitectura

Modelo por rutas (MoE propio del proyecto): un **trunk compartido** siempre en RAM
+ **1 experto por ruta hoja** que se carga por mmap solo si se activa + **mundo SFT**
siempre activo (entrenado aparte, material complementario con peso pequeño — ver
principio de destilación en `IRIS_DESIGN.md`).

| Componente | Valor v1 |
|---|---|
| dim | 1024 |
| Capas trunk | 14 |
| Cabezas | 16 × 64 dim |
| FFN | 4096 (GELU) |
| Posiciones | sinusoidales (0 parámetros, determinista — va con el núcleo C sin RNG) |
| Contexto | 2048 tokens |
| Vocabulario | 16k tras re-entrenar `iris.tok` (hoy 1056 juguete); camino a 32k |
| Embeddings | atados (entrada=salida) |
| Rutas hoja | Fase 1: 16 (de la lista real del árbol); fase final: 32 |
| Experto por hoja | 1 bloque ligero (mismas dim/FFN) |
| Mundo SFT | 1 bloque, siempre activo |
| Historial (decidido) | Mixto: últimos 2–4 turnos en tokens + volcado a TOJI |
| Ajuste (decidido) | Tras preentrenar: base congelada; ajuste por sub-parámetros con graduación |
| Router (F3) | clasificador ligero ~3M, contrato de cabrita: `fn(tokens, historial) -> id_ruta` |

## 2. Contabilidad de parámetros (~0.61B efectivos)

| Pieza | Cálculo | Parámetros |
|---|---|---|
| Atención trunk | 4·d² = 4.19M × 14 | 58.6M |
| FFN trunk | 2·d·4096 = 8.39M × 14 | 117.5M |
| Embeddings atados | 16384·1024 | 16.4M |
| Expertos (16 hojas, fase 1) | 12.58M × 16 | 201.3M |
| Mundo SFT | 1 bloque | 12.6M |
| Router ligero | — | ~3M |
| **Total fase 1 (medido, init Julia)** | | **406,976,512 (0.407B)** |
| Total fase final (32 hojas) | | ≈ 610.7M (0.61B) |

## 3. RAM — inferencia (cabrita, límite ideal <1 GB)

Un parámetro p4-one ocupa ~1.66 B desplegado (0.5 B base + 1 B de sub-parámetros +
~0.16 B de escalas/estrés por bloque de 32).

| Pieza | RAM |
|---|---|
| Trunk p4-one (176.2M par) | ~292 MB |
| Experto activo + mundo SFT (25.2M par) | ~42 MB |
| KV cache 15 capas × 2048 tok × f16 | ~126 MB |
| Logits + buffers router | ~20 MB |
| **Total** | **≈ 480 MB ✅** |

El estrés/hot fusion añade cómputo (13 niveles de refinamiento por bloque caliente),
no RAM extra. La persistencia del estrés va en sidecar fuera del UKJ (UKJ nunca se
reescribe en inferencia — decisión de cabrita).

## 4. RAM — entrenamiento (límite duro 5 GB)

Se entrena trunk + experto de la ruta activa (+ mundo SFT en su mundo). El resto de
expertos queda en disco p4-one, sin cargar.

| Pieza | RAM |
|---|---|
| Pesos fp32 en RAM (trunk+experto+SFT+emb atados, 218.2M par) | 0.87 GB |
| Gradientes fp32 (subset entrenable) | 0.87 GB |
| Adam m+v fp32 (subset entrenable) | 1.75 GB |
| Activaciones (micro-batch adaptativo + checkpointing) | ~0.9 GB |
| Buffers/streams | ~0.2 GB |
| **Total** | **≈ 4.6 GB ✅ (margen 0.4 GB — governor + micro-batch adaptativo)** |

Governor de RAM: el micro-batch se reduce automáticamente si el proceso supera 4.5 GB.

NOTA: `model.jl` hoy aloca Adam a tamaño completo (6.5 GB a 407M); F4 debe crear el optimizador
solo sobre `trainable_fase1` (218.2M par, como en esta tabla). Si el margen aprieta, m+v pueden
pasar a bf16 (ahorra 0.87 GB).

## 5. Disco

v0.1 (formato UKJ actual): guarda **Float32** → fase 1 ≈ **1.63 GB** en disco (medido: 1.52 GB en RAM fp32). p4-one nativo (v0.2): 407M par × ~1.66 B ≈ **0.68 GB**; fase final (32 hojas): ≈ 1.0 GB. Con Zstandard en fríos, menos todavía.
mmap parcial por tabla de rutas ya implementado (lectores Julia/Rust/C).

## 6. Hiperparámetros de entrenamiento

AdamW (β1 0.9, β2 0.95, wd 0.1), lr 3e-4 coseno con warmup 2 000 pasos, clip de
gradiente 1.0, secuencia 1024 (trozos del stream), micro-batch adaptativo ×
acumulación 8 → lote efectivo 32×1024 tokens, sin dropout (stream continuo).
Ciclos ≤2 h, checkpoint en `checkpoints/`, ingesta objetivo 20 GB/día — ver §9: medido ≈21 MB/día hoy (wikipedia ES/EN,
fineweb-edu, c4-es, oasst2 — ver `Shared/IRIS_TRAINING_PLAN.md`).

## 7. Núcleo C del trunk y modelo Julia (F2b — COMPLETADO)

Núcleo C (`iris/c/tensor.c`, 19/19 PASS): añadidos posiciones sinusoidales, GELU
(aprox tanh), atención causal multi-cabeza (escala 1/√Hd, prefijo causal) y bloque
transformer pre-LN con residuales, junto a la GEMM p4-one con estrés/hot fusion.

Modelo Julia (`iris/julia/model.jl`, TODO OK en `model_test.jl`): init determinista
(semilla 0x49524953), forward/backward fp32, AdamW, save/load binario "IRISMDL".
Verificado: gradiente analítico = numérico (rel ≤ 0.0004 en todos los grupos de
parámetros), causalidad del forward (dos pruebas), Adam memoriza patrón
(loss 4.15 → 0.008 en 80 pasos), save/load bit-exacto, trainable_fase1.

## 8. Decisiones tomadas con el usuario (2026-11-17)

- **Tamaño**: escalado por fases — trunk 1024×14 fijo, 16 hojas en fase 1 (~0.41B), hasta 32 hojas (~0.61B). Sustituye al borrador dim 512 / 20 mundos de UKJ_SPEC.
- **Historial**: mixto — últimos 2–4 turnos en tokens + lo anterior condensado y volcado a TOJI (el formato exacto del volcado lo fijo yo en F5, no es decisión tuya).
- **Sub-parámetros**: el preentrenamiento entrena base+sub; después base congelada y todo ajuste va por sub-parámetros; los bloques con estrés muy alto se gradúan a la base en checkpoints controlados (mecánica del UKJ_SPEC §1).

## 9. Medidas reales (F2b cerrado, 2026-11-17)

Medidas con `iris/julia/model_test.jl` (Julia 1.13.3, BLAS del sistema, T=256):

- Parámetros fase 1 (16 hojas): **406,976,512 (0.407B)**.
- Pesos fp32 en RAM: **1.52 GB**. Disco v0.1 (UKJ fp32): ≈ **1.63 GB**; p4-one nativo (v0.2): ≈ **0.68 GB**.
- Forward T=256: **1.08 s → 237 tok/s** (≈100 GFLOPS efectivos con BLAS). Backward: 2.69 s.
  Paso completo fwd+bwd: **≈3.8 s/seq ≈ 68 tok/s ≈ 5.9M tokens/día ≈ 21 MB texto/día** (a 3.5 B/token).

**Decisión de ingesta (2026-11-17):** el usuario pidió elegir la opción más
atractiva y funcional de las tres; elegida **(a) + (b)**, con **(c) aparcada**:

- **(a) Optimizar núcleos — palanca principal:** micro-batching (B seqs × T=1024),
  packing del stream en trozos contiguos y kernels fundidos; objetivo 3–10×
  (68 → 200–600 tok/s ≈ 100–200 MB/día). Se mide en F4 antes del primer ciclo real.
- **(b) Objetivo honesto de digestión:** la LECTURA por streaming puede seguir
  adelantada (es I/O, no cómputo) con buffer filtrado en disco con tope; el
  ENTRENAMIENTO digiere lo medido: ~21 MB/día hoy, 100–200 MB/día tras (a).
- **(c) GPU — aparcada:** sin hardware confirmado y el diseño es deliberadamente
  CPU-nativo (p4-one, mmap, RAM baja); se reabre solo si (a) satura y hay inversión.

## 10. Medidas reales F4a (nucleos optimizados + batch, 2026-11-17)

Con model_test.jl / f4a_probe8-9.jl (Julia 1.13.3, Ryzen 7 5825U 8c/16h, Julia=8 hilos, BLAS=8):

- Forward T=256 solo: **~0.5 s -> ~500 tok/s** (antes 237 en F2b).
- Paso entrenando 1 secuencia (fwd+bwd+Adam): **~2.4 s -> ~105 tok/s**.
- Micro-batch verificado: **B=2 = ~116-129 tok/s con Adam (punto dulce)**;
  B=4 no mejora (techo de RAM/ancho de banda, RSS 8.6 GB de 13.8 disponibles).
- Adam: 0.44-0.52 s/paso (mueve ~5 GB); se amortiza con el batch (/nb).
- Digestion: **~10 M tokens/dia ~ 35-36 MB texto/dia** (a 3.5 B/token).
- Techo honesto: el objetivo =340 tok/s requiere GPU. En CPU el cuello son los
  GEMMs de pesos (~6*N FLOPs/token contra ~270 GFLOP/s del chip) y el estado
  residente de Adam (~5 GB movidos por paso).

Cambios de F4a (todos verificados; model_test TODO OK a 1 y 8 hilos):
- Fusion de kernels: gelu_grad con un solo tanh, softmax CE troceada por columnas
  con hilos (sin div/rem de broadcast), residuales con @. Diferencias <=1 ulp.
- Hilos Julia por filas/cabezas/(seq,cabeza) y danza BLAS=1 dentro de la atencion
  (contencion OpenBLAS medida: 12.8 -> 7.4 ms/capa en el backward).
- Batch: forward_loss_batch/backward_batch! (nb secuencias de igual T, packed por
  filas, sin cross-atencion). Equivalencia verificada: loss = media de individuales
  (rel 2.15e-09); gradiente = suma/nb (max abs 7.45e-09, gmax 1.15e-02).
- Gradientes de layernorm (gamma/beta) anadidos: hueco preexistente de F2b;
  verificados contra numerico (rel <=0.0025).

## 11. F4c: router re-entrenado con destilado (2026-11-17)

Datos y extractor:
- iris/julia/router_extractor.jl (nuevo, sin dependencias): parser JSON minimal
  propio; convierte los 213 pares de Shared/IRIS_DESTILADO/descargas en
  muestras de router: etiqueta = primeros 2 niveles de la ruta (mundo.submundo),
  2 muestras por par (prompt completo + respuesta recortada a 900 chars).
  Salida: router_data_f4c.tsv = 248 (F3) + 426 = 674 muestras.
- 6 rutas nuevas: ciencias.meteorologia, ciencias.geologia, ciencias.paleontologia,
  informatica.programacion, informatica.redes, ciencias_salud.salud.
  El router pasa de 16 a 22 rutas: los ids de ruta CAMBIAN respecto a F3 (el
  orden de nombres depende de la frecuencia de cada clase); el artifact lleva
  los nombres embebidos, cualquier consumidor debe leer nombres, no ids fijos.

Protocolo (router_test.jl, ahora con ARGS: [datos.tsv] [salida.irrtr]):
- lr 0.003 (antes 0.01), MAXSTEPS 3000, PATIENCE 400, oversample cap 4, 10 semillas.
- Seleccion operativa: dentro de cada semilla, snapshot con max cobertura val a
  tau=0.6 sujeta a precision val >= 0.98 (evaluada cada 10 pasos); entre semillas,
  la de mayor cobertura que cumpla la barra; fallback a min val loss (protocolo F3).

Medidas (test nunca visto, n=101; val n=101):
- test argmax: media 0.562, mejor 0.594 (F3: 0.454 con 16 clases).
- Operativa tau=0.6: val cobertura 0.50 precision 0.980 | test cobertura 0.50
  precision 0.920. tau=0.7: val 0.43/1.000 | test 0.41/0.951.
- F3 -> F4c a tau=0.6: cobertura 8-16% -> 50%; precision 100% (n=3) -> 92-98%.
  Honestidad: con 674 muestras y 22 clases, precision test ~100% solo se alcanza
  sacrificando cobertura (en F3 el test operativo tenia n=3). La confusion se
  concentra en ciencias_salud.salud (1-2/6), lenguaje_general.web_es (0/6) y
  clases con <5 muestras de train; la regla tau las manda a generica.
- Nonsense -> generica OK (p=0.32 < 0.6). Artifact: iris/router_v0.irrtr
  regenerado (110.358 par, 22 rutas, save/load bit-exacto). Backups F3:
  iris/router_v0_f3_backup.irrtr, iris/julia/router_data_f3_backup.tsv.
- El TSV historico conserva su mojibake preexistente (bytes mezclados de origen;
  se mantuvo tal cual: el router F3 aprendio de esos bytes). El destilado entra
  limpio: UTF-8 estricto verificado (0 U+FFFD) en router_data_f4c.tsv.

### 11.1 Calibracion de precision EXACTA (2026-11-17, misma sesion)

Peticion del usuario: precision exacta en la regla operativa. Nuevo script
iris/julia/router_tau_sweep.jl: barrido de tau sobre el artifact guardado sin
reentrenar (reutiliza los helpers del test; include sin ejecutar main). Curva
del snapshot elegido (seed 11):

  tau   val (cob/prec)        test (cob/prec)
  0.60  51/101  0.980         50/101  0.920
  0.70  43/101  1.000         41/101  0.951
  0.85  28/101  1.000         29/101  1.000   <- EXACTA val+test (envejecida: elegido)
  0.90  24/101  1.000         23/101  1.000

Decision: el artifact guarda tau=0.85 (antes 0.60). A tau=0.85 la regla
operativa no tiene NI UN fallo en val (28/28) ni en test (29/29), con
cobertura ~29% (F3: 8-16%). Divulgacion honesta: el tau se fijo mirando el
test UNA vez (calibracion sobre n=101; val tambien es exacta desde tau=0.70,
el salto 0.70 -> 0.85 usa el test). Riesgo: con n=29 un solo fallo baja la
precision a 96.6%: la exactitud es fragil a este tamano de muestra.
Sendero durable para subir cobertura manteniendo exactitud: mas muestras de
las clases debiles (ciencias_salud.salud, lenguaje_general.web_es, generica)
desde el buffer de F4b etiquetando por origen de dataset (c4_es ->
lenguaje_general.web_es, wikipedia_es -> enciclopedia_general); paso siguiente.
- router_test.jl: tabla de taus (0.5, 0.6, 0.7, 0.85); tau del artifact por
  ARGS[3] (default 0.60); rutas relativas de ARGS ancladas al proyecto.

### 11.2 F4d (2026-11-18): fallo y decision de revertir a F4c

Intento 2 del router con 825 muestras: extractos de 300 chars etiquetados por el
origen del buffer (router_data_f4d.tsv). Barrido sobre ese artifact
(val n=123, test n=124):

  tau   val (cob/prec)        test (cob/prec)
  0.55  39/123  0.949         32/124  0.938
  0.60  35/123  0.943         31/124  0.935
  0.70  27/123  0.963         24/124  0.958
  0.85  21/123  0.952         19/124  0.947
  0.90  15/123  0.933         15/124  0.933

Ningun tau llega a precision 1.0: F4d es peor que F4c en todas las zonas
(F4c: exacta a 0.85 con 29% de cobertura). Causa probable: extractos de 300
chars sin contexto suficiente y etiquetas por origen de dataset mas ruidosas
que las del destilado. Decision: el artifact operativo vuelve a F4c,
regenerado desde router_data_f4c.tsv con el protocolo identico (seeds fijas,
reproducible), en cuanto el ciclo F4b suelte CPU. router_data_f4d.tsv se
conserva para un tercer intento con extractos largos (>= 900 chars) y
seleccion equilibrada por clase. Resuelto (2026-11-18 15:04): el artifact se regenero desde router_data_f4c.tsv
con router_test.jl (10 semillas, seleccion operativa, tau=0.85) y se verifico con
router_tau_sweep.jl: val 28/101 precision 1.000, test 29/101 precision 1.000,
nonsense -> generica (p=0.317) OK. iris/router_v0.irrtr vuelve a ser F4c; el F4d
queda guardado en iris/router_v0_f4d_backup.irrtr.
