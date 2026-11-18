# Plan de entrenamiento inicial de Iris (fase de datos)

Fecha: 2026-11-17. Objetivo: ~20 GB/día de texto procesado, en ciclos de CPU de ≤2 h con checkpoint.

## Los 5 datasets elegidos (conocimiento general + SFT)

| # | Dataset | Idioma | Licencia | Tamaño aprox. | Rol en Iris | Estado de Verificación |
|---|---|---|---|---|---|---|
| 1 | `wikimedia/wikipedia` (20231101.es) | ES | CC BY-SA 3.0 / 4.0 | ~4.5 GB | Mundo general español (historia, ciencias, cultura) | **Verificado** (Parquet streaming) |
| 2 | `wikimedia/wikipedia` (20231101.en) | EN | CC BY-SA 3.0 / 4.0 | ~20 GB | Conocimiento general inglés (bilingüe) | **Verificado** (Parquet streaming) |
| 3 | `HuggingFaceFW/fineweb-edu` | EN | ODC-By 1.0 | 1.3T tokens (streaming por shards) | Calidad educativa alta: ciencias, razonamiento, tecnología | **Verificado** (Parquet streaming) |
| 4 | `allenai/c4` (configuración `es`) *(Sustituto oficial de somosnlp)* | ES | ODC-By | ~150 GB (filtrable por shards) | Lenguaje español nativo contemporáneo (noticias, blogs, web) | **Verificado** (100% público, Parquet nativo) |
| 5 | `OpenAssistant/oasst2` | ES+EN | Apache-2.0 | ~135K mensajes | Mundo SFT: estructura de respuesta y diálogo | **Verificado** (Parquet streaming) |

*Nota sobre somosnlp:* `somosnlp/somos-clean-web-2023` no está disponible públicamente en la API de Hugging Face (retorna error 401 / no encontrado). Se activó su suplente oficial `allenai/c4` (`es`), el cual está completamente verificado y libre de restricciones.

Otros suplentes disponibles en reserva: `open-web-math/open-web-math` (matemáticas, 6.3M docs / 14.7B tokens), `PlanTL-GOB-ES/SQAC` (QA en español, CC BY-SA 4.0).

## Presupuesto por ciclo (CPU, sin GPU, ≤2 h)

- Objetivo diario: **20 GB/día** de texto crudo leído por streaming. ✅ Decisión (2026-11-17, elegida por el usuario): lectura adelantada por streaming (I/O barato, buffer filtrado con tope en disco); la digestión de entrenamiento es el cuello real: 68 tok/s ≈ 21 MB/día hoy → objetivo 100–200 MB/día tras optimizar núcleos (micro-batching + packing, se mide en F4). GPU aparcada. Ver `IRIS_MODEL_SPEC.md` §9.
- Reparto: 4 ciclos de ~2 h al día, ~5 GB por ciclo.
- Mezcla por ciclo (peso): wikipedia ES 15% · wikipedia EN 15% · fineweb-edu 25% · c4 (es) 30% · oasst2 15%.
- Cada ciclo guarda: checkpoint UKJ + cursor de streaming (dataset, shard, offset) → el siguiente ciclo continúa donde quedó.
- El mundo SFT (oasst2) entra en **todos** los ciclos.
- Estimación de rendimiento: 1–10 GB por ciclo de 2 h en CPU según fase (tokenización, forward/backward). La cifra real se calibra con el primer ciclo; el plan se ajusta al número medido, nunca por encima de 2 h.

## Reglas

1. Siempre streaming (HTTP Range sobre Parquet), sin descargar el dataset completo.
2. Filtro por idioma y calidad antes de tokenizar; el texto filtrado alimenta las rutas.
3. Checkpoint + cursor cada ciclo; si el ciclo se corta, se reanuda desde el cursor.
4. Los 20 GB/día son de **texto leído**; el filtrado puede reducir a menos texto útil (normal).
5. Se actualiza el cursor de cada dataset de manera independiente.
6. El esquema de etiquetado y calibración del router sigue la especificación en `IRIS_ROUTING_SCHEMA.md`.
7. Datasets gated (`bigcode/the-stack`, `cerebras/SlimPajama`, `lavita/medical-qa-*`): requieren HF_TOKEN del usuario; NO entran en el plan inicial. Si el usuario conecta un token, se añaden como ciclos extra.
8. `teknium/OpenHermes-2.5` queda descartado para producción (sin licencia declarada; compilación de outputs de GPT-4 con restricciones de OpenAI). Solo uso de investigación privada.
9. **La destilación es material COMPLEMENTARIO**: entra al mundo SFT con peso pequeño (estilo/diálogo). Iris NO aprende a hablar mediante destilación — va contra los principios del proyecto. La vía principal es next-token-prediction sobre el stream propio.

## Estado de Tareas

- [x] **co-ayudador de ia:** Verificada la licencia y streaming de `somosnlp` (no público) y activado el suplente oficial `allenai/c4` (`es`).
- [x] **co-ayudador de ia:** Preparado el esquema de etiquetado por rutas de 7 niveles y reglas de priorización (`IRIS_ROUTING_SCHEMA.md`).
- [x] **co-ayudador de ia:** Muestreo etiquetado (250 total) en `IRIS_MUESTRAS_ETIQUETADAS/` + spec del corpus sintético (`IRIS_SFT_SPEC.md`) + auditoría OpenHermes (descartado para producción).
- [x] **co-ayudador de ia:** Wikipedia ES/EN re-etiquetadas con rutas temáticas reales (37/50 y 42/50 específicas; el resto, artículos cortos sin señal temática, queda en nodo genérico).
- [x] **Desarrollador de ia:** Fase 1a — escritor UKJ (Julia) + lectores Rust y C con mmap parcial y checksum; prueba end-to-end OK.
- [x] **Desarrollador de ia:** F1b — tokenizer propio BPE (`iris/julia/BPE.jl`): 1056 tokens (vocab byte-base + 800 fusiones), roundtrip OK, determinista, guardado en `iris/iris.tok` (formato `IRISTOK`). Compactación inicial 1.77 chars/token sobre 250 muestras; se re-entrenará con el corpus real en streaming.
- [x] **Desarrollador de ia:** F2 — núcleo de tensores en C con p4-one (`iris/c/tensor.c`, test 14/14 OK): GEMM base cuantizada error 2.0%; bajo estrés la hot fusion de sub-parámetros baja el error a 0.18% (11x mejor). Contador de estrés por bloque con decaimiento y eviction verificados. Incluye softmax, layernorm, embedding lookup y argmax.
- [x] **Desarrollador de ia:** F2b — modelo Julia completo (`iris/julia/model.jl` + `model_test.jl` TODO OK): trunk 1024×14 + 16 hojas + mundo SFT = 406.976.512 par (0.407B). Núcleo C extendido (sinusoidales, GELU, atención causal, bloque pre-LN) 19/19 PASS. Gradiente analítico verificado contra numérico (rel ≤ 0.0004 en todos los grupos); causalidad y save/load OK. Medido: fwd 237 tok/s, fwd+bwd 68 tok/s ≈ 5.9M tokens/día (~21 MB/día).
- [x] **Desarrollador de ia:** F3 — router clasificador (iris/julia/router.jl + router_test.jl TODO OK): bag-of-embeddings por contenido + MLP; emb propia 1056×64 (no comparte el trunk), ph (historial) a CEROS hasta tener datos de F4; ruta 0 = "generica" (clase aprendida) + regla τ=0,60 del esquema co-ayudador. IRISRTR v1: artifact iris/router_v0.irrtr (108.816 par, 16 rutas, save/load bit-exacto). Verificado: gradiente numérico por grupos, forward independiente, 5 semillas → test argmax media 45,4% (datos escasos); operativo: precisión 100% en rutas p≥0,6 (3/3 test), cobertura 8–16%, nonsense→genérica p=0,14. F4: más muestras + destilado para subir cobertura.
- [x] **Desarrollador de ia:** F4a - optimizacion de nucleos + micro-batch en model.jl (CERRADO 2026-11-17, techo medido: el objetivo =340 tok/s requeriria GPU). Verificado con model_test.jl TODO OK a 1 y 8 hilos. Cambios: (A) fusion de kernels: gelu_grad de un solo tanh, softmax CE troceada por columnas con hilos, residuales con @. (diferencias <=1 ulp); (B) hilos Julia en layernorm/gelu/atencion por unidades (seq,cabeza)/Adam + danza BLAS=1 dentro de _attn!/_attn_bwd! (contencion del pool OpenBLAS medida: 12.8 -> 7.4 ms/capa en backward); (C) forward_loss_batch/backward_batch!: nb secuencias de igual T empaquetadas por filas, sin cross-atencion, forward_loss/backward! = caso nb=1; gradientes de layernorm gamma/beta anadidos (hueco preexistente de F2b, verificado vs numerico rel <=0.0025). EQUIVALENCIA BATCH (f4a_batch_test.jl): loss batch = media de individuales (rel 2.15e-09); gradiente = suma/nb (max abs 7.45e-09 sobre gmax 1.15e-02); sin cross-atencion (diff 0). NUMEROS (Ryzen 7 5825U, Julia=8 BLAS=8): forward solo ~500 tok/s; paso entrenando (fwd+bwd+Adam) B=2 = 116-129 tok/s (punto dulce); B=4 no mejora (RSS 8.6 GB de 13.8); ~10 M tokens/dia ~ 35 MB texto/dia. Adam 0.44-0.52 s/paso. Probes de referencia: f4a_probe6 (matriz hilos), f4a_probe7 (danza BLAS), f4a_probe8/9 (desglose batch).
- [x] **Desarrollador de ia:** F4b — pipeline streaming HF (HTTP Range sobre parquet): filtro idioma/calidad → buffer filtrado en disco con tope → tokenizador → ciclos de 2 h con cursor por dataset + checkpoint UKJ reanudable.
  Operativa 2026-11-18: tarea programada `IrisF4bCycle` + `iris/julia/run_f4b_detached.ps1` (mutex Local\IrisF4bCycle, log UTF-8 por corrida en `iris/logs/`), estado con `iris/julia/f4b_status.ps1`, runbook en `iris/F4B_RUNBOOK.md`. El ciclo sobrevive al cierre de OpenCode; ante reinicio del equipo se reanuda desde checkpoint+cursor con `Start-ScheduledTask -TaskName IrisF4bCycle`. Primer ciclo real: 79 min el 17/11 (108,9 tok/s, loss 7,80 a 5,88) y 30 min el 18/11 (checkpoint 12:22, cursor 12:27) antes del reinicio.
  CERRADA 2026-11-18: ciclo real completo de 2 h (12:47-14:48) reanudado desde checkpoint: 1526 pasos, 781.312 tokens, 108,4 tok/s medio sostenido, loss medio 5,7708 (arranque 5,95; plano en ~5,77 durante la segunda hora), RSS 7,34 GB, 1,75 MB de texto leido (0,87 MB/h). Coincide con el ciclo previo (108,9 tok/s). Cuello real: el entrenador; el lector Python del co-ayudador mide 0,5-0,7 MB/s (dato suyo, no re-verificado aqui). 20 GB/dia digeridos siguen fuera de alcance sin GPU. Siguiente: revisar el plateau de loss (lr) y decidir si se encadenan 4 ciclos/dia.
- [x] **Desarrollador de ia:** F4c CERRADA - re-entrenado router con 674 muestras (248 F3 + 426 del destilado via router_extractor.jl): 22 rutas (+6); lr 0.003 + seleccion operativa (max cobertura val a tau=0.6 con precision val >= 0.98, 10 semillas); test argmax 0.562 media / 0.594 mejor (F3: 0.454 a 16 clases). Calibracion final: artifact guarda tau=0.85 -> regla operativa EXACTA en val (28/28) y test (29/29) con cobertura ~29% (F3: 8-16%); ver router_tau_sweep.jl y spec 11.1. Backups F3: iris/router_v0_f3_backup.irrtr + iris/julia/router_data_f3_backup.tsv.
- [ ] **Desarrollador de ia:** F1c — BPE re-entrenado sobre el stream real (20 GB/día) cuando arranque la ingesta.
- [x] **Desarrollador de ia:** Corpus complementario convertido desde `OpenBot\Downloads` → `IRIS_DESTILADO\descargas\` (213 pares resueltos con rutas por contenido: historia 100, ciencia/tecnología 100, salud 13; 0 fallbacks, JSON válido).
- [ ] **Ayudador general (alcance reducido):** destilación COMPLEMENTARIA — solo rellenar huecos (matemáticas de Qwen, algo de salud vía Z.ai). NO enviar lotes masivos: el grueso ya existe.
