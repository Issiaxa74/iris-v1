# IRIS — Diseño del proyecto

Asistente conversacional + razonamiento + visión (fase futura), construido desde cero con **C**, **Rust** y **Julia**.

> **Regla de publicación**: NO subir nada a GitHub hasta tener el primer modelo UKJ funcional Y el runtime cabrita terminado. Todo queda local.

> **Principio de destilación**: material COMPLEMENTARIO solo (estilo/diálogo del mundo SFT). Iris NO aprende a hablar mediante destilación — eso va contra los principios del proyecto. La vía principal de aprendizaje es next-token-prediction sobre el stream propio con la arquitectura UKJ + rutas.

## 1. Formato UKJ (formato de modelo propio)

- Contenedor propio tipo GGUF/safetensors con **sub-parámetros**: `p4-one` = 1 parámetro con 4 sub-parámetros comprimidos; máximo previsto `p300-one`.
- **Mecánica de estrés**: cada uso activa el parámetro (contador); con muchas activaciones libera temporalmente sus sub-parámetros, luego se re-comprimen. Los sub-parámetros nunca se exponen sin estrés → ahorro de RAM.
- Cabecera + tabla de rutas → **mmap parcial**: solo la ruta activa va a RAM.
- Tokenizer y vocabulario: archivos separados (`iris.tok`, formato IRISTOK, BPE propio).
- Especificación binaria: `iris/UKJ_SPEC.md`. Implementada v0.1: escritor Julia (`iris/julia/UKJ.jl`), lectores Rust (`iris/rust/ukj_reader`) y C con mmap (`iris/c/ukj.c`).

## 2. Rutas jerárquicas de conocimiento (routing MoE)

- 7 niveles configurables y extensibles: mundo→submundo→categoría→subcategoría→tema→subtema→concepto.
- Solo se cargan los pesos de la ruta activada; **mundo SFT siempre activo** (mundo plano, sin niveles, entrenado aparte).
- Temas superpuestos: se activan en todas sus rutas con contexto distinto; una se prioriza.
- **Router por contenido semántico, nunca por sonido** ("¿en qué estado está el mundo?" ≠ Estados Unidos). Set de validación con frases trampa.
- Fallback honesto: si no hay ruta, Iris lo dice.
- Esquema de etiquetado y reglas: `Shared/IRIS_ROUTING_SCHEMA.md`.

## 3. TOJI — memoria del usuario

- Elementos: **sectores** (un `.mem` por sector), **neuronas** (palabras), **satélites** (recuerdos/emociones de esa neurona), **neuronas anormales** (concepto del usuario + matiz), **cables** (conexiones entre sectores).
- Se crean **instintivamente durante el pensamiento profundo antes de responder**.
- Persistencia: archivos binarios `.mem` (magic `TOJI`), leídos como "idioma natal" sin gastar tokens de contexto.
- Especificación: `iris/TOJI_SPEC.md`.

## 4. Modelo objetivo

- **0.6B parámetros**, conversacional.
- Inferencia (Rust, runtime **cabrita**): solo carga la ruta activa; **<1 GB RAM ideal**.
- Entrenamiento: **5 GB RAM máx**, CPU, ciclos de ≤2 h, checkpoint en `checkpoints/`, ~20 GB/día streaming HF.
- datasets: wikipedia ES/EN, fineweb-edu, c4-es, oasst2 (`Shared/IRIS_TRAINING_PLAN.md`).

## 5. Entrenamiento

- Núcleo: next-token-prediction sobre la ruta activa (+ mundo SFT siempre).
- **Feedback por terminal**: `/si [texto]`, `/no`, `/up` (ruta bien/respuesta mal), `/ent-my archivo.txt` (LoRA complementario). Especificación: `iris/FEEDBACK_SPEC.md`.
- Router: clasificador ligero por contenido con frases trampa.

## 6. Reparto de lenguajes

| Componente | Lenguaje |
|---|---|
| Núcleo de tensores, cuantización p4-one, mmap UKJ/TOJI | **C** |
| **Inferencia + runtime cabrita** | **Rust** (@Programador/web/full stack construye cabrita) |
| **Entrenamiento** | **Julia** |
| Cabrita (app de usuario) + web de documentación | Rust / estático multi-archivo |

## 7. Orden de construcción (arquitectura completa ANTES de entrenar)

1. ✅ F1 — UKJ v0.1 (Julia escribe; Rust y C leen con mmap + checksum).
2. ✅ F1b — Tokenizer BPE propio (`iris.tok`).
3. ✅ F2 — Núcleo de tensores en C con p4-one (iris/c/tensor.c, test 14/14): GEMM base 2.0% de error; bajo estrés la hot fusion de sub-parámetros baja a 0.18%.
4. F2b ✅ — Decisiones del usuario tomadas (escalado por fases, historial mixto, graduación) + núcleo C completo (19/19 PASS) + modelo Julia con test verde: 406,976,512 par (0.407B), gradiente verificado contra el numérico en todos los grupos, causalidad OK, save/load exacto. Throughput CPU medido: fwd+bwd 68 tok/s ≈ 5.9M tokens/día (~21 MB texto/día) — objetivo 20 GB/día en revisión (`IRIS_MODEL_SPEC.md` §9).
5. ✅ F3 — Router clasificador ligero (router.jl + router_test.jl TODO OK): bag-of-embeddings por CONTENIDO + MLP; 108.816 par (0,42 MB fp32), formato IRISRTR, artifact iris/router_v0.irrtr (16 rutas, save/load bit-exacto, forward independiente y gradiente numérico por grupos OK). Datos: 248 muestras 70/15/15 — argmax test 45,4% (media 5 semillas; datos escasos, honesto); operativo: precisión 100% en rutas confiadas p≥0,6 (3/3 test, 6/6 val), cobertura 8–16%, fallback τ→genérica OK (nonsense p=0,14). F4 escala datos (destilado + más muestras) para subir cobertura.
6. F4 — Pipeline streaming HF → tokens → ciclos 2 h → checkpoints → UKJ.
7. F5 — TOJI v0.1 en C/Rust.
8. ✅ F6 — Cabrita construido por @Programador/web/full stack en `Shared/CABRITA/`: 52/52 tests, tokenizer y UKJ de prueba byte-idénticos, REPL + `/si /no /up /ent-my` verificados en disco.
9. ✅ F6b — Web de documentación multi-archivo en `Shared/IRIS_DOCS_WEB/` (5 páginas autocontenidas, 0 enlaces rotos).
10. F7 — Visión (módulo aparte, después).
11. **GitHub solo cuando exista: primer UKJ funcional + cabrita terminado.**

## 8. Pendiente del usuario

- Destilación complementaria con alcance reducido (@Ayudador general): solo rellenar huecos (matemáticas de Qwen, algo de salud). Corpus base ya convertido: Shared/IRIS_DESTILADO/descargas/ (213 pares resueltos desde OpenBot\Downloads).
- Publicación final de Iris.
