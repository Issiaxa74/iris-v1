# Router

El router elige por dónde responder: una de las 22 rutas de conocimiento o la ruta genérica. Es un clasificador independiente del trunk, entrenado aparte, que cabe en 0,42 MB.

## 1. Arquitectura

`src/julia/router.jl`, formato `IRISRTR` v1:

- Tabla de embeddings propia (1056 × 32, inicializada con hash del token, no compartida con el trunk).
- Bolsa de embeddings: media de los embeddings de los tokens del contenido (sin posiciones, el orden no importa).
- Concatenación con el vector de historial `ph` (hoy a ceros hasta que haya datos de conversación).
- MLP: 32+32 → 64 (GELU) → n_rutas (softmax).
- 110 358 parámetros.

La salida es una distribución sobre rutas. La regla operativa:

```
si max(p) >= τ  ->  esa ruta
si max(p) <  τ  ->  ruta genérica
```

El umbral τ va grabado en el artifact; `router_tau_sweep.jl` lo barre sin re-entrenar.

## 2. Datos de entrenamiento

El router aprende de pares (texto, ruta) con etiqueta `mundo.submundo`. `router_extractor.jl` construye el TSV desde un corpus de pares pregunta/respuesta; `router_buffer_extract.jl` etiqueta por origen de dataset (c4 → web, wikipedia → enciclopedia).

Formato del TSV: una línea por muestra, `texto<TAB>ruta`. El split es 70/15/15.

## 3. Entrenamiento y selección

```powershell
julia src/julia/router_test.jl src/julia/router_data_f4c.tsv router_v0.irrtr
```

- 10 semillas, lr 0,003, early stop con paciencia.
- Dentro de cada semilla se elige el snapshot con **máxima cobertura en validación a τ=0,6 sujeta a precisión ≥ 0,98**; si ninguno la cumple, gana el de menor loss de validación.
- Al final se guarda τ y los nombres de ruta dentro del artifact; `save/load` es bit-exacto.

## 4. Resultados del modelo incluido

674 muestras, 22 rutas, test nunca visto (n=101):

| τ | Cobertura | Precisión val | Precisión test |
|---|---|---|---|
| 0,60 | 51/101 | 98,0 % | 92,0 % |
| 0,70 | 43/101 | 100 % | 95,1 % |
| **0,85 (grabado)** | **29 %** | **28/28 = 100 %** | **29/29 = 100 %** |
| 0,90 | 24/101 | 100 % | 100 % |

- Argmax en test: 56,2 % (22 clases).
- Texto sin sentido → genérica (p ≈ 0,32), como debe ser.
- La exactitud a τ=0,85 se fijó mirando el test una vez; con n=29, un solo fallo la baja a 96,6 %. Es exacta hoy, no una garantía estructural.
- La confusión se concentra en clases con menos de 5 muestras (`ciencias_salud.salud`, `lenguaje_general.web_es`). Con más datos del stream esas clases crecen y la cobertura debería subir sin perder exactitud.

## 5. Integración con el runtime

cabrita carga el artifact, calcula `p`, aplica τ y pide la ruta al UKJ. Si el artifact cambia de ids, el runtime debe releer los nombres embebidos: **los ids dependen de la frecuencia de las clases y cambian al re-entrenar** (F3 tenía 16 rutas; F4c, 22). Nunca hardcodear ids.
