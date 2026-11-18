# IRIS_SFT_SPEC.md — Especificación del Corpus Sintético SFT para IRIS

Versión: 1.0 · Elaborado por: **co-ayudador de ia** · Consumidor: **Desarrollador de ia** (pipeline de entrenamiento)
Propósito: Contrato exacto que define el corpus de ≥4000 mensajes sintéticos generados por Z.ai para alimentar el **Mundo SFT** (plano, siempre activo) de IRIS.

---

## 1. Volumen y Reparto ES/EN

| Parámetro | Valor |
|---|---|
| Total mínimo de ejemplos | **4 000** |
| Distribución de idioma | 60% español (≥2 400 ej.) · 40% inglés (≥1 600 ej.) |
| Formato de cada ejemplo | Un par `{human, assistant}` con hasta 2 turnos de seguimiento opcionales |
| Longitud objetivo por respuesta | 80–600 tokens (texto plano, sin markdown salvo listas numeradas/viñetas) |

---

## 2. Esquema de Campos por Ejemplo (JSON / JSONL)

Cada línea del archivo de salida es un objeto JSON con exactamente estos campos:

```json
{
  "id": "string — sft_{uuid4_8chars}",
  "language": "string — 'es' | 'en'",
  "style": "string — ver sección 3",
  "situation": "string — ver sección 4",
  "topic_route": {
    "mundo": "string",
    "submundo": "string",
    "categoria": "string"
  },
  "stays_on_topic": true,
  "conversations": [
    {"role": "human", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "quality_flags": {
    "no_topic_drift": true,
    "no_hallucination_markers": true,
    "language_consistent": true,
    "length_in_range": true
  },
  "generator": "string — 'z.ai/model-name'"
}
```

### Campos obligatorios y validaciones automáticas

| Campo | Tipo | Regla de validación |
|---|---|---|
| `id` | string | Formato `sft_[a-f0-9]{8}` único en el corpus |
| `language` | enum | Solo `"es"` o `"en"` |
| `style` | enum | Uno de los 7 estilos listados en §3 |
| `situation` | enum | Uno de los 12 situaciones listadas en §4 |
| `topic_route.mundo` | string | No vacío; coincide con taxonomía IRIS (§5) |
| `stays_on_topic` | bool | Siempre `true`; ejemplos con `false` son **rechazados** |
| `conversations` | array | Mínimo 1 turno; máximo 3 turnos; primer rol siempre `"human"` |
| `conversations[*].content` | string | Mínimo 10 caracteres; sin PII ni contenido dañino |
| `quality_flags.no_topic_drift` | bool | `true` obligatorio; ver regla §6.1 |
| `quality_flags.length_in_range` | bool | Respuesta final entre 80 y 600 tokens |
| `generator` | string | Nombre del modelo Z.ai que lo generó |

---

## 3. Distribución de Estilos (7 estilos, suma = 100%)

| Estilo | Descripción | Porcentaje |
|---|---|---|
| `explicacion_didactica` | Respuesta paso a paso, clara, con ejemplos | 25% |
| `respuesta_directa` | Pregunta factual, respuesta concisa y precisa | 20% |
| `comparativa` | Contraste de dos conceptos, ventajas/desventajas | 12% |
| `resumen` | Síntesis breve de un texto, lista de puntos clave | 12% |
| `razonamiento_cadena` | Chain-of-thought visible, razonamiento explícito | 15% |
| `creativo_estructurado` | Texto creativo (relato, poema, diálogo) bien estructurado | 8% |
| `seguimiento_conversacional` | 2–3 turnos de profundización progresiva sobre el mismo tema | 8% |

---

## 4. Distribución de Situaciones (12 situaciones, suma = 100%)

| Situación | Descripción | Porcentaje |
|---|---|---|
| `consulta_tecnica` | Duda concreta de programación, matemáticas o ciencia | 18% |
| `explicacion_concepto` | Pedir que se defina o explique un término o idea | 16% |
| `tarea_educativa` | Ejercicio o problema escolar/universitario | 12% |
| `revision_texto` | El usuario pide corregir, mejorar o resumir un fragmento | 10% |
| `solicitud_listado` | Pedir una lista, ranking o comparativa de opciones | 10% |
| `busqueda_consejo` | Consejo práctico sobre un problema personal o profesional | 8% |
| `traduccion_breve` | Traducción de frases o párrafos cortos (ES↔EN) | 6% |
| `generacion_creativa` | Pedir un relato, diálogo, poema o redacción temática | 6% |
| `depuracion_codigo` | Encontrar y corregir un error en un fragmento de código | 7% |
| `analisis_datos` | Interpretar un conjunto pequeño de datos o estadísticas | 4% |
| `debate_argumento` | Pedir argumentos a favor/en contra de una posición | 2% |
| `seguimiento_previo` | Continuación lógica de un turno anterior (sólo estilo `seguimiento_conversacional`) | 1% |

---

## 5. Cobertura de Rutas Temáticas

Los ejemplos deben distribuirse proporcionalmente a los mundos de IRIS:

| Mundo | Porcentaje de ejemplos |
|---|---|
| `informatica` (programación, algoritmos, IA) | 22% |
| `matematicas` (álgebra, cálculo, estadística, lógica) | 18% |
| `ciencias` (física, biología, química, astronomía) | 16% |
| `humanidades` (historia, filosofía, literatura, arte) | 14% |
| `salud` (medicina, nutrición, bienestar) | 10% |
| `ciencias_sociales` (economía, política, psicología) | 10% |
| `lenguaje_general` (redacción, gramática, traducción) | 10% |

---

## 6. Reglas de Calidad y Validación

### 6.1 Regla de No Deriva de Tema (`no_topic_drift`)
Un ejemplo **pasa** si y solo si:
- La respuesta del asistente responde exactamente a la pregunta o solicitud del turno humano.
- No introduce información de un mundo/categoría diferente a la indicada en `topic_route` sin que el usuario lo haya pedido.
- En conversaciones de seguimiento (`seguimiento_conversacional`), cada turno humano profundiza o clarifica el tema original; cualquier cambio de tema que no sea explícitamente solicitado por el turno humano hace que `no_topic_drift = false` y el ejemplo **se descarta**.

**Heurística de validación automática:** Calcular la similitud TF-IDF entre el primer turno humano y la respuesta final del asistente. Si la similitud cae por debajo de 0.25, marcar para revisión manual.

### 6.2 Regla de Consistencia de Idioma
- Si `language == "es"`, todos los turnos deben estar en español. Se toleran términos técnicos en inglés (nombres de funciones, bibliotecas) pero no frases completas.
- Si `language == "en"`, todos los turnos deben estar en inglés.

### 6.3 Regla de Longitud
- Respuesta final del asistente: entre **80** y **600 tokens** (estimados como palabras × 1.3).
- Turnos humanos: entre **10** y **200 tokens**.

### 6.4 Restricciones de Contenido
- Sin datos de personas reales identificables (PII).
- Sin contenido que induzca daño, violencia o discriminación.
- Sin instrucciones sobre actividades ilegales.
- Sin alucinaciones verificables: si el estilo es `respuesta_directa` o `razonamiento_cadena`, los hechos deben poder verificarse contra los datasets de entrenamiento (Wikipedia, FineWeb-Edu, C4-ES).

---

## 7. Formato de Archivo de Salida

- **Archivo único:** `IRIS_SFT_CORPUS.jsonl` (una línea JSON por ejemplo, UTF-8).
- **Ruta de entrega:** `C:/Users/angel/OpenBot/Shared/IRIS_SFT_CORPUS.jsonl`
- **Archivo de rechazos:** `IRIS_SFT_CORPUS_RECHAZADOS.jsonl` con campo adicional `rejection_reason: string`.
- **Informe de estadísticas:** `IRIS_SFT_STATS.json` con conteos por `language`, `style`, `situation`, `topic_route.mundo` y tasa de rechazo.

---

## 8. Proceso de Generación Recomendado (Z.ai)

1. Para cada ejemplo, generar con el prompt de sistema: *"Eres un asistente de IA útil y preciso. Responde SOLO la pregunta que se te hace, sin salirte del tema. Responde en [IDIOMA]."*
2. Generar el turno humano a partir de la situación + estilo + mundo elegido (como plantilla de entrada variable).
3. Aplicar las validaciones §6.1–6.4 automáticamente y descartar o regenerar los que fallen.
4. Guardar cada ejemplo en JSONL con todos los campos del §2.
