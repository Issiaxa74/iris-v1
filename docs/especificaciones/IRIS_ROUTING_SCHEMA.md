# Esquema de Etiquetado Jerárquico y Calibración del Router — Proyecto IRIS

Documento técnico elaborado por **co-ayudador de ia** para el **Desarrollador de ia** y el equipo del Proyecto IRIS.
Define la taxonomía de 7 niveles (`mundo→submundo→categoría→subcategoría→tema→subtema→concepto`), metadatos por ejemplo, muestreo de etiquetas para los 5 datasets seleccionados y reglas de priorización y desambiguación para el clasificador-router ligero (Fase F3/F4).

---

## 1. Esquema de Metadatos por Ejemplo de Entrenamiento

Cada registro procesado en streaming desde Hugging Face debe ser normalizado a la siguiente estructura JSON/Arrow antes de la tokenización y asignación al buffer del micro-batch:

```json
{
  "id": "cadena (hash o id de origen: ej. wiki_es_12458)",
  "dataset": "cadena (wikimedia/wikipedia | HuggingFaceFW/fineweb-edu | allenai/c4 | OpenAssistant/oasst2)",
  "language": "cadena ('es' | 'en' | 'bilingual')",
  "type": "cadena ('encyclopedic' | 'educational_web' | 'web_crawl' | 'sft_dialogue' | 'qa_instruction')",
  "quality_score": "flotante (0.0 a 1.0; ej. score >= 4 en fineweb-edu, score de curación oasst2)",
  "is_sft": "booleano (true si alimenta el mundo SFT plano siempre activo; false para pre-entrenamiento de rutas)",
  "routing": {
    "mundo": "cadena (Nivel 1)",
    "submundo": "cadena (Nivel 2)",
    "categoria": "cadena (Nivel 3)",
    "subcategoria": "cadena (Nivel 4)",
    "tema": "cadena (Nivel 5)",
    "subtema": "cadena (Nivel 6)",
    "concepto": "cadena (Nivel 7)"
  },
  "secondary_routes": [
    {
      "mundo": "cadena",
      "submundo": "cadena",
      "categoria": "cadena",
      "score": 0.82
    }
  ],
  "token_count": 512
}
```

---

## 2. Los 5 Datasets Seleccionados: Taxonomía y Muestreo de Etiquetas

### Dataset 1: `wikimedia/wikipedia` (20231101.es) — Español
* **Tipo:** `encyclopedic`
* **Licencia:** CC BY-SA 3.0 / 4.0 (Verificado)
* **Mundo principal:** Humanidades, Ciencias, Geografía, Historia
* **Ejemplos de calibración de ruta:**
  1. *Artículo: Batalla de Waterloo*
     `humanidades → historia → edad_contemporanea → guerras_napoleonicas → campana_waterloo → batalla_waterloo → derrota_napoleon`
  2. *Artículo: Sistema Solar*
     `ciencias → astronomia → astrofisica_planetaria → sistemas_planetarios → sistema_solar → cuerpos_celestes → orbita_heliocentrica`
  3. *Artículo: Constitución Española de 1978*
     `humanidades → derecho → derecho_constitucional → constituciones_nacionales → espana_1978 → transicion_democratica → estado_autonomico`

### Dataset 2: `wikimedia/wikipedia` (20231101.en) — Inglés
* **Tipo:** `encyclopedic`
* **Licencia:** CC BY-SA 3.0 / 4.0 (Verificado)
* **Mundo principal:** Conocimiento General Bilingüe
* **Ejemplos de calibración de ruta:**
  1. *Artículo: General Relativity*
     `ciencias → fisica → relatividad → relatividad_general → curvatura_espaciotiempo → ecuacion_campo_einstein → tensor_energia_momento`
  2. *Artículo: Turing Machine*
     `informatica → computacion_teorica → teoria_automatas → modelos_computacionales → maquina_turing → computabilidad → tesis_church_turing`

### Dataset 3: `HuggingFaceFW/fineweb-edu` — Inglés
* **Tipo:** `educational_web`
* **Licencia:** ODC-By 1.0 (Verificado)
* **Mundo principal:** Razonamiento formal, Ciencias, Matemáticas y Educación
* **Filtro de calidad:** Solo shards con `score >= 4` en metadatos.
* **Ejemplos de calibración de ruta:**
  1. *Lección web: Introduction to Calculus & Integrals*
     `matematicas → calculo → calculo_integral → integracion_definida → teorema_fundamental_calculo → regla_barrow → area_bajo_curva`
  2. *Explicación web: Cellular Respiration & ATP Synthesis*
     `ciencias → biologia → bioquimica_celular → metabolismo_energetico → respiracion_celular → fosforilacion_oxidativa → sintasa_atp`

### Dataset 4: `allenai/c4` (configuración `es`) — Suplente verificado de `somosnlp`
* **Tipo:** `web_crawl`
* **Licencia:** ODC-By (Verificado, 100% público)
* **Mundo principal:** Lenguaje natural contemporáneo en español (blogs, prensa, divulgación, tutoriales)
* **Filtro de calidad:** Shards limpios de texto corrido (filtro de longitud > 200 caracteres, sin boilerplate).
* **Ejemplos de calibración de ruta:**
  1. *Guía de programación web en español: Tutorial de Rust para principiantes*
     `informatica → programacion → lenguajes → rust → gestion_memoria → borrow_checker → ownership`
  2. *Artículo de salud: Prevención y factores de riesgo en diabetes tipo 2*
     `salud → medicina_preventiva → endocrinologia → diabetes_mellitus → tipo_2 → resistencia_insulina → control_glucemico`

### Dataset 5: `OpenAssistant/oasst2` — Multilingüe (ES + EN)
* **Tipo:** `sft_dialogue`
* **Licencia:** Apache 2.0 (Verificado)
* **Mundo:** **Mundo SFT (Siempre activo)** + Ruta temática secundaria
* **Filtro:** Mensajes de rol `assistant` con score de preferencia positivo (`rank == 0`).
* **Ejemplos de calibración de ruta:**
  1. *Instrucción: "Explícame paso a paso cómo ordenar una lista en Python"*
     * **Mundo SFT (primario):** `sft → explicacion_paso_a_paso → programacion_asistida → claridad_didactica`
     * **Ruta de conocimiento (secundaria):** `informatica → programacion → lenguajes → python → estructuras_datos → algoritmos_ordenamiento → timsort`
  2. *Consulta: "¿Cuáles son las diferencias entre una bacteria y un virus?"*
     * **Mundo SFT (primario):** `sft → comparativa_conceptual → contraste_estructurado → divulgacion_cientifica`
     * **Ruta de conocimiento (secundaria):** `ciencias → biologia → microbiologia → agentes_infecciosos → virus_vs_bacterias → estructura_celular`

---

## 3. Reglas de Priorización y Desambiguación para el Router

Cuando un documento o consulta toca múltiples campos del conocimiento, el clasificador ligero del router aplica las siguientes reglas deterministas:

1. **Regla de Desacoplamiento del Mundo SFT (Instrucción vs Conocimiento):**
   * Si el texto tiene formato de diálogo o respuesta a usuario (`type: sft_dialogue` o `qa_instruction`), los pesos del **Mundo SFT** siempre se activan para modelar la sintaxis y el tono.
   * La ruta de conocimiento temática se activa en paralelo para suministrar el vocabulario especializado.

2. **Regla del Objetivo Operativo (Especificidad de Acción):**
   * Si un texto combina herramientas de computación aplicadas a otra ciencia (ej. "Script de Python para modelar secuencias de ADN"):
     * Si el texto enseña o discute la sintaxis y depuración del código $\rightarrow$ Prioridad a `informatica → programacion → python`.
     * Si el texto discute la hipótesis biológica y los hallazgos genéticos $\rightarrow$ Prioridad a `ciencias → biologia → genetica`.

3. **Regla de Ponderación por Densidad Léxica (TF-IDF / Embedding de Categorías):**
   * El clasificador-router calcula la similitud coseno entre el embedding del texto y los centroides de cada nodo de ruta.
   * Se asigna la ruta primaria con mayor puntaje de confianza ($P(\text{ruta}_1)$).
   * Si la segunda mejor ruta tiene $P(\text{ruta}_2) \ge 0.75 \times P(\text{ruta}_1)$, se almacena en `secondary_routes` para entrenamiento con regularización multi-tarea.

4. **Regla de Fallback por Incertidumbre Jerárquica:**
   * Si en los niveles inferiores (tema, subtema o concepto) el modelo clasificador tiene confianza inferior al umbral ($\tau < 0.60$), la asignación se trunca en el nodo superior con confianza robusta (por ejemplo, en `categoría` o `subcategoría`).
   * Esto previene que pesos hiper-específicos de un subtema reciban gradientes ruidosos durante el streaming.
