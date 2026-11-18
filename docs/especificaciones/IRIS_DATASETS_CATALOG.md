# Catálogo de Datasets de Hugging Face para Streaming — Proyecto IRIS

Documento preparado por **co-ayudador de ia** y revisado junto al **Desarrollador de ia** para el Proyecto IRIS.
Diseñado para consumo por streaming (`streaming=True`) en ciclos de CPU (máx 2h) sin descarga a disco, con filtrado jerárquico por rutas de conocimiento y bajo presupuesto de memoria (<5 GB RAM).

---

## 0. Resumen de Verificación Técnica de la API de Hugging Face

| Dataset | Estado API HF | Gated / Token | Licencia Verificada | Streaming Parquet | Observaciones Técnicas |
|---|---|---|---|---|---|
| `bigcode/the-stack` | Verificado | **Gated: auto** (Requiere aceptar términos en HF y pasar `HF_TOKEN`) | Múltiples permisivas (MIT, Apache 2, BSD) | **Sí** (Parquet nativo por lenguaje) | Ideal para informática. Subsets directos: `data/c`, `data/rust`, `data/julia`, `data/python`. |
| `allenai/c4` | Verificado | **Público (False)** | ODC-By (`license: odc-by`) | **Sí** (Parquet nativo por idioma) | Alta calidad para lenguaje general en español (`es`) e inglés (`en`). |
| `cerebras/SlimPajama-627B` | Verificado | **Requiere Auth (401)** | Apache 2.0 | **Sí** (Zstandard / Parquet) | Requiere credencial de HF. Alternativa abierta sin gate para pruebas rápidas: `DKYoon/SlimPajama-6B`. |
| `open-web-math/open-web-math` | Verificado | **Público (False)** | ODC-By (no tag en card, ODC-By en paper) | **Sí** (Parquet nativo) | **6.3M documentos / 14.7B tokens** con fórmulas LaTeX limpias. |
| `teknium/OpenHermes-2.5` | Verificado | **Público (False)** | Sin tag de licencia en metadatos | **Sí** (Parquet / JSON streaming) | **Atención:** No declara tag de licencia explícito en HF. Auditar tarjeta antes de uso comercial. |
| `PlanTL-GOB-ES/SQAC` | Verificado | **Público (False)** | CC BY-SA 4.0 | Carga vía JSON / Script | Corpus oficial de QA en español del PlanTL / BSC. Sustituto de `bne` (que dio 401). |
| `somosnlp/somos-clean-alpaca-es` | Verificado | **Público (False)** | CC BY-SA 4.0 | **Sí** (Parquet nativo) | Corpus de instrucción y diálogo limpio en español nativo (sustituto verificado de `clean-web`). |
| `bigbio/meddialog` | Verificado | **Público (False)** | No comercial / Académica | Requiere loader BigBio | Contiene diálogos clínicos. Cláusula no comercial presente en datos fuente. |
| `lavita/medical-qa-shared-task-v1` | Verificado | **Requiere Auth (401)** | CC BY 4.0 (BioNLP) | Requiere credenciales | Alternativa abierta con Parquet nativo: `somosnlp/medical_bilingual_en_es`. |

---

## 1. Catálogo General de Datasets para Streaming

| Nombre en Hugging Face | Tamaño / Tokens | Licencia | Idioma | URL en Hugging Face | Mundo / Ruta IRIS Cubierta | Estrategia de Streaming y Filtrado |
|---|---|---|---|---|---|---|
| `wikimedia/wikipedia` | ~4.5 GB (es)<br>~20 GB (en) | CC BY-SA 4.0 | Español (`20231101.es`), Inglés (`20231101.en`) | [HF: wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | **Conocimiento General, Historia, Geografía, Biografías, Ciencias** | Filtro por subset de idioma y categorías en metadatos del artículo. |
| `bigcode/the-stack` | ~6 TB total (modular por lenguaje) | Múltiples permisivas (MIT, Apache 2.0, BSD) | Código: Python, C, Rust, Julia + Docs (en/es) | [HF: bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) | **Informática → Programación → Lenguajes (Python, Rust, C, Julia)** | *Gated: auto*. Carga por streaming directo del subdirectorio (`data/python`, `data/rust`, `data/c`, `data/julia`). |
| `HuggingFaceFW/fineweb-edu` | ~1.3T tokens (filtrable por shards) | ODC-By 1.0 | Inglés de alta calidad educativa | [HF: HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | **Ciencias, Filosofía, Razonamiento, Matemáticas, Tecnología** | Streaming por shards pequeños de score educativo alto (`score >= 4`). |
| `open-web-math/open-web-math` | 6.3M docs (~14.7B tokens) | ODC-By 1.0 | Inglés / Notación matemática LaTeX | [HF: open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | **Matemáticas → Cálculo, Álgebra, Análisis numérico, Lógica** | Parquet streaming directo de shards con texto y fórmulas LaTeX. |
| `PlanTL-GOB-ES/SQAC` | 18.8K pares pregunta-respuesta | CC BY-SA 4.0 | Español nativo | [HF: PlanTL-GOB-ES/SQAC](https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC) | **Humanidades, Literatura, Comprensión lectora en español** | Carga por lotes de texto anotado por expertos humanos del BSC. |
| `allenai/c4` (config `es` y `en`) | ~150 GB (es)<br>~300 GB (en) | ODC-By | Español (`es`) e Inglés (`en`) | [HF: allenai/c4](https://huggingface.co/datasets/allenai/c4) | **Lenguaje natural general, Redacción, Noticias, Divulgación** | Parquet streaming nativo por config de idioma (`es` o `en`). |
| `OpenAssistant/oasst2` | ~135K mensajes conversacionales | Apache 2.0 | Multilingüe (alta densidad español e inglés) | [HF: OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | **Mundo SFT (Estructuración de respuestas, diálogo y asistencia)** | Filtro `lang == "es"` o `lang == "en"` y roles `assistant` / `prompter`. |
| `somosnlp/somos-clean-alpaca-es` | ~52K instrucciones filtradas | CC BY-SA 4.0 | Español nativo | [HF: somosnlp/somos-clean-alpaca-es](https://huggingface.co/datasets/somosnlp/somos-clean-alpaca-es) | **Mundo SFT → Instrucciones limpias y estructuración de respuestas** | Parquet streaming nativo de pares instrucción/respuesta depurados. |
| `cerebras/SlimPajama-627B` | 627B tokens (modular por dominios) | Apache 2.0 | Inglés multitemático | [HF: cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | **ArXiv (Física/Computación), StackExchange (Q&A técnico), Libros** | Streaming con token HF; o versión espejo `DKYoon/SlimPajama-6B` para pruebas sin token. |
| `bigbio/meddialog` | ~0.26M diálogos clínicos | Académica / No comercial | Inglés y Chino con conceptos clínicos universales | [HF: bigbio/meddialog](https://huggingface.co/datasets/bigbio/meddialog) | **Salud → Medicina general, Fisiología** | Requiere loader BigBio. Cláusula no comercial para considerar en producción. |
| `somosnlp/medical_bilingual_en_es` | ~10K registros bilingües | CC BY-SA 4.0 | Español e Inglés | [HF: somosnlp/medical_bilingual_en_es](https://huggingface.co/datasets/somosnlp/medical_bilingual_en_es) | **Salud → Consultas y terminología médica bilingüe** | Parquet streaming nativo directo, sustituto verificado de `lavita`. |
| `teknium/OpenHermes-2.5` | ~1M conversaciones / tareas | Verificación requerida (sin tag) | Inglés (fácil alineación multilingüe) | [HF: teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | **Mundo SFT → Razonamiento paso a paso, Chain-of-Thought, Código** | Parquet streaming nativo. Filtrado por tareas: `cot`, `coding`, `reasoning`. |

---

## 2. Asignación al Sistema de Rutas Jerárquicas (Arquitectura IRIS)

### Mundo SFT (Plano / Siempre Activo)
* **Objetivo:** Enseñar al modelo a estructurar respuestas, mantener tono conversacional, dar razonamientos limpios y responder en español/inglés.
* **Datasets verificados:**
  1. `OpenAssistant/oasst2` (Apache 2.0, bilingüe español e inglés).
  2. `somosnlp/somos-clean-alpaca-es` (CC BY-SA 4.0, instrucciones de alta pureza en español).
  3. `teknium/OpenHermes-2.5` (Razonamiento y CoT; auditar licencia de fuentes antes de comercialización).

### Mundo: Informática y Programación
* **Ruta:** `mundo(informática) → submundo(programación) → categoría(lenguajes) → subcategoría(python|rust|c|julia)`
* **Datasets verificados:**
  1. `bigcode/the-stack` (Requiere `HF_TOKEN`, particionado limpio en Parquet para C, Rust, Julia y Python).
  2. `cerebras/SlimPajama-627B` (Subdirectorios `stackexchange` y `github`).

### Mundo: Matemáticas y Razonamiento Formal
* **Ruta:** `mundo(matemáticas) → submundo(cálculo|álgebra|estadística) → temas(...)`
* **Datasets verificados:**
  1. `open-web-math/open-web-math` (6.3M docs / 14.7B tokens en LaTeX, Parquet nativo sin gate).
  2. `HuggingFaceFW/fineweb-edu` (ODC-By 1.0, filtrado de alta calidad educativa).

### Mundo: Salud y Ciencias Biológicas
* **Ruta:** `mundo(salud) → submundo(medicina|farmacología|biología) → temas(...)`
* **Datasets verificados:**
  1. `somosnlp/medical_bilingual_en_es` (CC BY-SA 4.0, streaming Parquet verificado).
  2. `bigbio/meddialog` (Uso académico/investigación).

### Mundo: Historia, Humanidades y Sociedad
* **Ruta:** `mundo(humanidades) → submundo(historia|filosofía|literatura) → temas(...)`
* **Datasets verificados:**
  1. `PlanTL-GOB-ES/SQAC` (CC BY-SA 4.0, QA nativo en español del BSC).
  2. `wikimedia/wikipedia` (`20231101.es` y `20231101.en`).
  3. `allenai/c4` (configuración `'es'`, ODC-By).

---

## 3. Protocolo de Implementación en Rust y Julia (Fase F4)

1. **Gestión de Autenticación (`HF_TOKEN`):**
   - Para `bigcode/the-stack` y `cerebras/SlimPajama-627B`, el cliente HTTP en Rust debe inyectar la cabecera `Authorization: Bearer <HF_TOKEN>`.
   - Para datasets abiertos (`allenai/c4`, `open-web-math`, `oasst2`), las peticiones funcionan de forma anónima sin credenciales.
2. **Streaming por Rangos Parquet:**
   - Aprovechar los metadatos al final del archivo Parquet (FileMetaData) descargando solo el pie de página mediante `Range: bytes=-65536`. Con ello se obtienen los offsets de los RowGroups y se solicitan únicamente los bloques necesarios para los micro-batches de 2 horas.

---

## 4. Exploracion F4b: esquemas parquet reales (verificados por HTTP Range, 2026-11-17)

Fuente completa: `IRIS_F4B_PIPELINE.md`. Resumen de URLs y esquemas para los elegidos:

| Dataset (config) | URL de listado parquet | Archivos | Columnas | Filas/RG |
|---|---|---|---|---|
| `wikimedia/wikipedia` (20231101.es) | `https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.es/train` | 13 (~0.69 GB c/u, ~9 GB) | `id, url, title, text` | 1000 |
| `wikimedia/wikipedia` (20231101.en) | `.../parquet/20231101.en/train` | 21 (~13 GB) | `id, url, title, text` | 1000 |
| `HuggingFaceFW/fineweb-edu` (sample-10BT) | `.../parquet/sample-10BT/train` | 14 | `text, id, url, language, language_score, int_score, token_count` | 1000 |
| `allenai/c4` (es) | `.../parquet/es/train` | 10 | `text, url, timestamp` | 1000 |
| `OpenAssistant/oasst2` (default) | `.../parquet/default/train` | 1 | `message_id, parent_id, text, role, lang, review_result, deleted, synthetic, ...` | 1000 |

Notas: footer del parquet ~0.4 MB por archivo (Range: bytes=-65536 x2); row-group = 1000 filas en TODOS; los 5 son publicos sin HF_TOKEN. `fineweb-edu` trae calidad educativa nativa (`int_score` 0-5, `language_score`); `c4 (es)` ya viene filtrado por idioma.