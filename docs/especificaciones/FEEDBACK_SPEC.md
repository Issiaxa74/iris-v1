# Mecánica de sub-parámetros y feedback — especificación v0.1

## 1. Estrés de parámetros (cuantización p4-one en acción)

- Cada parámetro tiene 4 sub-parámetros comprimidos que **nunca se exponen** durante la inferencia normal (peso comprimido, no activo).
- Cada uso de un peso genera una **activación** (contador de uso por parámetro).
- **Estrés**: cuando un parámetro recibe muchas activaciones (umbral por ciclo), libera sus sub-parámetros **temporalmente** (se descomprimen y participan) porque necesita su conocimiento; después se vuelven a ignorar y se re-comprimen al guardar.
- Reglas:
  - Sub-parámetros activos solo bajo estrés; jamás expuestos "por si acaso" (ahorro de RAM).
  - El contador de estrés decae por ciclo; umbral configurable por ruta.
  - En entrenamiento, la activación bajo estrés es la señal que decide qué sub-parámetros se actualizan (el resto queda congelado).

## 2. Feedback del usuario por terminal (machine learning con señal humana)

Comandos aplicados a la última respuesta de Iris:

| Comando | Significado | Efecto |
|---|---|---|
| `/si [texto]` | La respuesta fue correcta (opcionalmente da la correcta) | Refuerza: la ruta y los pesos activados reciben señal positiva |
| `/no` | Respuesta equivocada | Penaliza la ruta/parámetros usados |
| `/up` | La ruta fue correcta pero la respuesta equivocada | Refuerza el ROUTER (su elección fue bien), penaliza los pesos de la ruta |
| `/ent-my archivo.txt` | Fine-tuning desde txt estructurado (preguntas y respuestas) | Auto-entrenamiento complementario tipo LoRA: añade/adapta pesos SIN reemplazar los base |

Reglas del entrenamiento con feedback:
- Núcleo factible: **next-token-prediction** con las rutas activas.
- `/si//no//up` son pares (entrada, corrección) que entran a la cola de entrenamiento del siguiente ciclo; no perturban la inferencia en vivo.
- `/ent-my` entrena un **adaptador** (LoRA-like sobre sub-parámetros liberados bajo estrés) y lo guarda como parche junto al UKJ, nunca como reemplazo del base.
- El router entrena por **contenido semántico, nunca por sonido superficial**: el clasificador no debe guiar por palabras-sonido ("estado" → Estados Unidos). Se entrena con pares y contraejemplos de desambiguación y evalúa con frases trampa incluidas en su set de validación (ej.: "¿en qué estado está el mundo?" no puede caer en estados_unidos en ningún nivel).
