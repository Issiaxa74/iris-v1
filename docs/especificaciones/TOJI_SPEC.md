# TOJI — Sistema de memoria de Iris (especificación v0.1)

TOJI es la memoria de Iris: guarda **todo lo que le dice el usuario** y lo que pasa en la conversación. NO es el conocimiento (eso son las rutas/UKJ); es la memoria viva.

## 1. Elementos

| Elemento | Qué es |
|---|---|
| **Neurona** | Guarda una palabra/unidad. Es el átomo de memoria. |
| **Satélite** | Recuerdos de esa neurona: eventos y emociones. Ej.: neurona "suma" + satélite "el usuario preguntó 2+2, yo respondí 4". |
| **Neurona anormal** | Neurona grande: base de datos del concepto de una palabra **según el usuario**. Dentro tiene neuronas normales con palabras similares y la diferencia entre ellas. Ej.: "poseer" contiene "tener" y apunta el matiz de diferencia. |
| **Sector** | Barrio de neuronas por tema: si hay muchas neuronas de programación se crea el sector `programacion`. Equivale a los "mundos", pero para la memoria vivida. |
| **Cables** | Conexiones complejas entre neuronas, sin importar el sector. Definen relaciones (causa, similitud, oposición, secuencia…). |

## 2. Creación instintiva

- El modelo **no** crea neuronas/satélites con código explícito por respuesta: se crean **automáticamente durante el pensamiento profundo antes de responder** (la fase interna de razonamiento de Iris).
- Es instintivo: si algo sorprende o marca, se registra sin decisión consciente.
- Fronteras entre sectores: generan relación (quién limita con quién). Los cables cruzan sectores libremente.

## 3. Persistencia: archivos .mem

- Todo el estado de neuronas/satélites/cables va a **archivos binarios `.mem`**, un archivo por sector: `programacion.mem`, `salud.mem`, `general.mem`…
- El modelo los lee "como su idioma natal": el estado completo está cargado y actualizado en cada respuesta, **sin gastar tokens de lectura** (no entra en el contexto del prompt; es estado interno del runtime).
- Activación por parámetros: al procesar la entrada se activan las neuronas/satélites relevantes; el resto queda dormido en disco. (Mismo principio de carga selectiva que UKJ.)

### Borrador de formato .mem v0.1
```
magic:    "TOJI\0"
version:  u32
sector:   utf8 (nombre)
n_neuronas, n_anormales, n_cables: u32
neuronas:  [id u32, id_str utf8, n_satelites u32, satelites { t_epoch u64, tipo u8 (evento|emocion), fuerza f32, texto utf8 }]
anormales: [id u32, concepto utf8, hijos [id u32], matiz utf8]
cables:    [a u32, b u32, tipo u8, peso f32]
fronteras: [sector_a utf8, sector_b utf8, peso f32]
footer:    checksum
```
(gzip/zstd opcional sobre el binario; el tamaño debe permitir abrir/actualizar un .mem en <100 ms para sectores de tamaño razonable.)

## 4. Ciclo de una respuesta (integración con el modelo)

1. Entra la pregunta → el router elige ruta UKJ (conocimiento) + se activan sectores .mem relevantes (memoria).
2. Pensamiento profundo: razona con conocimiento + memoria; durante ese proceso se **crean/actualizan neuronas, satélites, cables y fronteras** (instintivo).
3. Responde.
4. Al final del turno se hace flush de los .mem tocados (append-only con compactación diferida).

## 5. Relación con el resto de Iris

- UKJ = conocimiento del mundo (pre-entrenado, casi estático por rutas).
- TOJI = memoria vivida del usuario (crece con cada conversación, por sector).
- El "pensamiento profundo" es configurable (nº de pasadas) y es el motor que toca ambos sistemas.
