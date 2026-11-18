# TOJI — memoria viva

TOJI no es conocimiento: es la memoria de la conversación. UKJ guarda lo que Iris sabe del mundo; TOJI guarda lo que le ha pasado con el usuario.

## 1. Elementos

| Elemento | Qué guarda |
|---|---|
| **Neurona** | Una palabra o unidad. Es el átomo de memoria. |
| **Satélite** | Recuerdos de esa neurona: eventos y emociones. Ej.: neurona «suma» + satélite «el usuario preguntó 2+2, respondí 4». |
| **Neurona anormal** | Neurona grande: base de datos del concepto según el usuario. Contiene neuronas similares y el matiz que las diferencia. Ej.: «poseer» contiene «tener» y anota la diferencia. |
| **Sector** | Barrio de neuronas por tema (`programacion`, `salud`, `general`). Equivale a los «mundos», pero para la memoria vivida. |
| **Cables** | Conexiones entre neuronas de cualquier sector: causa, similitud, oposición, secuencia. |

## 2. Creación instintiva

El modelo no crea neuronas con código explícito por respuesta: se crean durante el **pensamiento profundo** previo a responder (la fase interna de razonamiento). Si algo sorprende o marca, se registra sin decisión consciente. Las fronteras entre sectores generan relación; los cables cruzan sectores libremente.

## 3. Persistencia: archivos `.mem`

Un archivo binario por sector:

```
magic:    "TOJI\0"
version:  u32
sector:   utf8
n_neuronas, n_anormales, n_cables: u32
neuronas:  [id u32, id_str utf8, n_satelites u32,
            satelites { t_epoch u64, tipo u8 (evento|emocion), fuerza f32, texto utf8 }]
anormales: [id u32, concepto utf8, hijos [id u32], matiz utf8]
cables:    [a u32, b u32, tipo u8, peso f32]
fronteras: [sector_a utf8, sector_b utf8, peso f32]
footer:    checksum
```

El estado completo se mantiene cargado en el runtime y se actualiza en cada respuesta **sin gastar tokens de contexto**: no entra en el prompt, es estado interno. La activación es selectiva, igual que UKJ: al procesar la entrada se activan las neuronas relevantes y el resto duerme en disco. Objetivo: abrir y actualizar un `.mem` en <100 ms.

## 4. Ciclo de una respuesta

1. Entra la pregunta → el router elige ruta UKJ y se activan los sectores `.mem` relevantes.
2. Pensamiento profundo: razona con conocimiento y memoria; durante el proceso crea y actualiza neuronas, satélites, cables y fronteras.
3. Responde.
4. Al cerrar el turno se hace flush de los `.mem` tocados (append-only con compactación diferida).

## 5. Estado

El formato está implementado de forma base en `runtime/src/toji/` (memoria, volcado y tests). Falta la creación automática durante el pensamiento profundo y la compactación. La fase F5 del plan cubre el resto.
