# UKJ — Especificación v0.1 (borrador de trabajo)

UKJ es el formato contenedor de modelos de Iris, equivalente propio a GGUF/safetensors.

## 1. Concepto clave

- Cada **parámetro** contiene N **sub-parámetros** (comprimidos, no activos por defecto).
- La "cuantización" de UKJ = cuántos sub-parámetros tiene cada parámetro: `p4-one` = 1 parámetro ("one") con 4 sub-parámetros. Nivel máximo previsto: `p300-one`.
- Comportamiento en inferencia: los sub-parámetros no se activan en una respuesta normal; al liberar tensión algunos se activan y **convergen como parámetros normales**, y se vuelven a guardar comprimidos al cerrar el ciclo.

## 2. Estructura binaria (little-endian)

```
[0..3]   magic       "UKJ\0"
[4..7]   version     u32 (1)
[8..11]  quant       u8 len + utf8 ("p4-one")
[12..15] n_rutas     u32
[16..19] flags       u32 (bit0: tokenizer dentro del archivo; default 0 = archivos separados)
[20..]   tabla_rutas n_rutas × { u32 id_ruta, u16 n_niveles, niveles utf8, u64 offset, u64 nbytes, dim tensor }
[..]     tensores    por ruta, alineados a 64 bytes
[final]  footer      crc64 de todo + offset de tabla
```

- **Tokenizer y vocabulario**: archivos separados (`iris.tok`), nunca dentro del UKJ por defecto.
- **Compresión**: la más rápida e inteligente → por defecto Zstandard (ratio/velocidad alto), probando XZ solo en mundos fríos.
- **mmap parcial**: la tabla de rutas permite abrir el archivo y cargar en RAM solo el offset de la ruta activa. Requisito para el límite de 5 GB.

## 3. Implementación por fase

1. **F1a — escritor UKJ en Julia**: dado un modelo entrenado por rutas, escribe el archivo.
2. **F1b — lector UKJ en Rust**: abre, lee tabla, mapea solo la ruta pedida.
3. **F1c — leedor mínimo en C**: verificación y carga para el núcleo de tensores.
4. F1 se considera hecho cuando: Julia escribe un UKJ de prueba con 3 rutas y el lector Rust devuelve los mismos tensores byte a byte.

## 4. Pendiente de decidir (con el usuario)

- DECIDIDO (2026-11-17): historial mixto — últimos 2–4 turnos en tokens + volcado a TOJI. Ver IRIS_MODEL_SPEC.md.
- DECIDIDO (2026-11-17): preentrenamiento entrena base+sub; luego base congelada, ajuste por sub-parámetros y graduación de bloques calientes.
- DECIDIDO (2026-11-17): escalado por fases, trunk 1024×14, 16→32 hojas (sustituye al borrador dim 512).
