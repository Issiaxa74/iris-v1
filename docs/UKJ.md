# UKJ — formato de modelo v0.1

UKJ es el contenedor propio de Iris, equivalente a GGUF o safetensors, diseñado para cargar **solo la ruta que se usa** en lugar del modelo entero.

## 1. Idea

Un parámetro puede tener N sub-parámetros comprimidos que no se activan en inferencia normal. La cuantización de UKJ se nombra por ese número: `p4-one` = 1 parámetro con 4 sub-parámetros. Nivel máximo previsto: `p300-one`.

En inferencia, los sub-parámetros duermen. Bajo **estrés** (muchas activaciones de un peso) se descomprimen temporalmente, participan en el cálculo, y se vuelven a comprimir al cerrar el ciclo. El contador de estrés decae por ciclo y el umbral es configurable por ruta.

En esta versión el modelo entrena en fp32 plano: la compresión solo existe al serializar.

## 2. Estructura binaria (little-endian)

```
[0..3]    magic        "UKJ\0"
[4..7]    version      u32 (1)
[8]       quant_len    u8
[9..]     quant        utf8 ("p4-one")
[..]      n_rutas      u32
[..]      flags        u32 (bit0: tokenizer dentro del archivo; 0 = separado)
[..]      tabla de rutas, n_rutas entradas:
              u32 id_ruta
              u16 n_niveles
              por nivel: u16 len + utf8 (jerarquía, p. ej. "informatica", "programacion")
              u64 offset
              u64 nbytes
              u32 dim          (número de floats del blob)
[..]      tensores     Float32, en el orden de la tabla
[final]   u64 offset_tabla
          u64 checksum    (suma de bytes; v0.2 prevé crc64)
```

- **Tokenizer fuera** del UKJ por defecto: `iris.tok` va como archivo aparte (`flags = 0`).
- **Sin nombres ni formas**: UKJ v0.1 guarda un blob por ruta. El manifiesto `*.ukj.json` que genera `export_ukj.jl` guarda nombre, offset y tamaño de cada tensor dentro de su ruta.
- **mmap parcial**: con la tabla se abre el archivo y se mapea solo el rango de la ruta activa. Es el requisito para el límite de ~1 GB de RAM en inferencia.

## 3. Implementaciones

| Lenguaje | Archivo | Uso |
|---|---|---|
| Julia | `src/julia/UKJ.jl` | escritor (`write_ukj`) |
| Rust | `src/rust/ukj_reader/` | lector de referencia, valida tabla y checksum |
| C | `src/c/ukj.c` | carga mínima para el núcleo de tensores |
| Rust | `runtime/src/ukj/mod.rs` | lector del runtime con mmap y modo estrés |

`src/julia/test_writer.jl` escribe un UKJ de 3 rutas y el lector Rust debe devolver los mismos tensores byte a byte. Ese es el criterio de aceptación de la fase F1.

## 4. Exportar el modelo real

```powershell
julia src/julia/export_ukj.jl models/model.irmdl models/model.ukj p4-one
```

Agrupa los tensores en 18 rutas: `trunk` (embeddings + 14 capas + LN final), `hoja` ×16 y `sft`. Tras escribir, relee el archivo, parsea la tabla y compara cada ruta byte a byte con el checkpoint.

## 5. Lo que falta

- CRC64 real en el footer (hoy suma de bytes).
- Sub-parámetros comprimidos de verdad (hoy `p4-one` es una etiqueta).
- Alineación a 64 bytes de cada tensor (la tabla ya lo reserva, el escritor aún no la aplica).
- Escritor que acepte tensores dispersos por nombre sin concatenar.
