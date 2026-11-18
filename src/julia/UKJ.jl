# UKJ.jl — Escritor del formato UKJ v0.1
# Uso: include("UKJ.jl"); UKJ.write_ukj("test.ukj", "p4-one", rutas)

module UKJ

export Ruta, write_ukj

struct Ruta
    id::UInt32
    niveles::Vector{String}
    data::Vector{Float32}
    dim::UInt32
end

const MAGIC = UInt8[0x55, 0x4B, 0x4A, 0x00]  # "UKJ\0"
const VERSION = UInt32(1)

# suma de bytes como checksum provisional (v0.2 -> crc64 real)
mutable struct Sink
    io::IOStream
    sum::UInt64
end

function _write(s::Sink, b::AbstractVector{UInt8})
    write(s.io, b)
    s.sum += sum(b) % UInt64
    nothing
end

_write(s::Sink, x::Unsigned) = _write(s, reinterpret(UInt8, [x]))

function _table_size(r::Ruta)
    sz = UInt64(4 + 2 + 8 + 8 + 4)  # id, n_niveles, offset, nbytes, dim
    for n in r.niveles
        sz += UInt64(2 + length(Vector{UInt8}(n)))
    end
    sz
end

"""
    write_ukj(path, quant, rutas)

Escribe un archivo UKJ: cabecera (magic, versión, cuantización, nº de rutas, flags),
tabla de rutas (id, niveles jerárquicos, offset, nbytes, dim), tensores Float32
little-endian por ruta y footer (offset de tabla + checksum).
"""
function write_ukj(path::AbstractString, quant::AbstractString, rutas::Vector{Ruta})
    open(path, "w") do io
        s = Sink(io, UInt64(0))
        qb = Vector{UInt8}(quant)
        @assert 1 <= length(qb) <= 255 "cuantización demasiado larga"

        _write(s, MAGIC)
        _write(s, VERSION)
        _write(s, UInt8(length(qb)))
        _write(s, qb)
        _write(s, UInt32(length(rutas)))
        _write(s, UInt32(0))  # flags: bit0 = tokenizer dentro del archivo (0 = separado)

        tabla_offset = 4 + 4 + 1 + length(qb) + 4 + 4
        tabla_size = sum(_table_size, rutas; init = UInt64(0))
        offset = tabla_offset + tabla_size

        for r in rutas
            _write(s, r.id)
            _write(s, UInt16(length(r.niveles)))
            for n in r.niveles
                nb = Vector{UInt8}(n)
                _write(s, UInt16(length(nb)))
                _write(s, nb)
            end
            nbytes = UInt64(length(r.data)) * 4
            @assert length(r.data) == r.dim "dim no coincide con nº de elementos"
            _write(s, UInt64(offset))
            _write(s, nbytes)
            _write(s, r.dim)
            offset += nbytes
        end

        for r in rutas
            _write(s, reinterpret(UInt8, r.data))
        end

        _write(s, UInt64(tabla_offset))  # footer: dónde está la tabla
        _write(s, s.sum)                 # checksum v0.1 (suma de bytes)
    end
    path
end

end # module
