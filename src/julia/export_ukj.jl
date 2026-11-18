# export_ukj.jl - convierte un checkpoint IRISMDL a UKJ v0.1 + manifiesto JSON.
#
# UKJ v0.1 guarda, por ruta, un blob de Float32 y un `dim` (numero de floats).
# No guarda nombres ni formas: eso va en el manifiesto `<salida>.json` que este
# script escribe al lado. Cualquier lector (runtime/ en Rust, src/c/ukj.c) puede
# reconstruir los tensores leyendo el manifiesto.
#
# Uso:
#   julia src/julia/export_ukj.jl [checkpoint.irmdl] [salida.ukj] [quant]
#   IRIS_CKPT=models/model.irmdl julia src/julia/export_ukj.jl
#
# Salida por defecto: models/model.ukj (y models/model.ukj.json)

using Printf
include(joinpath(@__DIR__, "model.jl")); using .IrisModel
include(joinpath(@__DIR__, "UKJ.jl"));   using .UKJ

ckpt = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "IRIS_CKPT", normpath(@__DIR__, "..", "..", "models", "model.irmdl"))
salida = length(ARGS) >= 2 ? ARGS[2] : splitext(ckpt)[1] * ".ukj"
quant = length(ARGS) >= 3 ? ARGS[3] : "p4-one"

isfile(ckpt) || error("no existe el checkpoint: $ckpt")

println("cargando $ckpt ...")
m = load_model(ckpt)
cfg = m.cfg
@printf("modelo: dim %d | %d capas | %d cabezas | ffn %d | vocab %d | ctx %d | %d hojas\n",
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ffn, cfg.vocab, cfg.ctx, cfg.n_hojas)

# ------------------------- agrupacion en rutas -------------------------
# Ruta 1  = trunk: emb + fin.ln + t1..tN  (el modelo que hoy corre el forward)
# Rutas 2..1+n_hojas = una por hoja (e1..eN), congeladas hoy
# Ruta final = mundo sft
function grupo_de(nombre::String)
    nombre == "emb" && return 1
    startswith(nombre, "fin.") && return 1
    startswith(nombre, "t") && return 1
    startswith(nombre, "e") && return 1 + parse(Int, match(r"^e(\d+)\.", nombre).captures[1])
    startswith(nombre, "sft.") && return 2 + cfg.n_hojas
    error("tensor sin grupo: $nombre")
end

n_rutas = 2 + cfg.n_hojas
tensores = [Tuple{String,UnitRange{Int}}[] for _ in 1:n_rutas]
for nombre in m.names
    push!(tensores[grupo_de(nombre)], (nombre, m.ranges[nombre]))
end

rutas = UKJ.Ruta[]
manifiesto = String[]
push!(manifiesto, "{")
push!(manifiesto, "  \"formato\": \"UKJ v0.1\",")
push!(manifiesto, "  \"quant\": \"$quant\",")
push!(manifiesto, @sprintf("  \"n_params\": %d,", length(m.p)))
push!(manifiesto, @sprintf("  \"config\": {\"dim\": %d, \"n_layers\": %d, \"n_heads\": %d, \"ffn\": %d, \"vocab\": %d, \"ctx\": %d, \"n_hojas\": %d},",
                           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ffn, cfg.vocab, cfg.ctx, cfg.n_hojas))
push!(manifiesto, "  \"rutas\": [")

for (r, ts) in enumerate(tensores)
    isempty(ts) && continue
    datos = Float32[]
    for (_, rg) in ts
        append!(datos, view(m.p, rg))
    end
    nivel = r == 1 ? "trunk" : r == 2 + cfg.n_hojas ? "sft" : "hoja"
    push!(rutas, UKJ.Ruta(UInt32(r), [nivel], datos, UInt32(length(datos))))

    push!(manifiesto, @sprintf("    {\"id\": %d, \"nivel\": \"%s\", \"n_floats\": %d, \"tensores\": [", r, nivel, length(datos)))
    off = 0
    for (k, (nombre, rg)) in enumerate(ts)
        push!(manifiesto, @sprintf("      {\"nombre\": \"%s\", \"offset\": %d, \"n\": %d}%s",
                                   nombre, off, length(rg), k == length(ts) ? "" : ","))
        off += length(rg)
    end
    push!(manifiesto, @sprintf("    ]}%s", r == length(tensores) ? "" : ","))
end
push!(manifiesto, "  ]")
push!(manifiesto, "}")

# ------------------------------ escritura ------------------------------
write_ukj(salida, quant, rutas)
write(salida * ".json", join(manifiesto, "\n") * "\n")

@printf("escrito %s (%.1f MB) y %s.json (%d rutas)\n",
        salida, filesize(salida) / 1e6, salida, length(rutas))


# --------------------------- auto-verificacion -------------------------
# Relee el UKJ, parsea la tabla y compara los bytes de cada ruta con el origen.
function verificar_ukj(salida, rutas)
    raw = read(salida)
    @assert raw[1:4] == UInt8[0x55, 0x4B, 0x4A, 0x00] "magic UKJ incorrecto"
    pos = 9
    qlen = Int(raw[pos]); pos += 1 + qlen
    n = Int(reinterpret(UInt32, raw[pos:pos+3])[1]); pos += 8   # n_rutas + flags
    @assert n == length(rutas) "n_rutas no coincide"

    leidos = Dict{Int,UnitRange{Int}}()
    for _ in 1:n
        id = Int(reinterpret(UInt32, raw[pos:pos+3])[1]); pos += 4
        nn = Int(reinterpret(UInt16, raw[pos:pos+1])[1]); pos += 2
        for _ in 1:nn
            ln = Int(reinterpret(UInt16, raw[pos:pos+1])[1]); pos += 2 + ln
        end
        off = Int(reinterpret(UInt64, raw[pos:pos+7])[1]); pos += 8
        nb = Int(reinterpret(UInt64, raw[pos:pos+7])[1]); pos += 8
        pos += 4
        leidos[id] = (off + 1):(off + nb)   # offset 0-based en el archivo, arrays 1-based
    end

    for r in rutas
        rg = leidos[Int(r.id)]
        @assert length(rg) == length(r.data) * 4 "tamaño de ruta $(r.id)"
        @assert @view(raw[rg]) == reinterpret(UInt8, r.data) "bytes de ruta $(r.id)"
    end
    return n
end

n = verificar_ukj(salida, rutas)
@printf("verificacion: OK (%d rutas, bytes identicos al checkpoint)\n", n)
