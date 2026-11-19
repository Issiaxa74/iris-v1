# f4b_cycle.jl — Iris F4b: primer ciclo real: buffer (jsonl.gz) -> BPE -> trunk
# Uso: julia f4b_cycle.jl [minutos=120] [ckpt=../model_f4b.irmdl] [cursor=f4b_cursor.json]
#
# Consume el buffer del co-ayudador (C:/Users/angel/OpenBot/iris/datos/buffer):
# chunks <ds>_<N>.jsonl.gz de 20k filas, FIFO por dataset, campo "text".
# Tokeniza con el BPE propio (iris.tok, vocab 1056; ids 1..1056: encode es 0-based,
# el modelo indexa emb con tok-1 -> sumar +1). Packing a T=256, micro-batch B=2
# (patron verificado de f4a_batch_test.jl). Mezcla por pesos del plan
# (c4 30%, fineweb 25%, wiki ES/EN 15+15%, oasst2 15%): cada paso va al dataset
# con menor (servidos/peso) entre los que tienen tokens listos.
# Cursor plano por dataset (chunk, linea cruda, tokens leidos) en JSON con valores
# string (parseable con jobj de router_extractor.jl); f.line cuenta TODAS las
# lineas crudas del chunk, asi el skip del relance cuadra siempre.
# Checkpoint del modelo cada 15 min; cursor cada 5. Borrado del chunk SOLO al
# terminarlo (contrato IRIS_F4B_PIPELINE.md). Depende de CodecZlib (solo pipeline;
# el nucleo del modelo sigue sin dependencias externas).

using Printf
using CodecZlib
include(joinpath(@__DIR__, "model.jl")); using .IrisModel
include(joinpath(@__DIR__, "BPE.jl"));   using .BPE
include(joinpath(@__DIR__, "router_extractor.jl"))   # jobj

const BUFFER = get(ENV, "IRIS_BUFFER", normpath(@__DIR__, "..", "..", "datos", "buffer"))
const PESOS = Dict("c4_es" => 0.30, "fineweb_edu" => 0.25,
                   "wikipedia_es" => 0.15, "wikipedia_en" => 0.15, "oasst2" => 0.15)
const T = 256          # packing
const B = 2            # micro-batch (punto dulce F4a)
const W = T * B        # tokens por paso de entrenamiento
const LINEAS_POR_REFILL = 32
const CKPT_MIN = 15.0
const CURSOR_MIN = 5.0

mutable struct Fuente
    ds::String
    w::Float64
    chunk::Int
    line::Int                # lineas CRUDAS ya consumidas del chunk actual (cursor)
    it::Any                  # iterator eachline del stream abierto (o nothing)
    io::Any; gz::Any
    buf::Vector{Int}         # tokens pendientes de entrenar
    servidos::Float64        # tokens entrenados en ESTA corrida
    tokens::Int              # tokens leidos historicos (cursor)
    bytes::Int               # bytes de texto leidos en ESTA corrida
    acabado::Bool
end

function chunkpath(f::Fuente, n::Int)
    return joinpath(BUFFER, f.ds, string(f.ds, "_", lpad(n, 6, "0"), ".jsonl.gz"))
end

function cerrar!(f::Fuente)
    if f.gz !== nothing
        try close(f.gz) catch end
        f.gz = nothing
    end
    if f.io !== nothing
        try close(f.io) catch end
        f.io = nothing
    end
    f.it = nothing
end

"""Abre el chunk actual y salta las f.line lineas crudas ya consumidas."""
function abrir!(f::Fuente)
    cerrar!(f)
    path = chunkpath(f, f.chunk)
    isfile(path) || (f.acabado = true; return)
    f.io = open(path)
    f.gz = GzipDecompressorStream(f.io)
    f.it = eachline(f.gz)
    for _ in 1:f.line
        iterate(f.it) === nothing && break
    end
    f.acabado = false
end

"""Lee hasta K lineas utiles mas del chunk actual. Al agotarse el chunk: lo
cierra, lo BORRA (contrato: solo chunks terminados) y avanza al siguiente."""
function refill!(f::Fuente, K::Int, tok)
    if f.it === nothing
        abrir!(f)
        f.acabado && return 0
    end
    leidas = 0
    terminado = false
    while leidas < K
        r = iterate(f.it)
        r === nothing && (terminado = true; break)
        f.line += 1                     # toda linea cruda cuenta para el cursor
        linea = r[1]
        isempty(strip(linea)) && continue
        d = try jobj(linea) catch; nothing end
        d === nothing && continue
        tx = get(d, "text", "")
        isempty(tx) && continue
        f.bytes += ncodeunits(tx)
        n0 = length(f.buf)
        append!(f.buf, (Int(id) + 1 for id in BPE.encode(tok, tx)))
        f.tokens += length(f.buf) - n0
        leidas += 1
    end
    if terminado
        cerrar!(f)
        path = chunkpath(f, f.chunk)
        isfile(path) && rm(path; force = true)
        f.chunk += 1
        f.line = 0
        abrir!(f)
    end
    return leidas
end

function escribe_cursor(curpath, fuentes)
    dic = Dict{String,String}()
    for f in fuentes
        dic[string(f.ds, ".chunk")] = string(f.chunk)
        dic[string(f.ds, ".line")] = string(f.line)
        dic[string(f.ds, ".tokens")] = string(f.tokens)
    end
    open(curpath, "w") do io
        write(io, "{")
        primero = true
        for k in sort(collect(keys(dic)))
            primero || write(io, ",")
            write(io, "\"", k, "\": \"", dic[k], "\"")
            primero = false
        end
        write(io, "}")
    end
end

function main()
    minutos = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 120.0
    ckpt = length(ARGS) >= 2 ? ARGS[2] : normpath(@__DIR__, "..", "model_f4b.irmdl")
    curpath = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "f4b_cursor.json")

    isdir(BUFFER) || error("no existe el buffer: $BUFFER")
    dss = sort(filter(d -> isdir(joinpath(BUFFER, d)) &&
        !isempty(filter(x -> endswith(x, ".jsonl.gz"), readdir(joinpath(BUFFER, d)))),
        readdir(BUFFER)))
    isempty(dss) && error("buffer sin chunks .jsonl.gz en $BUFFER")

    tok = BPE.load_tok(get(ENV, "IRIS_TOK", normpath(@__DIR__, "..", "..", "assets", "iris.tok")))
    println("BPE vocab: ", 256 + length(tok.merges), " | datasets: ", join(dss, ", ")); flush(stdout)

    fuentes = Fuente[]
    for ds in dss
        w = get(PESOS, ds, 0.0)
        w == 0.0 && (println("(sin peso en el plan: $ds, ignorado)"); continue)
        push!(fuentes, Fuente(ds, w, 0, 0, nothing, nothing, nothing, Int[], 0.0, 0, 0, false))
    end
    isempty(fuentes) && error("ningun dataset con peso del plan en el buffer")
    sw = sum(f.w for f in fuentes)
    for f in fuentes; f.w /= sw; end

    if isfile(curpath)
        d = jobj(read(curpath, String))
        for f in fuentes
            k = string(f.ds, ".chunk")
            haskey(d, k) || continue
            f.chunk = parse(Int, d[k])
            f.line = parse(Int, get(d, string(f.ds, ".line"), "0"))
            f.tokens = parse(Int, get(d, string(f.ds, ".tokens"), "0"))
        end
        println("cursor: relance desde ", curpath)
    else
        println("cursor: nuevo (", curpath, ")")
    end
    reanudar = isfile(ckpt)
    m = reanudar ? load_model(ckpt) : init_model(ModelConfig())
    cfg = m.cfg
    println("modelo: ", reanudar ? "checkpoint $ckpt" : "init_model nuevo",
            " (dim ", cfg.dim, " L", cfg.n_layers, " vocab ", cfg.vocab,
            " ctx ", cfg.ctx, ", ", length(m.p), " par)")
    println("packing T=$T B=$B ($W tokens/paso) | ventana: $minutos min"); flush(stdout)

    st = sin_table(cfg)
    g = zeros(Float32, length(m.p))
    opt = AdamOpt(1.0f-3, length(m.p))
    tns = trainable_fase1(m, 1)

    t0 = time()
    deadline = t0 + minutos * 60.0
    tcur = t0; tckpt = t0; tprint = t0; tlast = t0
    pasos = 0; toks = 0; lossac = 0.0; lossn = 0
    lossv = 0.0; lossvn = 0          # loss de la ventana de 40 s (diagnostico)
    tokslast = 0

    println("--- ciclo en marcha (parada abrupta = se reanuda por cursor + ckpt) ---")
    while time() < deadline
        listas = [f for f in fuentes if length(f.buf) >= W]
        if isempty(listas)
            candidatos = [f for f in fuentes if !f.acabado]
            if isempty(candidatos)
                # sin mas chunk: junta restos en el mayor buf o termina
                total = sum(length(f.buf) for f in fuentes)
                if total < W
                    println("buffer agotado: $total tokens pendientes < W=$W")
                    break
                end
                sort!(fuentes; by = f -> (-length(f.buf), f.ds))
                big = fuentes[1]
                for f in fuentes[2:end]
                    append!(big.buf, f.buf); empty!(f.buf)
                end
                continue
            end
            sort!(candidatos; by = f -> (f.servidos / f.w, f.ds))
            refill!(candidatos[1], LINEAS_POR_REFILL, tok)
            continue
        end
        sort!(listas; by = f -> (f.servidos / f.w, f.ds))
        f = listas[1]
        tl = [f.buf[1:T], f.buf[(T + 1):W]]
        (l, lg, c) = forward_loss_batch(m, tl, st)
        fill!(g, 0.0f0)
        backward_batch!(m, g, tl, lg, c)
        adam_step!(m, g, opt, tns)
        deleteat!(f.buf, 1:W)
        f.servidos += W
        pasos += 1; toks += W
        lossac += l; lossn += 1
        lossv += l; lossvn += 1

        tn = time()
        if tn - tprint >= 40.0
            rec = toks - tokslast; dt = tn - tlast
            @printf("paso %5d | tok %9d | %.1f tok/s medio | %.1f tok/s ult | loss acum %.4f | loss ventana %.4f | %.1f min\n",
                    pasos, toks, toks / (tn - t0), dt > 0 ? rec / dt : 0.0,
                    lossac / lossn, lossv / max(lossvn, 1), (tn - t0) / 60)
            flush(stdout)
            lossv = 0.0; lossvn = 0
        end
        if tn - tcur >= CURSOR_MIN * 60
            escribe_cursor(curpath, fuentes); tcur = tn
        end
        if tn - tckpt >= CKPT_MIN * 60
            save_model(ckpt, m)
            escribe_cursor(curpath, fuentes)
            tckpt = tn
            println("[ckpt] ", round(Int, tn - t0), " s: modelo -> ", ckpt); flush(stdout)
            GC.gc()
        end
    end

    save_model(ckpt, m)
    escribe_cursor(curpath, fuentes)
    el = time() - t0
    println("--- fin del ciclo ---")
    @printf("pasos %d | tokens entrenados %d | %.1f tok/s medio | loss acumulada %.4f\n",
            pasos, toks, toks / max(el, 1e-9), lossn > 0 ? lossac / lossn : NaN)
    mb = sum(f.bytes for f in fuentes) / 1e6
    rss = try Sys.maxrss() / 2^30 catch; NaN end
    @printf("texto leido: %.2f MB | %.2f MB texto/h | RSS %.2f GB\n",
            mb, mb / (el / 3600), rss)
    for f in fuentes
        @printf("  %-14s servidos %8.0f tok (%4.1f%%) | chunk %d linea %d | bytes %.2f MB\n",
                f.ds, f.servidos, 100 * f.servidos / max(toks, 1), f.chunk, f.line, f.bytes / 1e6)
    end
    println("OK f4b_cycle")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
