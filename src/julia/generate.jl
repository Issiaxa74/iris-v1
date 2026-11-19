# generate.jl - muestreo de texto (inferencia) con un checkpoint IRISMDL.
#
# Sin cache de atencion: cada token nuevo repite el forward completo. Sirve para
# probar el modelo y para conversar con el; no es el runtime de produccion
# (ese es `runtime/`, el REPL cabrita en Rust que lee UKJ con mmap).
#
# Uso:
#   julia src/julia/generate.jl --prompt "Hola" --tokens 80
#   julia src/julia/generate.jl --prompt "..." --tokens 200 --temp 0.8 --top-k 40 --top-p 0.95 --seed 7
#
# Variables de entorno:
#   IRIS_CKPT  ruta al checkpoint .irmdl  (por defecto models/model.irmdl)
#   IRIS_TOK   ruta al tokenizer iris.tok (por defecto assets/iris.tok)
#
# Nota: el tokenizador tiene 1056 ids utiles (256 bytes + 800 fusiones) y el
# modelo reserva vocab=16384. Los ids muertos se enmascaran con -Inf antes de
# muestrear; sin eso el modelo puede escupir ids que no existen en el BPE.

using Printf, Random
include(joinpath(@__DIR__, "model.jl")); using .IrisModel
include(joinpath(@__DIR__, "BPE.jl"));   using .BPE

# ---------------------------- argumentos -------------------------------
function leer_args(argv)
    d = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("argumento no reconocido: $a")
        k = a[3:end]
        i += 1
        i <= length(argv) || error("falta el valor de --$k")
        d[k] = argv[i]
        i += 1
    end
    return d
end

args = leer_args(ARGS)
prompt = get(args, "prompt", "Hola")
nuevos = parse(Int, get(args, "tokens", "64"))
temp = parse(Float64, get(args, "temp", "0.9"))
topk = parse(Int, get(args, "top-k", "0"))
topp = parse(Float64, get(args, "top-p", "0.95"))
seed = parse(Int, get(args, "seed", "1"))

ckpt = get(ENV, "IRIS_CKPT", normpath(@__DIR__, "..", "..", "models", "model.irmdl"))
tokp = get(ENV, "IRIS_TOK", normpath(@__DIR__, "..", "..", "assets", "iris.tok"))

isfile(ckpt) || error("no existe el checkpoint: $ckpt (usa IRIS_CKPT=...)")
isfile(tokp) || error("no existe el tokenizer: $tokp (usa IRIS_TOK=...)")

# ---------------------------- modelo -----------------------------------
println("cargando $ckpt ...")
m = load_model(ckpt)
t = BPE.load_tok(tokp)
st = sin_table(m.cfg)
max_id = Int(BPE.vocab_size(t))          # ids validos: 1..max_id
@printf("modelo %d par | vocab modelo %d | vocab tokenizador %d | ctx %d\n",
        param_count(m), m.cfg.vocab, max_id, m.cfg.ctx)

ids = [Int(id) + 1 for id in BPE.encode(t, prompt)]
isempty(ids) && push!(ids, 1)            # el forward exige T >= 2
length(ids) < 2 && push!(ids, 1)

rng = Xoshiro(seed)

# --------------------------- muestreo ----------------------------------
"logits de la posicion siguiente a la ultima de `sec`"
function logits_siguiente(m, st, sec)
    # el forward exige 2 <= T <= ctx; recorta si la secuencia crece
    s = length(sec) > m.cfg.ctx - 1 ? sec[end-(m.cfg.ctx - 2):end] : sec
    _, logits, _ = forward_loss_batch(m, [s], st)
    return vec(logits[size(logits, 1), :])
end

"temperatura + top-k + top-p -> un id (1..vocab)"
function muestrear(lg::AbstractVector{Float32}, temp, topk, topp, max_id, rng)
    # temp=0 -> greedy determinista sobre los ids vivos (evita Inf/NaN en softmax)
    if temp <= 0
        mejor = 1
        for i in 2:min(max_id, length(lg))
            lg[i] > lg[mejor] && (mejor = i)
        end
        return mejor
    end
    v = length(lg)
    # enmascarar ids que el tokenizador no puede decodificar
    logp = Vector{Float32}(undef, v)
    @inbounds for i in 1:v
        logp[i] = (i <= max_id) ? lg[i] / Float32(temp) : -Inf32
    end
    # top-k por seleccion parcial barata (v=16384: ordenar es ~0.5 ms, suficiente)
    idx = sortperm(logp; rev = true)
    keep = topk > 0 ? min(topk, v) : v
    idx = idx[1:keep]

    # softmax sobre los candidatos
    mx = maximum(logp[i] for i in idx)
    p = Float64[]
    for i in idx
        push!(p, exp(Float64(logp[i] - mx)))
    end
    s = sum(p)
    p ./= s

    # top-p (nucleo)
    if topp < 1.0
        acc = 0.0
        corte = length(p)
        for j in 1:length(p)
            acc += p[j]
            if acc >= topp
                corte = j
                break
            end
        end
        p = p[1:corte]
        p ./= sum(p)
    end
    # muestreo categorico
    u = rand(rng)
    acc = 0.0
    for (j, pj) in enumerate(p)
        acc += pj
        if u <= acc
            return idx[j]
        end
    end
    return idx[end]
end

# --------------------------- generacion --------------------------------
print(prompt)
flush(stdout)
ultimo_texto = ""
for paso in 1:nuevos
    lg = logits_siguiente(m, st, ids)
    id = muestrear(lg, temp, topk, topp, max_id, rng)
    push!(ids, id)
    # decodifica el token nuevo; si son bytes parciales UTF-8 se acumulan
    txt = BPE.decode(t, UInt32[id - 1])
    print(txt)
    flush(stdout)
end
println()
