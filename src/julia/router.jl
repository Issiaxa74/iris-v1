# router.jl — Iris F3: clasificador-router ligero (~1M par con vocab 16k)
# Contrato con cabrita: fn(tokens_msg, tokens_historial) -> id_ruta.
# Bag-of-embeddings (por CONTENIDO, nunca por sonido/orden) + MLP.
# Ruta 0 = "generica": clase aprendida para temas fuera del árbol + regla de
# confianza tau (esquema de co-ayudador §3: P < tau -> fallback a generica).
# Emb propia y pequena (no comparte con el trunk): el router se carga entero
# en cabrita sin tocar los expertos. Determinista, sin dependencias externas.

module IrisRouter

using Random, LinearAlgebra

export RouterConf, Router, init_router, router_forward, router_loss_grad!,
       RouterAdam, adam_step!, router_predict, save_router, load_router

Base.@kwdef struct RouterConf
    vocab::Int = 1056
    dim::Int = 64
    hidden::Int = 256
    n_rutas::Int = 16
    seed::UInt32 = 0x49525452   # "IRTR"
end

mutable struct Router
    conf::RouterConf
    names::Vector{String}                 # names[1] = "generica" (id 0)
    ranges::Dict{String,UnitRange{Int}}
    p::Vector{Float32}
    tau::Float32
end

function _layout!(ranges, cfg::RouterConf)
    d, v, H, nr = cfg.dim, cfg.vocab, cfg.hidden, cfg.n_rutas
    pos = 1
    for (n, k) in (("emb", v * d), ("ph", d * d), ("w1", H * 2d),
                   ("b1", H), ("w2", nr * H), ("b2", nr))
        ranges[n] = pos:(pos + k - 1); pos += k
    end
    return pos - 1
end

function init_router(cfg::RouterConf, names::Vector{String}, tau::Float32 = 0.60f0)
    @assert length(names) == cfg.n_rutas && names[1] == "generica"
    ranges = Dict{String,UnitRange{Int}}()
    total = _layout!(ranges, cfg)
    p = Vector{Float32}(undef, total)
    rng = Xoshiro(cfg.seed)
    d = cfg.dim
    _filln!(p, ranges["emb"], rng, 0.02f0)
    fill!(view(p, ranges["ph"]), 0.0f0)      # historial sin datos aún: aporta 0
    _filln!(p, ranges["w1"], rng, 0.02f0)
    fill!(view(p, ranges["b1"]), 0.0f0)
    _filln!(p, ranges["w2"], rng, 0.02f0 / sqrt(Float32(cfg.hidden)))
    fill!(view(p, ranges["b2"]), 0.0f0)
    return Router(cfg, names, ranges, p, tau)
end

function _filln!(p, r, rng, s)
    @inbounds for i in 0:(length(r) - 1); p[r[1] + i] = s * randn(rng, Float32); end
end

@inline _gelu(x) = begin
    u = 0.7978845608f0 * (x + 0.044715f0 * x * x * x)
    0.5f0 * x * (1.0f0 + tanh(u))
end

@inline _gelu_grad(x) = begin
    t = tanh(0.7978845608f0 * (x + 0.044715f0 * x * x * x))
    0.5f0 * (1.0f0 + t) +
    0.5f0 * x * (1.0f0 - t * t) * 0.7978845608f0 * (1.0f0 + 3.0f0 * 0.044715f0 * x * x)
end

# media de embeddings de una lista de tokens (ids 1-based)
function _pool!(out, p, r_emb, toks, dim)
    fill!(out, 0.0f0)
    isempty(toks) && return 0
    @inbounds for t in toks
        base = r_emb[1] + (t - 1) * dim
        for k in 1:dim; out[k] += p[base + k - 1]; end
    end
    inv = 1.0f0 / length(toks)
    for k in 1:dim; out[k] *= inv; end
    return length(toks)
end

"""logits [n_rutas] para (tokens_msg, tokens_hist); solo inferencia."""
function router_forward(rt::Router, toks_msg, toks_hist)
    d = rt.conf.dim; H = rt.conf.hidden; nr = rt.conf.n_rutas
    R = rt.ranges; p = rt.p
    m = zeros(Float32, d); h = zeros(Float32, d)
    _pool!(m, p, R["emb"], toks_msg, d)
    _pool!(h, p, R["emb"], toks_hist, d)
    u = vcat(m, reshape(view(p, R["ph"]), d, d) * h)
    z1 = reshape(view(p, R["w1"]), H, 2d) * u .+ view(p, R["b1"])
    ha = _gelu.(z1)
    return reshape(view(p, R["w2"]), nr, H) * ha .+ view(p, R["b2"])
end

"""
    router_loss_grad!(rt, g, batch) -> loss_media

batch: Vector de (toks_msg, toks_hist, label) con label 1-based. Acumula el
gradiente medio en `g` (llegar en ceros). Forward+backward analítico exacto.
"""
function router_loss_grad!(rt::Router, g, batch)
    d = rt.conf.dim; H = rt.conf.hidden; nr = rt.conf.n_rutas; v = rt.conf.vocab
    R = rt.ranges; p = rt.p
    rE, rPH, rW1, rb1, rW2, rb2 = R["emb"], R["ph"], R["w1"], R["b1"], R["w2"], R["b2"]
    W1 = reshape(view(p, rW1), H, 2d)
    W2 = reshape(view(p, rW2), nr, H)
    PH = reshape(view(p, rPH), d, d)
    b1 = view(p, rb1); b2 = view(p, rb2)

    m = zeros(Float32, d); h = zeros(Float32, d)
    u = zeros(Float32, 2d); z1 = zeros(Float32, H); ha = zeros(Float32, H)
    dz1 = zeros(Float32, H); dhact = zeros(Float32, H)
    dlog = zeros(Float32, nr); logits = zeros(Float32, nr)
    du = zeros(Float32, 2d); dh = zeros(Float32, d)
    loss = 0.0
    n = length(batch)
    for (tm, th, lab) in batch
        nm = _pool!(m, p, rE, tm, d)
        nh = _pool!(h, p, rE, th, d)
        @inbounds for i in 1:d
            acc = 0.0f0
            for k in 1:d; acc += PH[i, k] * h[k]; end
            u[i] = m[i]; u[d + i] = acc
        end
        mul!(z1, W1, u); z1 .+= b1
        ha .= _gelu.(z1)
        mul!(logits, W2, ha); logits .+= b2
        mx = maximum(logits)
        s = 0.0f0
        @inbounds for j in 1:nr; s += exp(logits[j] - mx); end
        loss += (mx + log(s)) - logits[lab]
        fill!(dlog, 0.0f0)
        @inbounds for j in 1:nr; dlog[j] = exp(logits[j] - mx) / s; end
        dlog[lab] -= 1.0f0
        dW2 = reshape(view(g, rW2), nr, H)
        @inbounds for j in 1:nr, k in 1:H; dW2[j, k] += dlog[j] * ha[k]; end
        @inbounds for j in 1:nr; g[rb2[1] + j - 1] += dlog[j]; end
        mul!(dhact, adjoint(W2), dlog)
        @inbounds for k in 1:H; dz1[k] = dhact[k] * _gelu_grad(z1[k]); end
        dW1 = reshape(view(g, rW1), H, 2d)
        @inbounds for i in 1:H, k in 1:2d; dW1[i, k] += dz1[i] * u[k]; end
        @inbounds for i in 1:H; g[rb1[1] + i - 1] += dz1[i]; end
        mul!(du, adjoint(W1), dz1)
        dm = view(du, 1:d); duh = view(du, (d + 1):(2d))
        if nh > 0
            dPH = reshape(view(g, rPH), d, d)
            @inbounds for i in 1:d, k in 1:d; dPH[i, k] += duh[i] * h[k]; end
            mul!(dh, adjoint(PH), duh)
        end
        demb = reshape(view(g, rE), d, v)
        im = 1.0f0 / nm
        @inbounds for t in tm, k in 1:d; demb[k, t] += dm[k] * im; end
        if nh > 0
            ih = 1.0f0 / nh
            @inbounds for t in th, k in 1:d; demb[k, t] += dh[k] * ih; end
        end
    end
    g ./= n
    return loss / n
end

mutable struct RouterAdam
    lr::Float32; b1::Float32; b2::Float32; eps::Float32; wd::Float32; t::Int
    mm::Vector{Float32}; vv::Vector{Float32}
end
RouterAdam(lr::Real, n::Int) = RouterAdam(Float32(lr), 0.9f0, 0.95f0, 1.0f-8, 1.0f-4,
                                          0, zeros(Float32, n), zeros(Float32, n))

function adam_step!(rt::Router, g, opt::RouterAdam)
    opt.t += 1
    bc1 = 1 - opt.b1^opt.t; bc2 = 1 - opt.b2^opt.t
    for name in keys(rt.ranges)
        r = rt.ranges[name]
        decay = (name != "b1" && name != "b2")
        @inbounds for i in r
            gi = g[i]
            opt.mm[i] = opt.b1 * opt.mm[i] + (1 - opt.b1) * gi
            opt.vv[i] = opt.b2 * opt.vv[i] + (1 - opt.b2) * gi * gi
            mh = opt.mm[i] / bc1; vh = opt.vv[i] / bc2
            pi = rt.p[i]
            upd = mh / (sqrt(vh) + opt.eps)
            if decay; upd += opt.wd * pi; end
            rt.p[i] = pi - opt.lr * upd
        end
    end
    return nothing
end

"""Predicción con regla del esquema §3: argmax; si la mejor ruta no es generica
y su P < tau -> fallback a generica (id 0)."""
function router_predict(rt::Router, toks_msg, toks_hist; tau::Union{Nothing,Float32} = nothing)
    t = tau === nothing ? rt.tau : tau
    logits = router_forward(rt, toks_msg, toks_hist)
    mx = maximum(logits)
    s = 0.0f0
    @inbounds for j in eachindex(logits); s += exp(logits[j] - mx); end
    probs = [exp(logits[j] - mx) / s for j in eachindex(logits)]
    pm, i = findmax(probs)
    if i == 1 || pm < t
        return (id_ruta = UInt32(0), nombre = rt.names[1], p = pm, probs = probs)
    end
    return (id_ruta = UInt32(i - 1), nombre = rt.names[i], p = pm, probs = probs)
end

const MAGIC = UInt8[0x49, 0x52, 0x49, 0x53, 0x52, 0x54, 0x52, 0x00]  # "IRISRTR\0"

function save_router(path::AbstractString, rt::Router)
    open(path, "w") do io
        write(io, MAGIC); write(io, UInt32(1))
        for f in (:vocab, :dim, :hidden, :n_rutas); write(io, UInt32(getfield(rt.conf, f))); end
        write(io, rt.conf.seed)
        write(io, rt.tau)
        write(io, Int64(length(rt.p))); write(io, rt.p)
        write(io, UInt32(length(rt.names)))
        for (i, nm) in enumerate(rt.names)
            write(io, UInt32(i - 1))
            b = Vector{UInt8}(nm)
            write(io, UInt16(length(b))); write(io, b)
        end
    end
    return path
end

function load_router(path::AbstractString)
    open(path) do io
        @assert read(io, 8) == MAGIC "no es IRISRTR"
        @assert read(io, UInt32) == UInt32(1) "version desconocida"
        cfg = RouterConf(Int(read(io, UInt32)), Int(read(io, UInt32)),
                         Int(read(io, UInt32)), Int(read(io, UInt32)), read(io, UInt32))
        tau = read(io, Float32)
        n = read(io, Int64)
        p = Vector{Float32}(undef, n); read!(io, p)
        k = read(io, UInt32)
        names = String[]
        for i in 1:k
            @assert read(io, UInt32) == UInt32(i - 1) "ids desordenados"
            ln = read(io, UInt16)
            push!(names, String(read(io, ln)))
        end
        ranges = Dict{String,UnitRange{Int}}()
        total = _layout!(ranges, cfg)
        @assert total == n "layout no coincide"
        return Router(cfg, names, ranges, p, tau)
    end
end

end # module
