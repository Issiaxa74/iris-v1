# model.jl — Iris F2b/F4a: definición del modelo por rutas (fase 1: trunk 1024×14 + 16 hojas)
# Pesos maestros fp32; la cuantización p4-one ocurre al serializar con UKJ.jl.
# Determinista: init con semilla fija, sin RNG global. Sin dependencias externas.
#
# F4a (verificada con model_test.jl TODO OK):
#   · GELU / gelu_grad como broadcast @. fusionado (tanh/exp con kernel SIMD de Base).
#     Diferencia vs la versión escalar de F2b: ≤1 ulp en tanh — dentro del ruido fp32.
#   - softmax CE por trozos de columnas con hilos (accesos contiguos): el forward
#     guarda E=exp(logits-max_fila) y sc=k1/s; el backward construye dlogits=E.*sc
#     SIN exps nuevas ni div/rem de broadcast (el bloque [T,v] es el caliente).
#   · Cache guarda n1, n2, h y hfin → el backward NO recomputa layernorms ni gelu.
#   · Atención: escala 1/√Hd fundida en el softmax, cero del triángulo superior por fila.
#   · PARALELISMO (fase B): hilos de Julia sobre trabajo por filas/cabezas/rangos —
#     layernorm fwd/bwd por filas, atención por cabezas (buffer S/A/Gm por cabeza),
#     gelu por bloques de columnas y Adam por trozos de rango. Cada elemento se calcula
#     gelu por bloques de columnas y Adam por trozos de rango. Cada elemento se calcula
#     una sola vez → resultados idénticos con 1 u 8 hilos. Con 1 hilo = camino serial.
#   · MICRO-BATCH (fase C): forward_loss_batch/backward_batch! empaquetan nb secuencias
#     de igual longitud por filas (sin cross-atencion): los GEMMs procesan [nb*T,d], la
#     atención corre por unidades (seq,cabeza)=nb*H y Adam se amortiza entre secuencias.
#     forward_loss/backward! quedan como caso particular nb=1.
#     VERIFICADO (f4a_batch_test.jl, 2026-11-17): loss del batch = media de individuales
#     (rel 2.15e-09); gradiente del batch = suma/nb de individuales (max abs 7.45e-09
#     sobre gmax 1.15e-02); sin cross-atencion (diff exacta 0). model_test TODO OK a 1 y 8 hilos.

module IrisModel

using Random
using LinearAlgebra

export ModelConfig, Model, init_model, param_count, sin_table, layout,
       forward_loss, backward!, adam_step!, AdamOpt, save_model, load_model,
       forward_loss_batch, backward_batch!,
       default_trainable, trainable_fase1

Base.@kwdef struct ModelConfig
    dim::Int = 1024
    n_layers::Int = 14
    n_heads::Int = 16
    ffn::Int = 4096
    vocab::Int = 16384
    ctx::Int = 2048
    n_hojas::Int = 16
    seed::UInt32 = 0x49524953   # "IRIS"
end

mutable struct Model
    cfg::ModelConfig
    names::Vector{String}
    ranges::Dict{String,UnitRange{Int}}
    p::Vector{Float32}
end

param_count(m::Model) = length(m.p)

# ----------------------------- layout ----------------------------------

const CAMPOS_BLOQUE = ("ln1g", "ln1b", "wqkv", "wo", "ln2g", "ln2b", "w1", "w2")

_tam(c::String, d::Int, f::Int) =
    c == "wqkv" ? d * 3d : c == "wo" ? d * d : c == "w1" ? d * f : c == "w2" ? f * d : d

function _layout!(names, ranges, pos, cfg::ModelConfig)
    d, v, f = cfg.dim, cfg.vocab, cfg.ffn
    push!(names, "emb"); ranges["emb"] = pos:(pos + d * v - 1); pos += d * v
    for pref in vcat(["t$l" for l in 1:cfg.n_layers],
                     ["e$e" for e in 1:cfg.n_hojas], ["sft"]), c in CAMPOS_BLOQUE
        n = _tam(c, d, f)
        push!(names, "$pref.$c"); ranges["$pref.$c"] = pos:(pos + n - 1); pos += n
    end
    push!(names, "fin.ln_g"); ranges["fin.ln_g"] = pos:(pos + d - 1); pos += d
    push!(names, "fin.ln_b"); ranges["fin.ln_b"] = pos:(pos + d - 1); pos += d
    return pos - 1
end

function layout(cfg::ModelConfig)
    names = String[]
    ranges = Dict{String,UnitRange{Int}}()
    total = _layout!(names, ranges, 1, cfg)
    return names, ranges, total
end

function _filln!(p, r, rng, s)
    @inbounds for i in 0:(length(r) - 1)
        p[r[1] + i] = s * randn(rng, Float32)
    end
    return nothing
end

function init_model(cfg::ModelConfig)
    names, ranges, total = layout(cfg)
    p = Vector{Float32}(undef, total)
    rng = Xoshiro(cfg.seed)
    d, f = cfg.dim, cfg.ffn
    resid = 1.0f0 / sqrt(Float32(2 * cfg.n_layers))
    s0 = 0.02f0

    _filln!(p, ranges["emb"], rng, s0)
    for pref in vcat(["t$l" for l in 1:cfg.n_layers],
                     ["e$e" for e in 1:cfg.n_hojas], ["sft"])
        r = ranges["$pref.ln1g"]; p[r] .= 1.0f0
        r = ranges["$pref.ln1b"]; p[r] .= 0.0f0
        _filln!(p, ranges["$pref.wqkv"], rng, s0)
        _filln!(p, ranges["$pref.wo"], rng, s0 * resid)
        r = ranges["$pref.ln2g"]; p[r] .= 1.0f0
        r = ranges["$pref.ln2b"]; p[r] .= 0.0f0
        _filln!(p, ranges["$pref.w1"], rng, s0)
        _filln!(p, ranges["$pref.w2"], rng, s0 * resid)
    end
    p[ranges["fin.ln_g"]] .= 1.0f0
    p[ranges["fin.ln_b"]] .= 0.0f0
    return Model(cfg, names, ranges, p)
end

function sin_table(cfg::ModelConfig)
    out = Matrix{Float32}(undef, cfg.ctx, cfg.dim)
    for pos in 0:(cfg.ctx - 1), i in 0:(cfg.dim ÷ 2 - 1)
        freq = exp(-log(10000.0) * 2i / cfg.dim)
        ang = Float32(pos * freq)
        out[pos + 1, 2i + 1] = sin(ang)
        out[pos + 1, 2i + 2] = cos(ang)
    end
    return out
end

function default_trainable(m::Model)
    return copy(m.names)
end

"nombres entrenables de la fase 1: trunk + experto `hoja` + mundo SFT + embeddings + LN final"
function trainable_fase1(m::Model, hoja::Int)
    cfg = m.cfg
    ns = String[]
    for l in 1:cfg.n_layers, c in CAMPOS_BLOQUE
        push!(ns, "t$l.$c")
    end
    for c in CAMPOS_BLOQUE
        push!(ns, "e$hoja.$c")
    end
    for c in CAMPOS_BLOQUE
        push!(ns, "sft.$c")
    end
    append!(ns, ["emb", "fin.ln_g", "fin.ln_b"])
    return ns
end

# ------------------------ utilidades de hilos --------------------------

"divide 1:n en `parts` rangos contiguos casi iguales"
function _csplit(n::Int, parts::Int)
    parts = clamp(parts, 1, n)
    out = Vector{UnitRange{Int}}(undef, parts)
    base = n ÷ parts
    rem = n - base * parts
    s = 1
    for i in 1:parts
        len = base + (i <= rem ? 1 : 0)
        out[i] = s:(s + len - 1)
        s += len
    end
    return out
end

# --------------------------- forward -----------------------------------

mutable struct Cache
    T::Int                          # filas totales: nb secuencias de longitud bl
    bl::Int                         # longitud de cada secuencia
    nb::Int                         # numero de secuencias del batch
    xin::Vector{Matrix{Float32}}
    n1::Vector{Matrix{Float32}}
    m1::Vector{Vector{Float32}}; r1::Vector{Vector{Float32}}
    qkv::Vector{Matrix{Float32}}
    ao::Vector{Matrix{Float32}}
    x2::Vector{Matrix{Float32}}
    n2::Vector{Matrix{Float32}}
    m2::Vector{Vector{Float32}}; r2::Vector{Vector{Float32}}
    pre::Vector{Matrix{Float32}}
    h::Vector{Matrix{Float32}}
    xfin::Matrix{Float32}
    hfin::Matrix{Float32}
    mf::Vector{Float32}; rf::Vector{Float32}
    E::Matrix{Float32}              # exp(logits - max por fila) del softmax CE
    sc::Vector{Float32}             # k1/s por fila (fila T = 0)
end

# layernorm por filas (cada fila es independiente → seguro con @threads)
function _ln!(out, x, g, b, mean, rstd)
    T, D = size(x)
    Threads.@threads for t in 1:T
        @inbounds begin
            s = 0.0f0
            for k in 1:D; s += x[t, k]; end
            μ = s / D
            v = 0.0f0
            for k in 1:D; d = x[t, k] - μ; v += d * d; end
            r = 1.0f0 / sqrt(v / D + 1.0f-5)
            mean[t] = μ; rstd[t] = r
            for k in 1:D
                out[t, k] = (x[t, k] - μ) * r * g[k] + b[k]
            end
        end
    end
    return nothing
end

function _ln_bwd!(dx, dn, x, g, mean, rstd)
    T, D = size(x)
    Threads.@threads for t in 1:T
        @inbounds begin
            μ = mean[t]; r = rstd[t]
            c1 = 0.0f0; c2 = 0.0f0
            for k in 1:D
                xh = (x[t, k] - μ) * r
                dxh = dn[t, k] * g[k]
                c1 += dxh; c2 += dxh * xh
            end
            c1 /= D; c2 /= D
            for k in 1:D
                xh = (x[t, k] - μ) * r
                dx[t, k] += r * (dn[t, k] * g[k] - c1 - xh * c2)
            end
        end
    end
    return nothing
end

# gradientes de los parametros del layernorm (gamma g, beta b):
#   db[k] = sum_m dn[m,k];  dg[k] = sum_m dn[m,k]*(x[m,k]-mu[m])*rstd[m]
function _ln_param_grads!(dg, db, dn, x, mean, rstd)
    M, D = size(x)
    fill!(dg, 0.0f0); fill!(db, 0.0f0)
    @inbounds for m in 1:M
        mu = mean[m]; r = rstd[m]
        for k in 1:D
            dnk = dn[m, k]
            db[k] += dnk
            dg[k] += dnk * (x[m, k] - mu) * r
        end
    end
    return nothing
end

# GELU con tanh por bloques de columnas (columnas contiguas → vista continua + SIMD)
function _gelu_cols!(h, pre)
    T, f = size(pre)
    cc = _csplit(f, Threads.nthreads())
    Threads.@threads for ci in eachindex(cc)
        cols = cc[ci]
        pv = view(pre, :, cols)
        hv = view(h, :, cols)
        @. hv = 0.5f0 * pv * (1.0f0 + tanh(0.7978845608f0 * (pv + 0.044715f0 * pv * pv * pv)))
    end
    return nothing
end

# gelu_grad: 0.5(1+t) + 0.5x(1-t^2)A(1+3Bx^2), t = tanh(A(x+Bx^3)) - UN solo tanh
# (equivalente algebraico de 0.5(1+t)[1 + xA(1+3Bx^2)(1-t)]) por bloques de columnas
function _gelu_grad_cols!(dpre, dh, pre)
    T, f = size(pre)
    cc = _csplit(f, Threads.nthreads())
    tb = Matrix{Float32}(undef, T, f)
    Threads.@threads for ci in eachindex(cc)
        cols = cc[ci]
        pv = view(pre, :, cols)
        dhv = view(dh, :, cols)
        dpv = view(dpre, :, cols)
        tbv = view(tb, :, cols)
        @. tbv = tanh(0.7978845608f0 * (pv + 0.044715f0 * pv * pv * pv))
        @. dpv = dhv * (0.5f0 * (1.0f0 + tbv) +
                        pv * 0.5f0 * (1.0f0 - tbv * tbv) * 0.7978845608f0 *
                        (1.0f0 + 3.0f0 * 0.044715f0 * pv * pv))
    end
    return nothing
end

# atencion causal multi-cabeza sobre nb secuencias empaquetadas de longitud bl
# (qkv [nb*bl,3d], ao [nb*bl,d]); S: >=nb*H buffers [bl,bl], una unidad (seq,cabeza).
# Las secuencias NO se atienden entre si: cada una es causal dentro de su bloque.
# BLAS=1 dentro: los muls por (seq,cabeza) son pequenos; el pool de OpenBLAS solo
# anade contencion y spin-wait (medido: 12.8 -> 7.4 ms por capa en backward).
function _attn!(ao, qkv, d, H, bl, nb, S)
    Hd = d ÷ H
    inv = 1.0f0 / sqrt(Float32(Hd))
    b0 = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads for u in 1:(nb * H)
        s = (u - 1) ÷ H + 1
        h = (u - 1) % H + 1
        Sh = S[u]
        rows = ((s - 1) * bl) .+ (1:bl)
        qcols = (h - 1) * Hd .+ (1:Hd)
        Qh = view(qkv, rows, qcols)
        Kh = view(qkv, rows, d .+ qcols)
        Vh = view(qkv, rows, 2d .+ qcols)
        mul!(Sh, Qh, transpose(Kh))
        @inbounds for t in 1:bl
            mx = -Inf32
            for k in 1:t
                v2 = Sh[t, k] * inv
                Sh[t, k] = v2
                v2 > mx && (mx = v2)
            end
            ssum = 0.0f0
            for k in 1:t
                e = exp(Sh[t, k] - mx)
                Sh[t, k] = e; ssum += e
            end
            isi = 1.0f0 / ssum
            for k in 1:t; Sh[t, k] *= isi; end
            for k in (t + 1):bl; Sh[t, k] = 0.0f0; end
        end
        mul!(view(ao, rows, qcols), Sh, Vh)
    end
    BLAS.set_num_threads(b0)
    return nothing
end

# backward de la atencion: mismas unidades (seq,cabeza); dqkv escribe bloques disjuntos
function _attn_bwd!(dqkv, dao, qkv, d, H, bl, nb, S, A, G)
    Hd = d ÷ H
    inv = 1.0f0 / sqrt(Float32(Hd))
    b0 = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads for u in 1:(nb * H)
        s = (u - 1) ÷ H + 1
        h = (u - 1) % H + 1
        Sb = S[u]; Ab = A[u]; Gmb = G[u]
        rows = ((s - 1) * bl) .+ (1:bl)
        qcols = (h - 1) * Hd .+ (1:Hd)
        Qh = view(qkv, rows, qcols)
        Kh = view(qkv, rows, d .+ qcols)
        Vh = view(qkv, rows, 2d .+ qcols)
        dAh = view(dao, rows, qcols)
        mul!(Ab, Qh, transpose(Kh))
        @inbounds for t in 1:bl
            mx = -Inf32
            for k in 1:t
                v2 = Ab[t, k] * inv
                Ab[t, k] = v2
                v2 > mx && (mx = v2)
            end
            ssum = 0.0f0
            for k in 1:t
                e = exp(Ab[t, k] - mx)
                Ab[t, k] = e; ssum += e
            end
            isi = 1.0f0 / ssum
            for k in 1:t; Ab[t, k] *= isi; end
            for k in (t + 1):bl; Ab[t, k] = 0.0f0; end
        end
        mul!(Gmb, dAh, transpose(Vh))          # g[i,j] = dAh[i,:]*Vh[j,:]
        @inbounds for t in 1:bl
            cs = 0.0f0
            for k in 1:t; cs += Gmb[t, k] * Ab[t, k]; end
            for k in 1:t
                Sb[t, k] = Ab[t, k] * (Gmb[t, k] - cs) * inv
            end
            for k in (t + 1):bl; Sb[t, k] = 0.0f0; end
        end
        mul!(view(dqkv, rows, qcols), Sb, Kh)                 # dQh
        mul!(view(dqkv, rows, d .+ qcols), transpose(Sb), Qh) # dKh
        mul!(view(dqkv, rows, 2d .+ qcols), transpose(Ab), dAh) # dVh
    end
    BLAS.set_num_threads(b0)
    return nothing
end

"""
    forward_loss(m, tokens, st) -> (loss::Float64, logits::Matrix, cache)

Pase forward fp32 de UNA secuencia (next-token): caso particular del batch (nb=1).
	okens: ids 1..vocab, longitud T. st: tabla de sin_table.
Loss = CE media sobre posiciones 1..T-1.
"""
function forward_loss(m::Model, tokens::Vector{Int}, st::Matrix{Float32})
    return forward_loss_batch(m, [tokens], st)
end

"""
    forward_loss_batch(m, tokens_list, st) -> (loss::Float64, logits::Matrix, cache)

Micro-batch de 
b secuencias de longitud IGUAL T, empaquetadas por filas
(logits [nb*T, v]). Sin cross-atencion: cada secuencia es causal en su bloque.
Loss = media CE sobre nb*(T-1) posiciones; el backward asociado es ackward_batch!.
"""
function forward_loss_batch(m::Model, tl::Vector{Vector{Int}}, st::Matrix{Float32})
    cfg = m.cfg
    nb = length(tl)
    T = length(tl[1])
    @assert all(length(s) == T for s in tl) "las secuencias del batch deben tener la misma longitud"
    @assert 2 <= T <= cfg.ctx "T=$T fuera de rango"
    d = cfg.dim; f = cfg.ffn; H = cfg.n_heads; v = cfg.vocab
    M = nb * T
    W = m.p; R = m.ranges
    off_emb = R["emb"][1]
    k1 = 1.0f0 / Float32(nb * (T - 1))

    x = Matrix{Float32}(undef, M, d)
    @inbounds for s in 1:nb, t in 1:T, k in 1:d
        x[(s - 1) * T + t, k] = W[off_emb + (tl[s][t] - 1) * d + k - 1] + st[t, k]
    end

    xin = Matrix{Float32}[]; n1v = Matrix{Float32}[]
    m1v = Vector{Float32}[]; r1v = Vector{Float32}[]
    qkvv = Matrix{Float32}[]; aov = Matrix{Float32}[]; x2v = Matrix{Float32}[]
    n2v = Matrix{Float32}[]
    m2v = Vector{Float32}[]; r2v = Vector{Float32}[]
    prev = Matrix{Float32}[]; hv = Matrix{Float32}[]

    Sheads = [Matrix{Float32}(undef, T, T) for _ in 1:(nb * H)]   # buffer por (seq,cabeza)
    for l in 1:cfg.n_layers
        push!(xin, x)                        # entrada residual de esta capa (no se muta)
        n1 = similar(x); m1 = Vector{Float32}(undef, M); r1 = Vector{Float32}(undef, M)
        _ln!(n1, x, view(W, R["t$l.ln1g"]), view(W, R["t$l.ln1b"]), m1, r1)
        qkv = n1 * reshape(view(W, R["t$l.wqkv"]), d, 3d)           # [M,3d]
        ao = Matrix{Float32}(undef, M, d)
        _attn!(ao, qkv, d, H, T, nb, Sheads)
        mm = ao * reshape(view(W, R["t$l.wo"]), d, d)               # [M,d]
        x2 = Matrix{Float32}(undef, M, d)
        @. x2 = x + mm
        n2 = similar(x2); m2 = Vector{Float32}(undef, M); r2 = Vector{Float32}(undef, M)
        _ln!(n2, x2, view(W, R["t$l.ln2g"]), view(W, R["t$l.ln2b"]), m2, r2)
        pre = n2 * reshape(view(W, R["t$l.w1"]), d, f)              # [M,f]
        h = Matrix{Float32}(undef, M, f)
        _gelu_cols!(h, pre)
        ff = h * reshape(view(W, R["t$l.w2"]), f, d)                # [M,d]
        xn = Matrix{Float32}(undef, M, d)
        @. xn = x2 + ff
        x = xn
        push!(m1v, m1); push!(r1v, r1); push!(qkvv, qkv); push!(aov, ao)
        push!(x2v, x2); push!(m2v, m2); push!(r2v, r2); push!(prev, pre)
        push!(n1v, n1); push!(n2v, n2); push!(hv, h)
    end

    xfin = x
    mf = Vector{Float32}(undef, M); rf = Vector{Float32}(undef, M)
    hfin = similar(xfin)
    _ln!(hfin, xfin, view(W, R["fin.ln_g"]), view(W, R["fin.ln_b"]), mf, rf)

    emb = reshape(view(W, R["emb"]), d, v)      # [d, v]
    logits = hfin * emb                          # [M, v]

    # softmax CE por trozos de columnas (hilos): max, exp y suma con accesos contiguos
    ccv = _csplit(v, Threads.nthreads())
    maxs = [fill(-Inf32, M) for _ in ccv]
    Threads.@threads for ci in eachindex(ccv)
        Lg = view(logits, :, ccv[ci]); nm = maxs[ci]
        @inbounds for m in 1:M; nm[m] = -Inf32; end
        @inbounds for j in axes(Lg, 2), m in axes(Lg, 1)
            xj = Lg[m, j]
            xj > nm[m] && (nm[m] = xj)
        end
    end
    mxv = Vector{Float32}(undef, M)
    @inbounds for m in 1:M
        mmax = -Inf32
        for ci in eachindex(ccv)
            maxs[ci][m] > mmax && (mmax = maxs[ci][m])
        end
        mxv[m] = mmax
    end
    E = Matrix{Float32}(undef, M, v)
    sums = [zeros(Float32, M) for _ in ccv]
    Threads.@threads for ci in eachindex(ccv)
        cols = ccv[ci]; Lg = view(logits, :, cols); sm = sums[ci]
        @inbounds for j in axes(Lg, 2)
            for m in axes(Lg, 1)
                e = exp(Lg[m, j] - mxv[m])
                E[m, cols[j]] = e
                sm[m] += e
            end
        end
    end
    sv = Vector{Float32}(undef, M)
    @inbounds for m in 1:M
        ssum = 0.0f0
        for ci in eachindex(ccv); ssum += sums[ci][m]; end
        sv[m] = ssum
    end
    sc = Vector{Float32}(undef, M)
    @inbounds for m in 1:M
        sc[m] = m % T == 0 ? 0.0f0 : k1 / sv[m]   # ultima fila de CADA secuencia a 0
    end

    loss = 0.0
    @inbounds for s in 1:nb, t in 1:(T - 1)
        m = (s - 1) * T + t
        loss -= Float64(logits[m, tl[s][t + 1]]) -
                (Float64(mxv[m]) + log(Float64(sv[m])))
    end
    loss /= Float64(nb * (T - 1))

    return loss, logits,
           Cache(M, T, nb, xin, n1v, m1v, r1v, qkvv, aov, x2v, n2v, m2v, r2v, prev, hv,
                 xfin, hfin, mf, rf, E, sc)
end

# --------------------------- backward ----------------------------------

"""
    backward!(m, g, tokens, logits, cache)

Llena g (mismo tamano que m.p; debe llegar a ceros) con el backward de la
CE media de orward_loss. Caso particular (nb=1) de ackward_batch!.
"""
function backward!(m::Model, g, tokens::Vector{Int}, logits, c::Cache)
    @assert c.nb == 1 "cache de batch: usa backward_batch!"
    return backward_batch!(m, g, [tokens], logits, c)
end

"""
    backward_batch!(m, g, tokens_list, logits, cache)

Backward del micro-batch de orward_loss_batch: suma sobre las nb secuencias
en un solo paso (los GEMMs de pesos acumulan automaticamente sobre todas las
filas). Equivale a sumar los gradientes de las nb secuencias individuales.
"""
function backward_batch!(m::Model, g, tl::Vector{Vector{Int}}, logits, c::Cache)
    cfg = m.cfg
    nb = c.nb; T = c.bl; M = c.T
    @assert length(tl) == nb
    d = cfg.dim; f = cfg.ffn; H = cfg.n_heads; v = cfg.vocab
    W = m.p; R = m.ranges
    k1 = 1.0f0 / Float32(nb * (T - 1))

    # dlogits = softmax*k1 - onehot*k1, SIN exp nueva: reutiliza E y sc del forward
    # (troceado por columnas con hilos: accesos contiguos, sin div/rem de broadcast)
    dlogits = Matrix{Float32}(undef, M, v)
    ccv = _csplit(v, Threads.nthreads())
    Threads.@threads for ci in eachindex(ccv)
        Ev = view(c.E, :, ccv[ci]); dv = view(dlogits, :, ccv[ci])
        @inbounds for j in axes(dv, 2), m in axes(dv, 1)
            dv[m, j] = Ev[m, j] * c.sc[m]
        end
    end
    @inbounds for s in 1:nb, t in 1:(T - 1)
        dlogits[(s - 1) * T + t, tl[s][t + 1]] -= k1
    end

    # tied embeddings: dh = dlogits*embT; demb[k,tok] += dlogits[t,tok]*hf[t,k]
    dhf = dlogits * adjoint(reshape(view(W, R["emb"]), d, v))     # [M,d]
    demb = reshape(view(g, R["emb"]), d, v)
    mul!(demb, transpose(c.hfin), dlogits)   # denso: dlogits[m,j] != 0 salvo fila final

    dx = similar(dhf); fill!(dx, 0.0f0)
    _ln_bwd!(dx, dhf, c.xfin, view(W, R["fin.ln_g"]), c.mf, c.rf)
    _ln_param_grads!(view(g, R["fin.ln_g"]), view(g, R["fin.ln_b"]), dhf, c.xfin, c.mf, c.rf)

    # buffers por (seq,cabeza) (se reutilizan en las 14 capas)
    Sbuf = [Matrix{Float32}(undef, T, T) for _ in 1:(nb * H)]
    Abuf = [Matrix{Float32}(undef, T, T) for _ in 1:(nb * H)]
    Gmbuf = [Matrix{Float32}(undef, T, T) for _ in 1:(nb * H)]

    for l in reverse(1:cfg.n_layers)
        # --- FFN (n2 y h vienen del cache: sin recomputar layernorm ni gelu) ---
        dff = dx
        mul!(reshape(view(g, R["t$l.w2"]), f, d), transpose(c.h[l]), dff)   # [f,d]
        dh = dff * adjoint(reshape(view(W, R["t$l.w2"]), f, d))            # [M,f]
        dpre = Matrix{Float32}(undef, M, f)
        _gelu_grad_cols!(dpre, dh, c.pre[l])
        mul!(reshape(view(g, R["t$l.w1"]), d, f), transpose(c.n2[l]), dpre)
        dn2 = dpre * adjoint(reshape(view(W, R["t$l.w1"]), d, f))          # [M,d]
        _ln_param_grads!(view(g, R["t$l.ln2g"]), view(g, R["t$l.ln2b"]), dn2, c.x2[l], c.m2[l], c.r2[l])
        dx2 = similar(dn2); fill!(dx2, 0.0f0)
        _ln_bwd!(dx2, dn2, c.x2[l], view(W, R["t$l.ln2g"]), c.m2[l], c.r2[l])
        @. dx2 += dx

        # --- atencion (unidades (seq,cabeza) en paralelo; BLAS=1 dentro) ---
        dm = dx2
        mul!(reshape(view(g, R["t$l.wo"]), d, d), transpose(c.ao[l]), dm)
        dao = dm * adjoint(reshape(view(W, R["t$l.wo"]), d, d))    # [M,d]

        dqkv = Matrix{Float32}(undef, M, 3d)
        _attn_bwd!(dqkv, dao, c.qkv[l], d, H, T, nb, Sbuf, Abuf, Gmbuf)
        dn1 = dqkv * adjoint(reshape(view(W, R["t$l.wqkv"]), d, 3d))
        mul!(reshape(view(g, R["t$l.wqkv"]), d, 3d), transpose(c.n1[l]), dqkv)
        _ln_param_grads!(view(g, R["t$l.ln1g"]), view(g, R["t$l.ln1b"]), dn1, c.xin[l], c.m1[l], c.r1[l])

        dx = similar(dn1); fill!(dx, 0.0f0)
        _ln_bwd!(dx, dn1, c.xin[l], view(W, R["t$l.ln1g"]), c.m1[l], c.r1[l])
        @. dx += dx2
    end

    # lado entrada: cada embedding usado como input recibe su dx (tied emb)
    @inbounds for s in 1:nb, t in 1:T
        lab = tl[s][t]
        mk = (s - 1) * T + t
        for k in 1:d
            demb[k, lab] += dx[mk, k]
        end
    end
    return nothing
end

# ----------------------------- AdamW -----------------------------------

mutable struct AdamOpt
    lr::Float32
    b1::Float32
    b2::Float32
    eps::Float32
    wd::Float32
    t::Int
    mm::Vector{Float32}
    vv::Vector{Float32}
end

AdamOpt(lr::Real, n::Int) = AdamOpt(Float32(lr), 0.9f0, 0.95f0, 1.0f-8, 0.1f0, 0,
                                    zeros(Float32, n), zeros(Float32, n))

# paso AdamW sobre un rango (trozo) — cada elemento es independiente
@inline function _adam_chunk!(m::Model, g, opt::AdamOpt, r::UnitRange{Int}, w::Bool,
                              bc1::Float32, bc2::Float32)
    @inbounds for i in r
        gi = g[i]
        opt.mm[i] = opt.b1 * opt.mm[i] + (1 - opt.b1) * gi
        opt.vv[i] = opt.b2 * opt.vv[i] + (1 - opt.b2) * gi * gi
        mh = opt.mm[i] / bc1
        vh = opt.vv[i] / bc2
        p = m.p[i]
        upd = mh / (sqrt(vh) + opt.eps)
        if w
            upd += opt.wd * p
        end
        m.p[i] = p - opt.lr * upd
    end
    return nothing
end

"""
    adam_step!(m, g, opt, names)

Paso AdamW sobre los rangos de `names` (decoupled weight decay solo en `*.w*`).
Rangos grandes se parten en trozos y se reparten entre hilos.
"""
function adam_step!(m::Model, g, opt::AdamOpt, names)
    opt.t += 1
    bc1 = 1 - opt.b1^opt.t
    bc2 = 1 - opt.b2^opt.t
    nth = Threads.nthreads()
    for name in names
        r = m.ranges[name]
        w = occursin(".w", name)
        if nth > 1 && length(r) > 262_144
            cs = _csplit(length(r), nth)
            i0 = first(r) - 1
            Threads.@threads for ci in eachindex(cs)
                _adam_chunk!(m, g, opt, i0 .+ cs[ci], w, bc1, bc2)
            end
        else
            _adam_chunk!(m, g, opt, r, w, bc1, bc2)
        end
    end
    return nothing
end

# --------------------------- save / load -------------------------------

function save_model(path::AbstractString, m::Model)
    open(path, "w") do io
        write(io, UInt8[0x49, 0x52, 0x49, 0x53, 0x4D, 0x44, 0x4C, 0x00])  # "IRISMDL\0"
        write(io, UInt32(1))
        for f in (:dim, :n_layers, :n_heads, :ffn, :vocab, :ctx, :n_hojas)
            write(io, UInt32(getfield(m.cfg, f)))
        end
        write(io, UInt32(m.cfg.seed))
        write(io, Int64(length(m.p)))
        write(io, m.p)
    end
    return path
end

function load_model(path::AbstractString)
    open(path) do io
        magic = read(io, 8)
        @assert magic == UInt8[0x49, 0x52, 0x49, 0x53, 0x4D, 0x44, 0x4C, 0x00] "no es IRISMDL"
        @assert read(io, UInt32) == UInt32(1) "versión desconocida"
        cfg = ModelConfig(Int(read(io, UInt32)), Int(read(io, UInt32)), Int(read(io, UInt32)),
                          Int(read(io, UInt32)), Int(read(io, UInt32)), Int(read(io, UInt32)),
                          Int(read(io, UInt32)), read(io, UInt32))
        n = read(io, Int64)
        p = Vector{Float32}(undef, n)
        read!(io, p)
        names, ranges, total = layout(cfg)
        @assert total == n "layout no coincide"
        return Model(cfg, names, ranges, p)
    end
end

end # module
