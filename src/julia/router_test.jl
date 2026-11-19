# router_test.jl — Iris F3: test del router clasificador
# Uso: julia router_test.jl [datos.tsv] [salida.irrtr]   (desde iris/julia)
# 1) carga TSV de muestras, 2) forward independiente vs router_loss_grad!,
# 3) gradiente numerico vs analitico por grupo, 4) entrena 10 semillas con
# early-stopping por val + dropout IDF + oversampling de clases pequenas,
#    seleccion operativa: max cobertura val a tau=0.6 con
#    precision val >= 0.98 (fallback: min val loss); tau final calibrado por ARGS, 
# 5) accuracy honesta en test NUNCA visto (media de semillas),
# 6) metricas operativas por tau + fallback nonsense, 7) save/load del mejor.

include(joinpath(@__DIR__, "BPE.jl")); using .BPE
include(joinpath(@__DIR__, "router.jl")); using .IrisRouter
using Random, Printf, Statistics

const TOL_REL = 5e-3
const TOL_ABS = 1e-3

function read_tsv(path)
    rows = Vector{Tuple{String,String}}()
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        ix = findfirst('\t', line)
        ix === nothing && continue
        push!(rows, (String(line[1:prevind(line, ix)]), String(line[nextind(line, ix):end])))
    end
    return rows
end

function class_order(labels)
    freq = Dict{String,Int}()
    for l in labels; freq[l] = get(freq, l, 0) + 1; end
    rest = sort(collect(keys(freq)); by = x -> (-freq[x], x))
    names = vcat(["generica"], filter(!=("generica"), rest))
    return names, freq
end

function tok1(tok, tx, cap)
    ids = BPE.encode(tok, tx)
    t = Int.(ids) .+ 1
    length(t) > cap && (t = t[1:cap])
    isempty(t) && (t = [1])
    return t
end

# Forward escrito a mano, SIN usar las funciones del router: sirve para
# validar que router_forward/router_loss_grad! implementan lo declarado.
function ref_loss(rt, batch)
    d = rt.conf.dim; H = rt.conf.hidden; nr = rt.conf.n_rutas
    R = rt.ranges; p = rt.p
    emb = reshape(view(p, R["emb"]), d, rt.conf.vocab)
    PH = reshape(view(p, R["ph"]), d, d)
    W1 = reshape(view(p, R["w1"]), H, 2d)
    W2 = reshape(view(p, R["w2"]), nr, H)
    b1 = view(p, R["b1"]); b2 = view(p, R["b2"])
    loss = 0.0
    for (tm, th, lab) in batch
        m = vec(sum(view(emb, :, tm); dims = 2)) ./ Float32(length(tm))
        h = isempty(th) ? zeros(Float32, d) :
            vec(sum(view(emb, :, th); dims = 2)) ./ Float32(length(th))
        u = vcat(m, PH * h)
        z = W1 * u .+ b1
        a = @. 0.5f0 * z * (1.0f0 + tanh(0.7978845608f0 * (z + 0.044715f0 * z * z * z)))
        lg = W2 * a .+ b2
        loss += Float64(maximum(lg) - lg[lab] + log(sum(exp.(lg .- maximum(lg)))))
    end
    return loss / length(batch)
end

function grad_checks()
    noms = ["generica", "a", "b", "c", "d"]
    cfg = RouterConf(vocab = 64, dim = 8, hidden = 16, n_rutas = 5)
    rt = init_router(cfg, noms)
    rng = Xoshiro(7)
    batch = [
        (rand(rng, 1:64, 30), rand(rng, 1:64, 10), 3),
        (rand(rng, 1:64, 20), Int[], 1),
        (rand(rng, 1:64, 25), rand(rng, 1:64, 8), 5),
    ]
    g = zeros(Float32, length(rt.p))
    la = router_loss_grad!(rt, g, batch)
    lref = ref_loss(rt, batch)
    d1 = abs(la - lref) / max(abs(la), 1e-12)
    @printf("  forward indep: %.10f vs %.10f (rel %.2e) %s\n",
            la, lref, d1, d1 < 1e-5 ? "OK" : "FALLO")
    d1 < 1e-5 || error("forward del router no coincide con referencia independiente")

    h = 1.0f-3
    worst = Dict{String,Float32}()
    for name in sort(collect(keys(rt.ranges)))
        r = rt.ranges[name]
        wmax = 0.0f0
        for i in r
            pi_ = rt.p[i]
            rt.p[i] = pi_ + h; fp = ref_loss(rt, batch)
            rt.p[i] = pi_ - h; fm = ref_loss(rt, batch)
            rt.p[i] = pi_
            num = (fp - fm) / (2h)
            ana = g[i]
            err = abs(num - ana)
            denom = max(abs(num), abs(ana), 1.0f-6)
            okp = err <= TOL_ABS || err / denom <= TOL_REL
            if !okp
                @printf("    !! %s[%d] analitico %.6e numerico %.6e\n", name, i, ana, num)
                wmax = max(wmax, err / denom)
            end
        end
        worst[name] = wmax
    end
    for (n, w) in sort(collect(worst))
        @printf("  grad %-4s peor rel fallo: %.2e %s\n", n, w, w == 0 ? "OK" : "FALLO")
    end
    all(==(0.0f0), values(worst)) || error("gradiente router != numerico")
    println("  grad check router: OK")
end

# keep-prob por token segun IDF (tokens comunes caen mas; raros se quedan)
# df calculado SOLO sobre train original (sin oversample: no infla df)
function idf_keep(train)
    N = length(train)
    df = Dict{Int,Int}()
    for (l, t) in train
        for u in unique(t); df[u] = get(df, u, 0) + 1; end
    end
    keep = Dict{Int,Float32}()
    for (t, c) in df
        idf = log((N + 1) / (c + 1)) + 1.0
        keep[t] = 0.25f0 + 0.75f0 * min(idf / 3.0, 1.0)
    end
    return keep
end

function drop_batch(batch0, keep, rng)
    out = Vector{Tuple{Vector{Int},Vector{Int},Int}}(undef, length(batch0))
    for (i, (tm, th, lab)) in enumerate(batch0)
        m = [t for t in tm if rand(rng) < get(keep, t, 1.0f0)]
        h = [t for t in th if rand(rng) < get(keep, t, 1.0f0)]
        isempty(m) && (m = [tm[rand(rng, 1:length(tm))]])
        out[i] = (m, h, lab)
    end
    return out
end

# oversampling sqrt-balanceado de clases pequenas
function oversample(tr)
    nc = Dict{Int,Int}()
    for (l, t) in tr; nc[l] = get(nc, l, 0) + 1; end
    maxc = maximum(values(nc))
    out = Tuple{Int,Vector{Int}}[]
    for s in tr
        k = clamp(round(Int, sqrt(maxc / nc[s[1]])), 1, 4)
        for _ in 1:k; push!(out, s); end
    end
    return out
end

function train_one(cfg, names, batch0, keep, va_b, seed; MAXSTEPS = 3000, EVAL = 10, PATIENCE = 400)
    rt = init_router(cfg, names)
    opt = RouterAdam(0.003, length(rt.p))
    drng = Xoshiro(UInt32(seed))
    gv = zeros(Float32, length(rt.p))
    best_val = Inf; best_step = 0; best_p = copy(rt.p)
    op_cov = 0; op_ok = 0; op_step = 0; op_p = nothing
    step = 0
    while step < MAXSTEPS
        step += 1
        b = drop_batch(batch0, keep, drng)
        g = zeros(Float32, length(rt.p))
        loss = router_loss_grad!(rt, g, b)
        adam_step!(rt, g, opt)
        if step % EVAL == 0
            vloss = router_loss_grad!(rt, gv, va_b)
            if vloss < best_val - 1e-4
                best_val = vloss; best_step = step; best_p = copy(rt.p)
            elseif step - best_step >= PATIENCE
                break
            end
            # seguimiento operativo (tau=0.6): max cobertura val con prec >= 0.98
            cov = 0; ok = 0
            for (tm, th, lab) in va_b
                rp = router_predict(rt, tm, th; tau = 0.6f0)
                if rp.id_ruta != 0
                    cov += 1
                    (Int(rp.id_ruta) + 1 == lab) && (ok += 1)
                end
            end
            # precision val >= 0.98: snapshots bien calibrados (la exactitud
            # fina se fija despues con el tau almacenado del artifact)
            prec = cov == 0 ? 0.0 : ok / cov
            if prec >= 0.98 && cov > op_cov
                op_cov = cov; op_ok = ok; op_step = step; op_p = copy(rt.p)
            end
        end
    end
    if op_p !== nothing
        copyto!(rt.p, op_p)
        return (rt = rt, val = best_val, paso = op_step,
                op = (cov = op_cov, ok = op_ok))
    end
    copyto!(rt.p, best_p)
    return (rt = rt, val = best_val, paso = best_step, op = nothing)
end

function main()
    raiz = normpath(@__DIR__, "..", "..")
    tok = BPE.load_tok(joinpath(raiz, "assets", "iris.tok"))
    V = Int(BPE.vocab_size(tok))
    @printf("BPE vocab: %d\n", V)

    datafile = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(raiz, "data", "router_data_f4c.tsv")
    rows = read_tsv(datafile)
    @printf("filas TSV: %d\n", length(rows))
    names, freq = class_order([r[1] for r in rows])
    lab2id = Dict(n => i for (i, n) in enumerate(names))
    nr = length(names)
    @printf("clases: %d (generica + %d)\n", nr, nr - 1)

    data = Tuple{Int,Vector{Int}}[]
    maxtok = 0
    for (lab, tx) in rows
        t = tok1(tok, tx, 256)
        @assert maximum(t) <= V "id de token fuera de vocab"
        maxtok = max(maxtok, length(t))
        push!(data, (lab2id[lab], t))
    end
    @printf("tokens por muestra: max %d\n", maxtok)

    # split 70/15/15 determinista
    rng = Xoshiro(20261117)
    idx = shuffle(rng, collect(1:length(data)))
    n1 = round(Int, 0.70 * length(idx))
    n2 = round(Int, 0.85 * length(idx))
    tr = data[idx[1:n1]]
    va = data[idx[(n1 + 1):n2]]
    te = data[idx[(n2 + 1):end]]
    @printf("split: %d train / %d val / %d test\n", length(tr), length(va), length(te))

    println("[1/5] grad checks (conf pequena)")
    grad_checks()

    println("[2/5] entrenamiento x10 semillas (early-stop val + seleccion operativa, dropout IDF, oversample)")
    cfg = RouterConf(vocab = V, n_rutas = nr)
    nparams = length(init_router(cfg, names).p)
    @printf("params router: %d (%.2f MB fp32)\n", nparams, 4 * nparams / 1024^2)
    keep = idf_keep(tr)
    tr_os = oversample(tr)
    @printf("batch oversample: %d (de %d)\n", length(tr_os), length(tr))
    batch0 = [(t, Int[], l) for (l, t) in tr_os]
    va_b = [(t, Int[], l) for (l, t) in va]
    te_l = [(l, t) for (l, t) in te]

    accf(rt, ds) = count(findmax(router_forward(rt, t, Int[]))[2] == l for (l, t) in ds) / length(ds)
    gbest_p = nothing; gbest_seed = 0; gbest_cov = 0
    lb_val = Inf; lb_p = nothing; lb_seed = 0
    tests = Float64[]
    for seed in (11, 23, 37, 51, 69, 83, 97, 101, 113, 127)
        res = train_one(cfg, names, batch0, keep, va_b, seed)
        c = accf(res.rt, te_l)
        push!(tests, c)
        if res.op !== nothing
            @printf("  seed %2d: op paso %3d | val prec %.3f (%d/%d) cob | test %.3f (%d/%d)\n",
                    seed, res.paso, res.op.ok / res.op.cov, res.op.ok, res.op.cov,
                    c, round(Int, c * length(te)), length(te))
            if res.op.cov > gbest_cov
                gbest_p = copy(res.rt.p); gbest_seed = seed; gbest_cov = res.op.cov
            end
        else
            @printf("  seed %2d: parada paso %3d | val loss %.4f | (sin op) | test %.3f (%d/%d)\n",
                    seed, res.paso, res.val, c, round(Int, c * length(te)), length(te))
        end
        if res.val < lb_val
            lb_val = res.val; lb_p = copy(res.rt.p); lb_seed = seed
        end
    end
    if gbest_p === nothing
        gbest_p = lb_p; gbest_seed = lb_seed
        println("  (ninguna semilla con prec val >= 0.98 a tau=0.6; fallback a val loss)")
    end
    @printf("test argmax: media %.3f  min %.3f  max %.3f (n test=%d)\n",
            mean(tests), minimum(tests), maximum(tests), length(te))

    println("[3/5] mejor snapshot (seed $gbest_seed)")
    rt = init_router(cfg, names, length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : 0.60f0)
    copyto!(rt.p, gbest_p)
    for (nm, ds) in (("train", [(l, t) for (l, t) in tr]), ("val", va), ("test", te))
        c = count(findmax(router_forward(rt, t, Int[]))[2] == l for (l, t) in ds)
        @printf("  %-5s %.3f (%d/%d)\n", nm, c / length(ds), c, length(ds))
    end
    println("  por clase (test):")
    for (i, n) in enumerate(names)
        sub = [(l, t) for (l, t) in te if l == i]
        isempty(sub) && continue
        ck = count(findmax(router_forward(rt, t, Int[]))[2] == l for (l, t) in sub)
        @printf("    %-42s %2d/%2d\n", n, ck, length(sub))
    end

    println("[4/5] metricas operativas (esquema co-ayudador s3)")
    println("  cobertura/precision de rutas NO-genericas con p >= tau:")
    function operativa(ds, tau)
        cov = 0; ok = 0
        for (l, t) in ds
            rp = router_predict(rt, t, Int[]; tau = Float32(tau))
            if rp.id_ruta != 0
                cov += 1
                (Int(rp.id_ruta) + 1 == l) && (ok += 1)
            end
        end
        return (cov = cov, prec = cov == 0 ? NaN : ok / cov)
    end
    for tau in (0.5, 0.6, 0.7, 0.85)
        for (nm, ds) in (("val", va), ("test", te))
            o = operativa(ds, tau)
            if isnan(o.prec)
                @printf("  tau=%.2f %-4s cobertura %2d/%2d  precision n/a\n", tau, nm, o.cov, length(ds))
            else
                @printf("  tau=%.2f %-4s cobertura %2d/%2d (%.2f)  precision %.3f (%d/%d)\n",
                        tau, nm, o.cov, length(ds), o.cov / length(ds), o.prec,
                        round(Int, o.prec * o.cov), o.cov)
            end
        end
    end
    r = router_predict(rt, tok1(tok, "zzz qqq kjh xzz wf plt mm nn bb vv 87 34", 256), Int[])
    okf = r.id_ruta == 0 || r.p < rt.tau
    @printf("  nonsense -> id %d (%s) p=%.3f tau=%.2f %s\n",
            Int(r.id_ruta), r.nombre, r.p, rt.tau, okf ? "OK (fallback)" : "SIN fallback")
    pv = [router_predict(rt, t, Int[]).p for (l, t) in va]
    @printf("  p max media: val %.3f | nonsense %.3f\n", mean(pv), r.p)

    println("[5/5] save/load IRISRTR")
    path = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(raiz, "models", "router_v0.irrtr")
    save_router(path, rt)
    rt2 = load_router(path)
    @assert isequal(rt.p, rt2.p) "save/load no bit-exacto"
    @assert rt2.names == rt.names && rt2.tau == rt.tau "metadata no coincide"
    @printf("  %s: %d params, %d clases, roundtrip bit-exacto OK\n", path, length(rt2.p), length(rt2.names))

    println("TODO OK")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
