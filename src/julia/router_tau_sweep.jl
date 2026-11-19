# router_tau_sweep.jl — barrido de tau sobre el artifact guardado (sin entrenar)
# Uso: julia router_tau_sweep.jl [datos.tsv] [artifacto.irrtr]
# Reutiliza los helpers de router_test.jl (incluible sin ejecutar main).

include(joinpath(@__DIR__, "router_test.jl"))
using .BPE, .IrisRouter, Printf, Random

raiz = normpath(@__DIR__, "..", "..")
tok = BPE.load_tok(joinpath(raiz, "assets", "iris.tok"))
datafile = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(raiz, "data", "router_data_f4c.tsv")
artpath = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(raiz, "models", "router_v0.irrtr")

rows = read_tsv(datafile)
names, freq = class_order([r[1] for r in rows])
lab2id = Dict(n => i for (i, n) in enumerate(names))
data = [(lab2id[lab], tok1(tok, tx, 256)) for (lab, tx) in rows]

rng = Xoshiro(20261117)
idx = shuffle(rng, collect(1:length(data)))
n1 = round(Int, 0.70 * length(idx))
n2 = round(Int, 0.85 * length(idx))
va = data[idx[(n1 + 1):n2]]
te = data[idx[(n2 + 1):end]]

rt = load_router(artpath)
@printf("artifact: %s\n", artpath)
@printf("tau guardado %.2f | %d clases | %d params | val n=%d test n=%d\n",
        rt.tau, length(rt.names), length(rt.p), length(va), length(te))

function op(ds, tau)
    cov = 0; ok = 0
    for (l, t) in ds
        rp = router_predict(rt, t, Int[]; tau = Float32(tau))
        if rp.id_ruta != 0
            cov += 1
            (Int(rp.id_ruta) + 1 == l) && (ok += 1)
        end
    end
    return (cov, ok)
end

@printf("%-5s %-26s %-26s\n", "tau", "val", "test")
for tau in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    ov = op(va, tau)
    ot = op(te, tau)
    pv = ov[1] == 0 ? NaN : ov[2] / ov[1]
    pt = ot[1] == 0 ? NaN : ot[2] / ot[1]
    marca = (!isnan(pv) && !isnan(pt) && pv == 1.0 && pt == 1.0) ? "  <-- exacta val+test" : ""
    @printf("%.2f  cob %3d/%3d prec %.3f  cob %3d/%3d prec %.3f%s\n",
            tau, ov[1], length(va), pv, ot[1], length(te), pt, marca)
end
