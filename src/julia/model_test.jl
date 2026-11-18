# model_test.jl — sanity del modelo F2b (correr desde iris\julia: julia model_test.jl)
include(joinpath(@__DIR__, "model.jl"))
using .IrisModel
using Printf

g_fail = 0
function check(cond, name)
    global g_fail
    if cond
        println("PASS  ", name)
    else
        println("FAIL  ", name); global g_fail += 1
    end
end

cfg = ModelConfig(dim = 16, n_layers = 2, n_heads = 2, ffn = 32, vocab = 64,
                  ctx = 48, n_hojas = 2, seed = 0x1234)
m = init_model(cfg)
println("parámetros (test): ", param_count(m))
st = sin_table(cfg)

tokens = [(i % (cfg.vocab - 2)) + 2 for i in 0:cfg.ctx - 1]

loss0, l1, c1 = forward_loss(m, tokens, st)
loss0b, l2, _ = forward_loss(m, tokens, st)
check(loss0 ≈ loss0b && l1 == l2, "forward determinista")
println(@sprintf "  loss inicial: %.4f (ln vocab = %.4f)" loss0 log(cfg.vocab))

# --- causalidad del modelo completo ---
tk_a = copy(tokens)
tk_b = copy(tokens); tk_b[end] = tk_b[end] == 9 ? 11 : 9
_, la_, _ = forward_loss(m, tk_a, st)
_, lb_, _ = forward_loss(m, tk_b, st)
check(la_[1:cfg.ctx-1, :] == lb_[1:cfg.ctx-1, :],
      "causalidad: cambiar el último token no toca logits pasados")

tk_b2 = copy(tokens); tk_b2[3] = tk_b2[3] == 5 ? 7 : 5
_, la2, _ = forward_loss(m, tk_a, st)
_, lb2, _ = forward_loss(m, tk_b2, st)
check(la2[1:2, :] == lb2[1:2, :] && any(la2[3:end, :] .!= lb2[3:end, :]),
      "causalidad: cambiar el token 3 afecta solo desde la posición 3")

# --- gradiente analítico vs numérico: top-8 + el mejor de cada grupo con gradiente ---
function check_grad(m, tokens, st, g, ktop)
    idxs = Int[]
    ord = sortperm(abs.(g), rev = true)
    for i in 1:ktop
        abs(g[ord[i]]) > 0 && push!(idxs, ord[i])
    end
    for name in m.names
        r = m.ranges[name]
        _, j = findmax(abs.(view(g, r)))
        idx = r[1] + j - 1
        if abs(g[idx]) > 1e-9 && !(idx in idxs)
            push!(idxs, idx)
        end
    end
    ok = true
    for idx in sort(idxs)
        eps = 1.0f-2
        old = m.p[idx]
        m.p[idx] = old + eps; lplus = forward_loss(m, tokens, st)[1]
        m.p[idx] = old - eps; lminus = forward_loss(m, tokens, st)[1]
        m.p[idx] = old
        num = (lplus - lminus) / (2eps)
        ana = g[idx]
        if abs(num) + abs(ana) < 1e-4
            println(@sprintf "  idx %5d: ambos ≈ 0 (ok)" idx)
            continue
        end
        rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
        println(@sprintf "  idx %5d: num=%+.6f ana=%+.6f rel=%.4f" idx num ana rel)
        ok = ok && (rel < 0.08)
    end
    return ok
end

g = zeros(Float32, length(m.p))
backward!(m, g, tokens, l1, c1)
check(check_grad(m, tokens, st, g, 8), "gradiente analítico ≈ numérico (top-8 + por grupo)")

# --- Adam memoriza el patrón ---
opt = AdamOpt(0.02f0, length(m.p))
for step in 1:80
    fill!(g, 0.0f0)
    l, lg, cc = forward_loss(m, tokens, st)
    backward!(m, g, tokens, lg, cc)
    adam_step!(m, g, opt, IrisModel.default_trainable(m))
end
lfinal, _, _ = forward_loss(m, tokens, st)
println(@sprintf "  loss tras 80 pasos de Adam: %.4f" lfinal)
check(lfinal < 0.5 * loss0, "Adam reduce la loss (memoriza el patrón)")

# --- save/load exacto ---
save_model("test_modelo.bin", m)
m3 = load_model("test_modelo.bin")
l3, _, _ = forward_loss(m3, tokens, st)
check(l3 == lfinal, "save/load roundtrip exacto")
rm("test_modelo.bin")

# --- trainable_fase1 ---
names = IrisModel.trainable_fase1(m, 1)
check(all(haskey(m.ranges, n) for n in names), "trainable_fase1: nombres válidos")

# --- tamaño y velocidad reales (fase 1) ---
cfg2 = ModelConfig()
println("inicializando modelo fase 1 (", cfg2.n_hojas, " hojas)…")
m2 = init_model(cfg2)
n2 = param_count(m2)
println("parámetros fase 1: ", n2, " (", round(n2 / 1e9, digits = 3), " B)")
st2 = sin_table(cfg2)
tk2 = [(i % (cfg2.vocab - 2)) + 2 for i in 0:255]
t = @elapsed (lreal, lg2, cc2) = forward_loss(m2, tk2, st2)
println(@sprintf "  forward real T=256: %.2f s → %.0f tok/s; loss=%.4f" t 256 / t lreal)
g2 = zeros(Float32, n2)
tb = @elapsed begin
    fill!(g2, 0.0f0)
    backward!(m2, g2, tk2, lg2, cc2)
end
tt = t + tb
println(@sprintf "  backward real T=256: %.2f s → paso completo %.2f s/seq ≈ %.0f tok/s (fwd+bwd)" tb tt 256 / tt)
println(@sprintf "  ≈ %.1f M tokens/día ≈ %.0f MB texto/día (3.5 B/token)" 256 / tt * 86400 / 1e6 256 / tt * 86400 * 3.5 / 1e6)
println(@sprintf "  RAM aprox de p: %.2f GB" (sizeof(m2.p) / 2^30))

println(g_fail == 0 ? "\nTODO OK" : "\nFALLOS: $g_fail")
exit(g_fail == 0 ? 0 : 1)
