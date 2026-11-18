/* Test del núcleo de tensores IRIS (F2). Compilar desde iris\c:
   gcc -O2 -std=c11 -o tensor_test.exe tensor_test.c -lm   */
#include "tensor.c"

static uint32_t lcg_state = 42;
static float frand(void) /* determinista */
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 8) / 8388608.0f) * 2.0f - 1.0f; /* [-1,1] */
}

static int g_fail = 0;
#define CHECK(cond, name) do { \
    if (cond) printf("PASS  %s\n", name); \
    else { printf("FAIL  %s\n", name); g_fail++; } \
} while (0)

static void f32_gemm_ref(const float *A, int M, int K, const float *W, float *out, int N)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[m * K + k] * W[k * N + n];
            out[m * N + n] = acc;
        }
}

static float rel_error(const float *a, const float *b, int n)
{
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return (float)(sqrt(num) / sqrt(den));
}

int main(void)
{
    const int M = 8, K = 64, N = 32;

    /* datos deterministas */
    float *A  = malloc(sizeof(float) * M * K);
    float *Wf = malloc(sizeof(float) * K * N);
    for (int i = 0; i < M * K; i++) A[i] = frand();
    for (int i = 0; i < K * N; i++) Wf[i] = frand();

    float *ref  = malloc(sizeof(float) * M * N);
    float *o_q  = malloc(sizeof(float) * M * N);
    float *o_h  = malloc(sizeof(float) * M * N);
    f32_gemm_ref(A, M, K, Wf, ref, N);

    /* 1) cuantización + GEMM base */
    IrisQ4 *W = iris_q4_init(K, N);
    CHECK(W != NULL, "init tensor");
    iris_q4_quantize(W, Wf);
    memset(W->stress, 0, (size_t)W->nblocks);
    iris_q4_gemm(A, M, K, W, o_q, N);
    float e_base = rel_error(o_q, ref, M * N);
    printf("      error relativo base q4: %.4f\n", (double)e_base);
    CHECK(e_base < 0.05f, "GEMM base q4 error < 5%");

    /* 2) estrés: cada GEMM incrementa todos los bloques */
    CHECK(W->stress[W->nblocks - 1] == 1, "estres incrementado por GEMM");
    iris_q4_gemm(A, M, K, W, o_q, N);
    CHECK(W->stress[0] == 2, "estres acumulado en 2o pase");

    /* 3) hot fusion: umbral 2 -> todos calientes, error mejora */
    int nh = iris_q4_tick(W, 2, 0); /* sin decay aun */
    CHECK(nh == W->nblocks, "tick activa todos los bloques calientes");
    CHECK(W->hot[0] == 1, "bloque marcado caliente");
    /* los pases posteriores no deben volver a contar estrés para este check */
    memset(W->stress, 0, (size_t)W->nblocks);
    iris_q4_gemm(A, M, K, W, o_h, N);
    float e_hot = rel_error(o_h, ref, M * N);
    printf("      error relativo con hot fusion: %.4f\n", (double)e_hot);
    CHECK(e_hot < e_base, "hot fusion reduce el error");

    /* 4) decaimiento y eviction */
    nh = iris_q4_tick(W, 5, 1); /* stress=1 < umbral 5, decae a 0 */
    CHECK(nh == 0, "bloques se enfrían bajo umbral alto");
    CHECK(W->hot[0] == 0 && W->hotf[0] == 0.0f, "eviction limpia la correccion");

    /* 5) softmax / argmax / layernorm */
    float s[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    iris_softmax(s, 5);
    float ssum = 0.0f;
    for (int i = 0; i < 5; i++) ssum += s[i];
    CHECK(fabsf(ssum - 1.0f) < 1e-5f, "softmax suma 1");
    CHECK(iris_argmax(s, 5) == 4, "argmax");

    float x[6] = {1, 2, 3, 4, 5, 6};
    float g[6], bta[6];
    for (int i = 0; i < 6; i++) { g[i] = 1.0f; bta[i] = 0.0f; }
    iris_layernorm(x, 6, g, bta, 1e-5f);
    float xm = 0.0f;
    for (int i = 0; i < 6; i++) xm += x[i];
    xm /= 6.0f;
    CHECK(fabsf(xm) < 1e-4f, "layernorm media ~0");

    /* 6) embedding lookup: fila 3 == dequant del propio tensor */
    const int VOCAB = 100, DIM = 32;
    float *Ef = malloc(sizeof(float) * VOCAB * DIM);
    for (int i = 0; i < VOCAB * DIM; i++) Ef[i] = frand();
    IrisQ4 *E = iris_q4_init(VOCAB, DIM);
    iris_q4_quantize(E, Ef);
    int ids[3] = {7, 42, 99};
    float *emb = malloc(sizeof(float) * 3 * DIM);
    iris_embed_lookup(E, ids, 3, emb, DIM);
    int ok = 1;
    for (int d = 0; d < DIM; d++) {
        float want = q4_get(E, 42 * DIM + d);
        if (fabsf(emb[1 * DIM + d] - want) > 1e-6f) ok = 0;
    }
    CHECK(ok, "embedding lookup fila correcta");

    /* 7) posiciones sinusoidales */
    float *pos = malloc(sizeof(float) * 12);
    iris_pos_sinusoidal(pos, 3, 4);
    ok = fabsf(pos[0]) < 1e-6f && fabsf(pos[1] - 1.0f) < 1e-6f &&
         fabsf(pos[4] - 0.8414710f) < 1e-4f && fabsf(pos[5] - 0.5403023f) < 1e-4f &&
         fabsf(pos[8] - 0.9092974f) < 1e-4f && fabsf(pos[10] - 0.0199987f) < 1e-4f;
    CHECK(ok, "posiciones sinusoidales (sin/cos)");
    free(pos);

    /* 8) gelu (aprox tanh) */
    float gv[3] = {0.0f, 1.0f, -1.0f};
    iris_gelu(gv, 3);
    CHECK(fabsf(gv[0]) < 1e-6f && fabsf(gv[1] - 0.8412f) < 1e-3f &&
          fabsf(gv[2] + 0.1588f) < 1e-3f, "gelu tanh-aprox");

    /* 9) atención causal: fila t solo ve k<=t */
    {
        int T = 3, D = 2, H = 1;
        float Q[9]  = {1,0, 1,0, 1,0};
        float Kk[9] = {1,0, 2,0, 0,1};
        float V[9]  = {1,1, 1,1, 5,5};
        float out[6], scr[3];
        iris_attn_causal(Q, Kk, V, T, D, D, H, out, scr);
        ok = fabsf(out[0] - 1) < 1e-5f && fabsf(out[1] - 1) < 1e-5f &&
             fabsf(out[2] - 1) < 1e-5f && fabsf(out[3] - 1) < 1e-5f;
        float w0 = 0.2839947f, w1 = 0.5759747f; /* softmax de {1,2,0} escalado por 1/sqrt(2) */
        ok = ok && fabsf(out[4] - (w0 + w1 + (1.0f - w0 - w1) * 5.0f)) < 1e-3f;
        CHECK(ok, "atencion causal (fila t solo ve k<=t)");
    }

    /* 10) bloque pre-LN: residuo puro con pesos cero */
    {
        int T = 2, D = 8, H = 2, F = 16;
        IrisQ4 *wq = iris_q4_init(D, 3 * D), *wo_ = iris_q4_init(D, D);
        IrisQ4 *w1_ = iris_q4_init(D, F), *w2_ = iris_q4_init(F, D);
        float *zq = calloc((size_t)D * 3 * D, 4), *zo = calloc((size_t)D * D, 4);
        float *z1 = calloc((size_t)D * F, 4), *z2 = calloc((size_t)F * D, 4);
        iris_q4_quantize(wq, zq); iris_q4_quantize(wo_, zo);
        iris_q4_quantize(w1_, z1); iris_q4_quantize(w2_, z2);
        float g1[8], b1[8], g2[8], b2[8];
        for (int i = 0; i < D; i++) { g1[i] = 1.0f; b1[i] = 0.0f; g2[i] = 1.0f; b2[i] = 0.0f; }
        float xb[16] = {0.5f, -1.0f, 2.0f, 0.0f, 1.0f, -2.0f, 0.25f, -0.75f,
                        3.0f, 1.5f, -0.5f, 2.5f, -1.25f, 0.75f, 1.0f, -3.0f};
        float xref[16];
        memcpy(xref, xb, sizeof(xb));
        float *scr = malloc(sizeof(float) * iris_block_scratch_floats(T, D, F));
        iris_block_preln(wq, wo_, w1_, w2_, g1, b1, g2, b2, xb, T, D, H, F, scr);
        ok = 1;
        for (int i = 0; i < T * D; i++) if (fabsf(xb[i] - xref[i]) > 1e-5f) ok = 0;
        CHECK(ok, "bloque pre-LN con pesos cero = residuo puro");

        /* atención activa + Wo/W2 en cero -> sigue siendo residuo */
        IrisQ4 *wq2 = iris_q4_init(D, 3 * D);
        for (int i = 0; i < D * 3 * D; i++) zq[i] = frand();
        iris_q4_quantize(wq2, zq);
        memcpy(xb, xref, sizeof(xb));
        iris_block_preln(wq2, wo_, w1_, w2_, g1, b1, g2, b2, xb, T, D, H, F, scr);
        ok = 1;
        for (int i = 0; i < T * D; i++) if (fabsf(xb[i] - xref[i]) > 1e-5f) ok = 0;
        CHECK(ok, "atencion activa + proyeccion cero = residuo");
        free(scr); free(zq); free(zo); free(z1); free(z2);
        iris_q4_free(wq); iris_q4_free(wo_); iris_q4_free(w1_); iris_q4_free(w2_);
        iris_q4_free(wq2);
    }
    free(A); free(Wf); free(ref); free(o_q); free(o_h);
    free(Ef); free(emb);
    iris_q4_free(W); iris_q4_free(E);

    printf(g_fail == 0 ? "\nTODO OK\n" : "\nFALLOS: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
