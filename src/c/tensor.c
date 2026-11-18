/* =====================================================================
   IRIS · Núcleo de tensores (F2) — cuantización p4-one
   ---------------------------------------------------------------------
   p4-one: cada parámetro = 1 peso base cuantizado a 4 bits
           + 4 sub-parámetros de 2 bits que NUNCA se exponen en
           inferencia normal. Solo bajo estrés (muchas activaciones
           acumuladas del bloque) se libera temporalmente la
           corrección que codifican los sub-parámetros.
   - Escala fp16 por bloque de 32 parámetros (sin subnormales:
     las escalas se clampan a >= 1e-3).
   - Sub-parámetros: 2 bits cada uno (4 por parámetro, 1 byte total),
     con escala fp16 propia por bloque.
   - Estrés: contador uint8 por bloque, saturante, +1 por pase GEMM,
     decae en cada tick. Bloque "caliente" = stress >= umbral.
   - Hot fusion: corrección = sub_scale * media(s_i - 1.5), s_i en {0..3}
     → 13 niveles de refinamiento (paso 0.25 * sub_scale). Es el
     sustrato del adaptador tipo LoRA de /ent-my (FEEDBACK_SPEC §2).
   - Determinista: sin RNG interno, sin hilos en v0.1.
   Test:   tensor_test.c hace #include "tensor.c"
           gcc -O2 -std=c11 -o tensor_test.exe tensor_test.c -lm
   ===================================================================== */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Q4_BLOCK        32    /* parámetros por bloque de escala */
#define SUBS_PER_PARAM  4     /* p4-one: 4 sub-parámetros por parámetro */
#define STRESS_SAT      255   /* saturación del contador de estrés */
#define SCALE_MIN       1e-3f /* mínimo normal en fp16 con margen */
#define SCALE_MAX       60000.0f

typedef uint16_t f16; /* patrón de bits fp16 */

/* ---------------- conversión fp16 <-> f32 (por bits) ---------------- */

static float f16_to_f32(f16 h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man  = (uint32_t)(h & 0x03FFu);
    uint32_t bits;
    if (exp == 0) {
        bits = sign; /* ±0; subnormales flushed (escalas clamp >= 1e-3) */
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13); /* inf / nan */
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13); /* 127-15=112 */
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static f16 f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e    = (int32_t)((x >> 23) & 0xFFu);
    uint32_t m    = x & 0x007FFFFFu;
    if (e == 0xFF) return (f16)(sign | 0x7C00u);          /* inf/nan -> inf */
    int32_t he = e - 127 + 15;
    if (he >= 31) return (f16)(sign | 0x7C00u);           /* overflow -> inf */
    if (he <= 0)  return (f16)sign;                       /* underflow -> 0  */
    return (f16)(sign | ((uint32_t)he << 10) | (m >> 13)); /* truncado: determinista */
}

/* ----------------------------- tensor ------------------------------ */

typedef struct {
    int32_t  rows, cols;   /* W: [rows, cols] (rows=in_features, cols=out_features) */
    int32_t  n;            /* rows * cols */
    int32_t  nblocks;      /* ceil(n / Q4_BLOCK) */
    uint8_t *q;            /* base 4-bit empaquetada: 2 nibbles por byte */
    f16     *scale;        /* escala fp16 por bloque */
    uint8_t *sub;          /* 4 sub-parámetros de 2 bits por parámetro: n bytes */
    f16     *sub_scale;    /* escala fp16 de sub-parámetros por bloque */
    uint8_t *stress;       /* contador de estrés por bloque (saturante) */
    uint8_t *hot;          /* 1 si el bloque está fusionado en hotf */
    float   *hotf;         /* corrección f32 por parámetro (0.0 = sin fusionar) */
} IrisQ4;

void iris_q4_free(IrisQ4 *t); /* adelantada: init la usa en error */

IrisQ4 *iris_q4_init(int32_t rows, int32_t cols)
{
    IrisQ4 *t = (IrisQ4 *)calloc(1, sizeof(IrisQ4));
    if (!t) return NULL;
    t->rows = rows;
    t->cols = cols;
    t->n    = rows * cols;
    t->nblocks = (t->n + Q4_BLOCK - 1) / Q4_BLOCK;
    t->q        = (uint8_t *)calloc((size_t)((t->n + 1) / 2), 1);
    t->scale    = (f16 *)calloc((size_t)t->nblocks, sizeof(f16));
    t->sub      = (uint8_t *)calloc((size_t)t->n, 1);
    t->sub_scale= (f16 *)calloc((size_t)t->nblocks, sizeof(f16));
    t->stress   = (uint8_t *)calloc((size_t)t->nblocks, 1);
    t->hot      = (uint8_t *)calloc((size_t)t->nblocks, 1);
    t->hotf     = (float *)calloc((size_t)t->n, sizeof(float));
    if (!t->q || !t->scale || !t->sub || !t->sub_scale || !t->stress || !t->hot || !t->hotf) {
        iris_q4_free(t);
        return NULL;
    }
    return t;
}

void iris_q4_free(IrisQ4 *t)
{
    if (!t) return;
    free(t->q); free(t->scale); free(t->sub); free(t->sub_scale);
    free(t->stress); free(t->hot); free(t->hotf);
    free(t);
}

/* cuantiza una matriz f32 -> base q4 + sub-parámetros 2-bit */
void iris_q4_quantize(IrisQ4 *t, const float *w)
{
    for (int b = 0; b < t->nblocks; b++) {
        int start = b * Q4_BLOCK;
        int end   = start + Q4_BLOCK;
        if (end > t->n) end = t->n;

        /* escala base: amax/7 cubre el rango de nibbles [-8..7] */
        float amax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = fabsf(w[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 7.0f;
        if (scale < SCALE_MIN) scale = SCALE_MIN;
        if (scale > SCALE_MAX) scale = SCALE_MAX;
        t->scale[b] = f32_to_f16(scale);
        float s = f16_to_f32(t->scale[b]); /* escala real tras el clamp */

        /* cuantizar base y medir el residuo */
        float rmax = 0.0f;
        for (int i = start; i < end; i++) {
            float v = w[i] / s;
            int qi = (int)floorf(v + 0.5f);
            if (qi < -8)  qi = -8;
            if (qi > 7)   qi = 7;
            uint8_t nib = (uint8_t)(qi + 8);
            if (i & 1) t->q[i >> 1] |= (uint8_t)(nib << 4);
            else       t->q[i >> 1] = nib;
            float r = w[i] - s * (float)qi;
            float ra = fabsf(r);
            if (ra > rmax) rmax = ra;
        }

        /* sub-parámetros: codifican el residuo con 4 valores de 2 bits */
        float sscale = rmax / 1.5f;
        if (sscale < SCALE_MIN) sscale = SCALE_MIN;
        if (sscale > SCALE_MAX) sscale = SCALE_MAX;
        t->sub_scale[b] = f32_to_f16(sscale);
        float ss = f16_to_f32(t->sub_scale[b]);

        for (int i = start; i < end; i++) {
            float v = (w[i] - s * (float)((int)floorf(w[i] / s + 0.5f))) / ss;
            float tot_f = 4.0f * (v + 1.5f); /* suma objetivo de s_i, rango [0,12] */
            int tot = (int)lrintf(tot_f);
            if (tot < 0)  tot = 0;
            if (tot > 12) tot = 12;
            uint8_t s0 = (uint8_t)(tot > 3 ? 3 : tot); tot -= s0;
            uint8_t s1 = (uint8_t)(tot > 3 ? 3 : tot); tot -= s1;
            uint8_t s2 = (uint8_t)(tot > 3 ? 3 : tot); tot -= s2;
            uint8_t s3 = (uint8_t)tot;
            t->sub[i] = (uint8_t)(s0 | (s1 << 2) | (s2 << 4) | (s3 << 6));
        }
    }
}

/* peso base dequantizado del elemento idx */
static inline float q4_get(const IrisQ4 *t, int idx)
{
    uint8_t byte = t->q[idx >> 1];
    int nib = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
    return f16_to_f32(t->scale[idx / Q4_BLOCK]) * (float)(nib - 8);
}

/* dequantiza todo el tensor a f32 (solo base; para tests/debug) */
void iris_q4_dequantize_full(const IrisQ4 *t, float *out)
{
    for (int i = 0; i < t->n; i++) out[i] = q4_get(t, i);
}

/* llena la corrección f32 de un bloque desde sus sub-parámetros */
static void fill_hot_block(IrisQ4 *t, int b)
{
    float ss = f16_to_f32(t->sub_scale[b]);
    int start = b * Q4_BLOCK;
    int end   = start + Q4_BLOCK;
    if (end > t->n) end = t->n;
    for (int i = start; i < end; i++) {
        uint8_t sb = t->sub[i];
        int s0 = sb & 3, s1 = (sb >> 2) & 3, s2 = (sb >> 4) & 3, s3 = (sb >> 6) & 3;
        t->hotf[i] = ss * ((float)((s0 + s1 + s2 + s3) - 6) * 0.25f); /* centro 1.5: coincide con la codificacion */
    }
}

/* tick de ciclo: activa/evita la hot fusion según el umbral y decae estrés.
   Devuelve cuántos bloques quedaron calientes. */
int iris_q4_tick(IrisQ4 *t, uint8_t threshold, uint8_t decay)
{
    int nh = 0;
    for (int b = 0; b < t->nblocks; b++) {
        uint8_t s = t->stress[b];
        int now_hot = (s >= threshold);
        if (now_hot && !t->hot[b]) {
            fill_hot_block(t, b);
            t->hot[b] = 1;
        } else if (!now_hot && t->hot[b]) {
            int start = b * Q4_BLOCK;
            int end   = start + Q4_BLOCK;
            if (end > t->n) end = t->n;
            memset(t->hotf + start, 0, (size_t)(end - start) * sizeof(float));
            t->hot[b] = 0;
        }
        if (now_hot) nh++;
        if (s >= decay) t->stress[b] = (uint8_t)(s - decay);
        else            t->stress[b] = 0;
    }
    return nh;
}

/* out[M,N] = A[M,K] · W[K,N]; suma la corrección de bloques calientes.
   Cada bloque que participa en el pase recibe +1 de estrés. */
void iris_q4_gemm(const float *A, int M, int K, IrisQ4 *W, float *out, int N)
{
    if (W->rows != K || W->cols != N) {
        fprintf(stderr, "iris_q4_gemm: dims incompatibles W[%d,%d] vs A[M,%d]xN=%d\n",
                W->rows, W->cols, K, N);
        abort();
    }
    for (int m = 0; m < M; m++) {
        const float *arow = A + (size_t)m * K;
        float *orow = out + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                int idx = k * N + n;
                acc += arow[k] * q4_get(W, idx);
                if (W->hot[idx / Q4_BLOCK]) acc += arow[k] * W->hotf[idx];
            }
            orow[n] = acc;
        }
    }
    for (int b = 0; b < W->nblocks; b++)
        if (W->stress[b] < STRESS_SAT) W->stress[b]++;
}

/* ------------------------- ops auxiliares -------------------------- */

void iris_softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void iris_layernorm(float *x, int n, const float *gamma, const float *beta, float eps)
{
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        float v = (x[i] - mean) * inv;
        x[i] = v * gamma[i] + beta[i];
    }
}

/* E: [vocab, dim] cuantizada; ids: n_ids tokens; out: [n_ids, dim] */
void iris_embed_lookup(const IrisQ4 *E, const int *ids, int n_ids, float *out, int dim)
{
    for (int j = 0; j < n_ids; j++) {
        int tok = ids[j];
        if (tok < 0 || tok >= E->rows) {
            fprintf(stderr, "iris_embed_lookup: token %d fuera de vocab\n", tok);
            abort();
        }
        float *o = out + (size_t)j * dim;
        for (int d = 0; d < dim; d++)
            o[d] = q4_get(E, tok * dim + d);
    }
}

int iris_argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}
/* =====================================================================
   IRIS · Extensiones trunk (F2b) — posiciones sinusoidales, GELU,
   atención causal multi-cabeza y bloque transformer pre-LN.
   Determinista, sin hilos, sin RNG interno (mismo contrato que F2).
   ===================================================================== */

/* tabla sinusoidal: out[p*dim + d]; d par -> sin, d impar -> cos */
void iris_pos_sinusoidal(float *out, int n_pos, int dim)
{
    for (int p = 0; p < n_pos; p++) {
        for (int i = 0; i < dim / 2; i++) {
            float freq = expf(-logf(10000.0f) * (float)(2 * i) / (float)dim);
            float ang  = (float)p * freq;
            out[(size_t)p * dim + 2 * i]     = sinf(ang);
            out[(size_t)p * dim + 2 * i + 1] = cosf(ang);
        }
    }
}

/* GELU con aproximación tanh (determinista, sin libm extra) */
void iris_gelu(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float u = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}

/* atención causal multi-cabeza.
   Q,K,V: filas de T elementos con paso `stride`; el slice de la cabeza
   h ocupa D = H*Hd dentro de la fila. out: [T,D] contiguo.
   scratch: >= T floats. La máscara causal es el prefijo k <= t del
   softmax (no hay -inf: el prefijo ya es la distribución correcta). */
void iris_attn_causal(const float *Q, const float *K, const float *V,
                      int T, int stride, int D, int H, float *out, float *scratch)
{
    int Hd = D / H;
    float inv = 1.0f / sqrtf((float)Hd);
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < T; t++) {
            const float *q = Q + (size_t)t * stride + h * Hd;
            for (int k = 0; k <= t; k++) {
                const float *krow = K + (size_t)k * stride + h * Hd;
                float acc = 0.0f;
                for (int d = 0; d < Hd; d++) acc += q[d] * krow[d];
                scratch[k] = acc * inv;
            }
            iris_softmax(scratch, t + 1);
            float *o = out + (size_t)t * D + h * Hd;
            for (int d = 0; d < Hd; d++) o[d] = 0.0f;
            for (int k = 0; k <= t; k++) {
                float w = scratch[k];
                const float *vrow = V + (size_t)k * stride + h * Hd;
                for (int d = 0; d < Hd; d++) o[d] += w * vrow[d];
            }
        }
    }
}

/* floats de scratch que exige iris_block_preln */
size_t iris_block_scratch_floats(int T, int D, int F)
{
    /* lnA [T,D] + qkv [T,3D] + proj [T,D] + h1 [T,F] + lnB [T,D] + row [T] */
    return (size_t)T * (size_t)(6 * D + F + 1);
}

/* bloque transformer pre-LN (residual stream x [T,D] se modifica in situ).
   Wqkv [D,3D] · Wo [D,D] · W1 [D,F] · W2 [F,D]; ln g/b de tamaño D. */
void iris_block_preln(IrisQ4 *Wqkv, IrisQ4 *Wo,
                      IrisQ4 *W1, IrisQ4 *W2,
                      const float *ln1g, const float *ln1b,
                      const float *ln2g, const float *ln2b,
                      float *x, int T, int D, int H, int F, float *scratch)
{
    const size_t TD = (size_t)T * D;
    float *lnA  = scratch;                       /* [T,D] */
    float *qkv  = scratch + TD;                  /* [T,3D] */
    float *proj = qkv + (size_t)T * 3 * D;       /* [T,D] */
    float *h1   = proj + TD;                     /* [T,F] */
    float *lnB  = h1 + (size_t)T * F;            /* [T,D] */
    float *row  = lnB + TD;                      /* [T] */

    for (int t = 0; t < T; t++) {
        memcpy(lnA + (size_t)t * D, x + (size_t)t * D, sizeof(float) * (size_t)D);
        iris_layernorm(lnA + (size_t)t * D, D, ln1g, ln1b, 1e-5f);
    }
    iris_q4_gemm(lnA, T, D, Wqkv, qkv, 3 * D);
    iris_attn_causal(qkv, qkv + D, qkv + 2 * D, T, 3 * D, D, H, lnA, row);
    iris_q4_gemm(lnA, T, D, Wo, proj, D);
    for (size_t i = 0; i < TD; i++) x[i] += proj[i];

    for (int t = 0; t < T; t++) {
        memcpy(lnB + (size_t)t * D, x + (size_t)t * D, sizeof(float) * (size_t)D);
        iris_layernorm(lnB + (size_t)t * D, D, ln2g, ln2b, 1e-5f);
    }
    iris_q4_gemm(lnB, T, D, W1, h1, F);
    iris_gelu(h1, T * F);
    iris_q4_gemm(h1, T, F, W2, lnA, D);
    for (size_t i = 0; i < TD; i++) x[i] += lnA[i];
}