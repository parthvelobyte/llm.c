#define TESTING
#include "../train_gpt2.c"

#include <float.h>

static volatile float sink;

static uint32_t rng_u32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float rng_f32(uint64_t *state) {
    return ((rng_u32(state) >> 8) / 8388608.0f) - 1.0f;
}

static void build_synthetic_model(GPT2 *model, int L, int NH, int C, int V, int Vp, int maxT) {
    memset(model, 0, sizeof(*model));
    model->config.max_seq_len = maxT;
    model->config.vocab_size = V;
    model->config.padded_vocab_size = Vp;
    model->config.num_layers = L;
    model->config.num_heads = NH;
    model->config.channels = C;
    fill_in_parameter_sizes(model->param_sizes, model->config);
    model->num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) model->num_parameters += model->param_sizes[i];
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    uint64_t st = 0x123456789abcdef0ULL ^ (uint64_t)L << 48 ^ (uint64_t)C << 16 ^ (uint64_t)V;
    for (size_t i = 0; i < model->num_parameters; i++) {
        model->params_memory[i] = 0.02f * rng_f32(&st);
    }
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < C; i++) {
            model->params.ln1w[l*C + i] = 1.0f + 0.01f * rng_f32(&st);
            model->params.ln2w[l*C + i] = 1.0f + 0.01f * rng_f32(&st);
        }
    }
    for (int i = 0; i < C; i++) model->params.lnfw[i] = 1.0f + 0.01f * rng_f32(&st);
    model->mean_loss = -1.0f;
}

static void make_tokens(int *x, int *y, int B, int T, int V, uint64_t seed) {
    uint64_t st = seed;
    for (int i = 0; i < B*T; i++) {
        x[i] = (int)(rng_u32(&st) % (uint32_t)V);
        y[i] = (int)(rng_u32(&st) % (uint32_t)V);
    }
}

static void attention_backward_ref(float* dinp, float* dpreatt, float* datt,
                                   float* dout, float* inp, float* att,
                                   int B, int T, int C, int NH) {
    int C3 = C*3;
    int hs = C / NH;
    float scale = 1.f / sqrtf(hs);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C;
                    for (int i = 0; i < hs; i++) {
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

static int check_attention_backward(void) {
    const int B = 2, T = 17, C = 48, NH = 6;
    const int C3 = 3*C;
    size_t qkvn = (size_t)B*T*C3;
    size_t attn = (size_t)B*NH*T*T;
    size_t outn = (size_t)B*T*C;
    float *inp = (float*)mallocCheck(qkvn * sizeof(float));
    float *att = (float*)calloc(attn, sizeof(float));
    float *dout = (float*)mallocCheck(outn * sizeof(float));
    float *dinp0 = (float*)calloc(qkvn, sizeof(float));
    float *dinp1 = (float*)calloc(qkvn, sizeof(float));
    float *dpreatt0 = (float*)calloc(attn, sizeof(float));
    float *dpreatt1 = (float*)calloc(attn, sizeof(float));
    float *datt0 = (float*)calloc(attn, sizeof(float));
    float *datt1 = (float*)calloc(attn, sizeof(float));
    uint64_t st = 0x3141592653589793ULL;
    for (size_t i = 0; i < qkvn; i++) inp[i] = 0.25f * rng_f32(&st);
    for (size_t i = 0; i < outn; i++) dout[i] = 0.25f * rng_f32(&st);
    for (int b = 0; b < B; b++) for (int h = 0; h < NH; h++) for (int t = 0; t < T; t++) {
        float sum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float v = 0.1f + (float)(rng_u32(&st) % 1000) / 1000.0f;
            att[((b*NH + h)*T + t)*T + t2] = v;
            sum += v;
        }
        for (int t2 = 0; t2 <= t; t2++) att[((b*NH + h)*T + t)*T + t2] /= sum;
    }
    attention_backward_ref(dinp0, dpreatt0, datt0, dout, inp, att, B, T, C, NH);
    attention_backward(dinp1, dpreatt1, datt1, dout, inp, att, B, T, C, NH);
    float maxdiff = 0.0f;
    for (size_t i = 0; i < qkvn; i++) { float d = fabsf(dinp0[i] - dinp1[i]); if (d > maxdiff) maxdiff = d; }
    for (size_t i = 0; i < attn; i++) {
        float d = fabsf(dpreatt0[i] - dpreatt1[i]); if (d > maxdiff) maxdiff = d;
        d = fabsf(datt0[i] - datt1[i]); if (d > maxdiff) maxdiff = d;
    }
    free(inp); free(att); free(dout); free(dinp0); free(dinp1); free(dpreatt0); free(dpreatt1); free(datt0); free(datt1);
    if (maxdiff > 2e-5f) {
        printf("ATTENTION_BACKWARD maxdiff %.9g exceeds tolerance\n", maxdiff);
        return 1;
    }
    printf("ATTENTION_BACKWARD maxdiff %.9g OK\n", maxdiff);
    return 0;
}

static int run_model(int B, int T, int C, int NH, int L, int V, int Vp, int steps, int verbose) {
    GPT2 model;
    build_synthetic_model(&model, L, NH, C, V, Vp, T);
    int *x = (int*)mallocCheck((size_t)B*T*sizeof(int));
    int *y = (int*)mallocCheck((size_t)B*T*sizeof(int));
    make_tokens(x, y, B, T, V, 0x42424242ULL ^ (uint64_t)B << 32 ^ (uint64_t)T);
    float acc = 0.0f;
    for (int s = 0; s < steps; s++) {
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        acc += model.mean_loss;
    }
    sink = acc;
    int ok = isfinite(acc) && acc > 0.0f;
    if (verbose) printf("MODEL_CHECK B=%d T=%d C=%d NH=%d L=%d V=%d steps=%d loss_sum=%.9g %s\n", B, T, C, NH, L, V, steps, acc, ok ? "OK" : "BAD");
    free(x); free(y); gpt2_free(&model);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s check|bench [B T C NH L V Vp steps]\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "check") == 0) {
        int rc = check_attention_backward();
        rc |= run_model(2, 16, 48, 6, 2, 257, 384, 1, 1);
        rc |= run_model(1, 19, 64, 4, 1, 193, 256, 1, 1);
        if (rc == 0) printf("CHECK OK\n");
        return rc;
    }
    if (strcmp(argv[1], "bench") == 0) {
        if (argc != 10) {
            fprintf(stderr, "bench needs B T C NH L V Vp steps\n");
            return 2;
        }
        int B = atoi(argv[2]);
        int T = atoi(argv[3]);
        int C = atoi(argv[4]);
        int NH = atoi(argv[5]);
        int L = atoi(argv[6]);
        int V = atoi(argv[7]);
        int Vp = atoi(argv[8]);
        int steps = atoi(argv[9]);
        if (B <= 0 || T <= 0 || C <= 0 || NH <= 0 || L <= 0 || V <= 0 || Vp < V || steps <= 0 || C % NH != 0) return 2;
        int rc = run_model(B, T, C, NH, L, V, Vp, steps, 0);
        if (rc == 0) printf("BENCH_OK\n");
        return rc;
    }
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
