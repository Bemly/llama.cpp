#include "ggml-quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void turbo_cpu_fwht_inverse(float * x, int group_size);

static uint32_t rng_state = 0x12345678u;

static float next_value(void) {
    rng_state = rng_state*1664525u + 1013904223u;
    return ((float) (rng_state >> 8)/(float) 0x00ffffffu)*2.0f - 1.0f;
}

static void inverse_groups(float * values, int n) {
    for (int i = 0; i < n; i += 128) {
        turbo_cpu_fwht_inverse(values + i, 128);
    }
}

static float cosine(const float * a, const float * b, int n) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += (double) a[i]*b[i];
        na += (double) a[i]*a[i];
        nb += (double) b[i]*b[i];
    }
    return na > 0.0 && nb > 0.0 ? (float) (dot/sqrt(na*nb)) : 1.0f;
}

static float relative_norm_error(const float * a, const float * b, int n) {
    double na = 0.0;
    double nb = 0.0;
    for (int i = 0; i < n; ++i) {
        na += (double) a[i]*a[i];
        nb += (double) b[i]*b[i];
    }
    return na > 0.0 ? (float) fabs(sqrt(nb/na) - 1.0) : (float) sqrt(nb);
}

static void fill_values(float * values, int n, int pattern) {
    for (int i = 0; i < n; ++i) {
        switch (pattern) {
            case 0: values[i] = 0.0f; break;
            case 1: values[i] = sinf((float) i*0.071f)*10.0f; break;
            case 2: values[i] = next_value(); break;
            default: values[i] = (i % 29 == 0) ? 25.0f : next_value()*0.05f; break;
        }
    }
}

static int test_turbo3(int n, int pattern) {
    const size_t bytes = (size_t) (n/QK_TURBO3)*sizeof(block_turbo3_0);
    float * input = (float *) malloc((size_t) n*sizeof(float));
    float * output = (float *) malloc((size_t) n*sizeof(float));
    block_turbo3_0 * q0 = (block_turbo3_0 *) calloc(1, bytes);
    block_turbo3_0 * q1 = (block_turbo3_0 *) calloc(1, bytes);
    if (!input || !output || !q0 || !q1) return 1;

    fill_values(input, n, pattern);
    quantize_row_turbo3_0_ref(input, q0, n);
    quantize_row_turbo3_0_ref(input, q1, n);
    dequantize_row_turbo3_0(q0, output, n);
    inverse_groups(output, n);

    const float cos = cosine(input, output, n);
    const float norm_err = relative_norm_error(input, output, n);
    const int failed = memcmp(q0, q1, bytes) != 0 || cos < 0.90f || norm_err > 0.002f;
    if (failed) fprintf(stderr, "turbo3 n=%d pattern=%d cosine=%g norm_err=%g\n", n, pattern, cos, norm_err);

    free(q1);
    free(q0);
    free(output);
    free(input);
    return failed;
}

static int test_turbo4(int n, int pattern) {
    const size_t bytes = (size_t) (n/QK_TURBO4)*sizeof(block_turbo4_0);
    float * input = (float *) malloc((size_t) n*sizeof(float));
    float * output = (float *) malloc((size_t) n*sizeof(float));
    block_turbo4_0 * q0 = (block_turbo4_0 *) calloc(1, bytes);
    block_turbo4_0 * q1 = (block_turbo4_0 *) calloc(1, bytes);
    if (!input || !output || !q0 || !q1) return 1;

    fill_values(input, n, pattern);
    quantize_row_turbo4_0_ref(input, q0, n);
    quantize_row_turbo4_0_ref(input, q1, n);
    dequantize_row_turbo4_0(q0, output, n);
    inverse_groups(output, n);

    const float cos = cosine(input, output, n);
    const float norm_err = relative_norm_error(input, output, n);
    const int failed = memcmp(q0, q1, bytes) != 0 || cos < 0.97f || norm_err > 0.002f;
    if (failed) fprintf(stderr, "turbo4 n=%d pattern=%d cosine=%g norm_err=%g\n", n, pattern, cos, norm_err);

    free(q1);
    free(q0);
    free(output);
    free(input);
    return failed;
}

int main(void) {
    int failed = 0;
    const int dimensions[] = {128, 256, 384, 512, 640};
    for (size_t i = 0; i < sizeof(dimensions)/sizeof(dimensions[0]); ++i) {
        const int n = dimensions[i];
        for (int pattern = 0; pattern < 4; ++pattern) {
            failed += test_turbo3(n, pattern);
            failed += test_turbo4(n, pattern);
        }
    }
    if (failed) {
        fprintf(stderr, "%d TurboQuant checks failed\n", failed);
        return 1;
    }
    printf("TurboQuant CPU suite passed\n");
    return 0;
}
