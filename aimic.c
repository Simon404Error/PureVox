/* PureVox - AI microphone denoise tool
 * Copyright (C) 2024-2026 a2heng <752848283@qq.com>
 *
 * PureVox is licensed under the GNU General Public License v3.0 or
 * later (GPL-3.0-or-later).  See LICENSE for details.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The built-in AI models are NOT covered by the GPL; they are the
 * property of a2heng and may only be used with PureVox under
 * authorization.  See MODEL-LICENSE.md for details.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * PureVox pure-C audio engine: denoise/TSE/AEC/EQ/VAD/AGC/compressor + STFT + RingBuffer.
 * ONNX via C API (onnxruntime_c_api.h), FFT via pffft, resample via libsamplerate.
 * ctypes binding in aimic.py (loads libaimic.so); Python side has no numpy/torch dep.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "aimic.h"
#include "onnxruntime_c_api.h"
#include "pffft.h"
#include "samplerate.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
/* windows.h/windef.h define empty far/near macros (16-bit legacy) that swallow
 * our far param name and far_history_ identifiers and break compilation; clear. */
#undef far
#undef near
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const size_t HOP_LENGTH = 1024;
static const float  SAMPLE_RATE = 48000.0f;

/* ------------------------- cross-platform mutex ------------------------- */
#if defined(_WIN32)
typedef SRWLOCK pv_mutex_t;
static void pv_mutex_init(pv_mutex_t* m)   { InitializeSRWLock(m); }
static void pv_mutex_lock(pv_mutex_t* m)   { AcquireSRWLockExclusive(m); }
static void pv_mutex_unlock(pv_mutex_t* m) { ReleaseSRWLockExclusive(m); }
static void pv_mutex_destroy(pv_mutex_t* m){ (void)m; }
#else
#include <pthread.h>
typedef pthread_mutex_t pv_mutex_t;
static void pv_mutex_init(pv_mutex_t* m)   { pthread_mutex_init(m, NULL); }
static void pv_mutex_lock(pv_mutex_t* m)   { pthread_mutex_lock(m); }
static void pv_mutex_unlock(pv_mutex_t* m) { pthread_mutex_unlock(m); }
static void pv_mutex_destroy(pv_mutex_t* m){ pthread_mutex_destroy(m); }
#endif

/* ------------------------- small utilities ------------------------- */
static inline float clip_sample(float x) {
    if (isnan(x) || isinf(x)) return 0.0f;
    if (x > 1.0f) return 1.0f;
    if (x < -1.0f) return -1.0f;
    return x;
}

static void clip_buffer(float* data, size_t len) {
    for (size_t i = 0; i < len; ++i) data[i] = clip_sample(data[i]);
}

/* sqrt-Hann analysis window (torch.hann_window().pow(0.5) semantics) */
static void make_sqrt_hann(float* w, int n) {
    for (int i = 0; i < n; ++i) {
        float hann = 0.5f - 0.5f * (float)cos(2.0 * M_PI * i / n);
        w[i] = (float)sqrt(hann + 1e-10);
    }
}

/* growable float buffer (minimal equivalent of C++ std::vector<float>) */
typedef struct {
    float* data;
    size_t len;
    size_t cap;
} FVec;

static void fvec_init(FVec* v) { v->data = NULL; v->len = 0; v->cap = 0; }
static void fvec_free(FVec* v) { free(v->data); v->data = NULL; v->len = v->cap = 0; }
static void fvec_clear(FVec* v) { v->len = 0; }

static int fvec_reserve(FVec* v, size_t n) {
    if (n <= v->cap) return 0;
    size_t nc = v->cap ? v->cap : 64;
    while (nc < n) nc *= 2;
    float* nd = (float*)realloc(v->data, nc * sizeof(float));
    if (!nd) return -1;
    v->data = nd;
    v->cap = nc;
    return 0;
}

static void fvec_push(FVec* v, float x) {
    if (fvec_reserve(v, v->len + 1)) return;
    v->data[v->len++] = x;
}

static void fvec_append(FVec* v, const float* src, size_t n) {
    if (!n) return;
    if (fvec_reserve(v, v->len + n)) return;
    memcpy(v->data + v->len, src, n * sizeof(float));
    v->len += n;
}

static void fvec_fill(FVec* v, size_t n, float fill) {
    if (fvec_reserve(v, n)) return;
    for (size_t i = 0; i < n; ++i) v->data[i] = fill;
    v->len = n;
}

static void fvec_erase_front(FVec* v, size_t n) {
    if (n >= v->len) { v->len = 0; return; }
    memmove(v->data, v->data + n, (v->len - n) * sizeof(float));
    v->len -= n;
}

/* ------------------------- ONNX C API wrapper ------------------------- */
typedef struct {
    const OrtApi* api;
    OrtEnv* env;
    OrtSessionOptions* opts;
    OrtSession* session;
    OrtMemoryInfo* meminfo;
    OrtAllocator* allocator;
    char** input_names;
    char** output_names;
    size_t n_inputs;
    size_t n_outputs;
    int64_t** input_shapes;
    size_t* input_ndims;
    int effective_backend_;
    int backend_reason_;
} OnnxModel;

static int ort_ok(const OrtApi* api, OrtStatus* st) {
    if (st) {
        const char* msg = api->GetErrorMessage(st);
        fprintf(stderr, "aimic: onnxruntime error: %s\n", msg ? msg : "unknown");
        api->ReleaseStatus(st);
        return 0;
    }
    return 1;
}

#if defined(_WIN32)
static int ort_create_session_path(const OrtApi* api, OrtEnv* env, const char* path,
                                   OrtSessionOptions* opts, OrtSession** out) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    wchar_t* wpath = (wchar_t*)malloc((size_t)wlen * sizeof(wchar_t));
    if (!wpath) return 0;
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);
    OrtStatus* st = api->CreateSession(env, wpath, opts, out);
    free(wpath);
    return ort_ok(api, st);
}
#else
static int ort_create_session_path(const OrtApi* api, OrtEnv* env, const char* path,
                                   OrtSessionOptions* opts, OrtSession** out) {
    return ort_ok(api, api->CreateSession(env, path, opts, out));
}
#endif

/* CPU AVX support (gcc/clang builtin; mingw on Windows is gcc too) */
static int cpu_supports_avx(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx") ? 1 : 0;
#else
    return 0;
#endif
}

/* printed NPU fallback note once per process, not once per model */
static int g_npu_warned = 0;

/* Auto backend selection applied to every session:
 *   1) try an NPU-capable execution provider (OpenVINO on Linux / DirectML on
 *      Windows / CoreML on macOS) and use it when the runtime provides it;
 *   2) otherwise use the default CPU execution provider, whose MLAS kernels
 *      dispatch on CPUID to the best ISA available (AVX if the CPU has it,
 *      else SSE).  effective_backend_ reflects what will actually run and
 *      backend_reason_ why NPU is not in use (0 = NPU active). */
static int onnx_apply_backend(OnnxModel* m) {
    m->effective_backend_ = cpu_supports_avx() ? AIMIC_BACKEND_AVX : AIMIC_BACKEND_SSE;
    m->backend_reason_ = AIMIC_BACKEND_REASON_OK;
#if defined(_WIN32)
    m->backend_reason_ = AIMIC_BACKEND_REASON_NPU_NO_ENTRY;
    if (!g_npu_warned) {
        fprintf(stderr, "aimic: NPU backend (DirectML EP) not in this build, using CPU\n");
        g_npu_warned = 1;
    }
#elif defined(__APPLE__)
    m->backend_reason_ = AIMIC_BACKEND_REASON_NPU_NO_ENTRY;
    if (!g_npu_warned) {
        fprintf(stderr, "aimic: NPU backend (CoreML EP) not in this build, using CPU\n");
        g_npu_warned = 1;
    }
#else
    /* Linux: OpenVINO EP covers Intel NPU/GPU/CPU when the runtime is
     * OpenVINO-enabled; device_type can be overridden via PUREVOX_OV_DEVICE
     * (e.g. NPU).  Fails cleanly when the EP is not compiled in. */
    OrtOpenVINOProviderOptions ov;
    memset(&ov, 0, sizeof(ov));
    const char* dev = getenv("PUREVOX_OV_DEVICE");
    ov.device_type = dev ? dev : "CPU_FP32";
    ov.num_of_threads = 0;
    if (ort_ok(m->api, m->api->SessionOptionsAppendExecutionProvider_OpenVINO(m->opts, &ov))) {
        m->effective_backend_ = AIMIC_BACKEND_NPU;
    } else {
        m->backend_reason_ = AIMIC_BACKEND_REASON_NPU_UNAVAILABLE;
        if (!g_npu_warned) {
            fprintf(stderr, "aimic: NPU backend (OpenVINO EP) unavailable, using CPU\n");
            g_npu_warned = 1;
        }
    }
#endif
    return m->effective_backend_;
}

static int onnx_model_open(OnnxModel* m, const char* name, const char* path,
                           OrtLoggingLevel level, GraphOptimizationLevel gopt) {
    memset(m, 0, sizeof(*m));
    m->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!m->api) return -1;

    if (!ort_ok(m->api, m->api->CreateEnv(level, name, &m->env))) return -1;
    if (!ort_ok(m->api, m->api->CreateSessionOptions(&m->opts))) return -1;
    if (!ort_ok(m->api, m->api->SetIntraOpNumThreads(m->opts, 1))) return -1;
    if (!ort_ok(m->api, m->api->SetInterOpNumThreads(m->opts, 1))) return -1;
    if (!ort_ok(m->api, m->api->SetSessionExecutionMode(m->opts, ORT_SEQUENTIAL))) return -1;
    if (!ort_ok(m->api, m->api->SetSessionGraphOptimizationLevel(m->opts, gopt))) return -1;
    onnx_apply_backend(m);
    if (!ort_ok(m->api, m->api->GetAllocatorWithDefaultOptions(&m->allocator))) return -1;
    if (!ort_ok(m->api, m->api->CreateMemoryInfo("Cpu", OrtArenaAllocator, 0,
                                                 OrtMemTypeDefault, &m->meminfo))) return -1;
    if (!ort_create_session_path(m->api, m->env, path, m->opts, &m->session)) return -1;

    size_t ni = 0, no = 0;
    if (!ort_ok(m->api, m->api->SessionGetInputCount(m->session, &ni))) return -1;
    if (!ort_ok(m->api, m->api->SessionGetOutputCount(m->session, &no))) return -1;
    m->n_inputs = ni;
    m->n_outputs = no;
    m->input_names = (char**)calloc(ni ? ni : 1, sizeof(char*));
    m->output_names = (char**)calloc(no ? no : 1, sizeof(char*));
    m->input_shapes = (int64_t**)calloc(ni ? ni : 1, sizeof(int64_t*));
    m->input_ndims = (size_t*)calloc(ni ? ni : 1, sizeof(size_t));
    if (!m->input_names || !m->output_names || !m->input_shapes || !m->input_ndims) return -1;

    for (size_t i = 0; i < ni; ++i) {
        if (!ort_ok(m->api, m->api->SessionGetInputName(m->session, i, m->allocator,
                                                        &m->input_names[i]))) return -1;
        OrtTypeInfo* typeinfo = NULL;
        const OrtTensorTypeAndShapeInfo* tinfo = NULL;
        if (!ort_ok(m->api, m->api->SessionGetInputTypeInfo(m->session, i, &typeinfo))) return -1;
        if (!ort_ok(m->api, m->api->CastTypeInfoToTensorInfo(typeinfo, &tinfo))) {
            m->api->ReleaseTypeInfo(typeinfo);
            return -1;
        }
        size_t nd = 0;
        m->api->GetDimensionsCount(tinfo, &nd);
        m->input_ndims[i] = nd;
        m->input_shapes[i] = (int64_t*)calloc(nd ? nd : 1, sizeof(int64_t));
        if (nd > 0) m->api->GetDimensions(tinfo, m->input_shapes[i], nd);
        /* CastTypeInfoToTensorInfo result is owned by the type info; do NOT
         * ReleaseTensorTypeAndShapeInfo separately (1.11.1 double-free crash) */
        m->api->ReleaseTypeInfo(typeinfo);
    }
    for (size_t i = 0; i < no; ++i) {
        if (!ort_ok(m->api, m->api->SessionGetOutputName(m->session, i, m->allocator,
                                                         &m->output_names[i]))) return -1;
    }
    return 0;
}

static void onnx_model_close(OnnxModel* m) {
    if (!m->api) return;
    for (size_t i = 0; i < m->n_inputs; ++i) {
        if (m->input_names[i]) m->api->AllocatorFree(m->allocator, m->input_names[i]);
        free(m->input_shapes[i]);
    }
    for (size_t i = 0; i < m->n_outputs; ++i) {
        if (m->output_names[i]) m->api->AllocatorFree(m->allocator, m->output_names[i]);
    }
    free(m->input_names);
    free(m->output_names);
    free(m->input_shapes);
    free(m->input_ndims);
    if (m->session) m->api->ReleaseSession(m->session);
    if (m->opts) m->api->ReleaseSessionOptions(m->opts);
    if (m->meminfo) m->api->ReleaseMemoryInfo(m->meminfo);
    if (m->env) m->api->ReleaseEnv(m->env);
    memset(m, 0, sizeof(*m));
}

static size_t onnx_input_total(OnnxModel* m, const char* name, size_t fallback) {
    for (size_t i = 0; i < m->n_inputs; ++i) {
        if (m->input_names[i] && strcmp(m->input_names[i], name) == 0) {
            size_t total = 1;
            size_t nd = m->input_ndims[i];
            for (size_t j = 0; j < nd; ++j) {
                int64_t d = m->input_shapes[i][j];
                if (d <= 0) return fallback;
                total *= (size_t)d;
            }
            return total;
        }
    }
    return fallback;
}

static OrtValue* ort_tensor(OnnxModel* m, float* data, size_t count,
                            const int64_t* shape, size_t ndim) {
    OrtValue* v = NULL;
    OrtStatus* st = m->api->CreateTensorWithDataAsOrtValue(
        m->meminfo, data, count * sizeof(float), shape, ndim,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v);
    if (st) {
        fprintf(stderr, "aimic: CreateTensor failed: %s\n",
                m->api->GetErrorMessage(st));
        m->api->ReleaseStatus(st);
        return NULL;
    }
    return v;
}

/* build tensor array in input_names order; unknown inputs get zero-length
 * placeholder tensors so name/value indexes stay aligned */
static void ort_build_inputs(OnnxModel* m, OrtValue** vals,
                             const char* match[], float** ptrs[], size_t* counts[],
                             size_t n_match) {
    (void)match; (void)ptrs; (void)counts; (void)n_match;
    (void)vals;
}

static void ort_run(OnnxModel* m, OrtValue** in, size_t n_in,
                    OrtValue** out, size_t n_out) {
    const char** in_names = (const char**)malloc((n_in ? n_in : 1) * sizeof(char*));
    const char** out_names = (const char**)malloc((n_out ? n_out : 1) * sizeof(char*));
    for (size_t i = 0; i < n_in; ++i) in_names[i] = m->input_names[i];
    for (size_t i = 0; i < n_out; ++i) out_names[i] = m->output_names[i];
    OrtStatus* st = m->api->Run(m->session, NULL, in_names,
                                (const OrtValue* const*)in, n_in,
                                out_names, n_out, out);
    if (st) {
        fprintf(stderr, "aimic: Run failed: %s\n", m->api->GetErrorMessage(st));
        m->api->ReleaseStatus(st);
    }
    free(in_names);
    free(out_names);
}

static size_t ort_value_elems(OnnxModel* m, OrtValue* v) {
    OrtTensorTypeAndShapeInfo* info = NULL;
    size_t total = 1;
    if (m->api->GetTensorTypeAndShape(v, &info)) return 0;
    if (!info) return 0;
    size_t nd = 0;
    m->api->GetDimensionsCount(info, &nd);
    int64_t dims[16];
    if (nd > 16) nd = 16;
    if (nd > 0) m->api->GetDimensions(info, dims, nd);
    for (size_t i = 0; i < nd; ++i) {
        if (dims[i] <= 0) { total = 0; break; }
        total *= (size_t)dims[i];
    }
    m->api->ReleaseTensorTypeAndShapeInfo(info);
    return total;
}

static float* ort_value_data(OnnxModel* m, OrtValue* v) {
    void* p = NULL;
    if (m->api->GetTensorMutableData(v, &p)) return NULL;
    return (float*)p;
}

/* ───────────────────────── VAD ───────────────────────── */
struct VadGate {
    float threshold_linear_;
    int onset_frames_;
    int hang_frames_;
    bool active_;
    int voice_cnt_;
    int silence_cnt_;
};

VadGate* vad_new(float threshold_dbfs, float onset_ms, float hang_ms, float fs, int hop) {
    VadGate* v = (VadGate*)calloc(1, sizeof(VadGate));
    if (!v) return NULL;
    v->threshold_linear_ = (float)pow(10.0, threshold_dbfs / 20.0);
    v->onset_frames_ = (int)onset_ms / 1000.0f * fs / hop;
    if (v->onset_frames_ < 1) v->onset_frames_ = 1;
    v->hang_frames_ = (int)(hang_ms / 1000.0f * fs / hop);
    if (v->hang_frames_ < 1) v->hang_frames_ = 1;
    return v;
}

void vad_free(VadGate* v) { free(v); }
void vad_reset(VadGate* v) { v->active_ = false; v->voice_cnt_ = 0; v->silence_cnt_ = 0; }

int vad_process(VadGate* v, float* samples, size_t n) {
    float sq = 0.0f;
    for (size_t i = 0; i < n; ++i) sq += samples[i] * samples[i];
    float rms = (sq > 0.0f) ? (float)sqrt(sq / (float)n) : 0.0f;
    bool is_voice = rms > v->threshold_linear_;
    if (is_voice) { v->voice_cnt_++; v->silence_cnt_ = 0; }
    else { v->silence_cnt_++; v->voice_cnt_ = 0; }
    if (!v->active_ && v->voice_cnt_ >= v->onset_frames_) v->active_ = true;
    else if (v->active_ && v->silence_cnt_ >= v->hang_frames_) v->active_ = false;
    if (!v->active_) {
        for (size_t i = 0; i < n; ++i) samples[i] = 0.0f;
    }
    return v->active_ ? 1 : 0;
}

int vad_is_active(const VadGate* v) { return v->active_ ? 1 : 0; }
void vad_set_threshold(VadGate* v, float dbfs) {
    v->threshold_linear_ = (float)pow(10.0, dbfs / 20.0);
}
float vad_threshold_dbfs(const VadGate* v) {
    return 20.0f * (float)log10(v->threshold_linear_);
}

/* ───────────────────────── AGC ───────────────────────── */
#define AGC_SILENT_TAIL_FRAMES 15

struct AgcController {
    float target_dbfs_;
    float target_linear_;
    float gain_min_linear_;
    float gain_max_linear_;
    float silence_thr_linear_;
    float rms_floor_linear_;
    float attack_alpha_;
    float release_alpha_;
    float decay_factor_;
    float call_interval_ms_;
    float attack_ms_;
    float release_ms_;
    float dead_zone_;
    float rms_alpha_;
    float smoothed_gain_linear_;
    float rms_ema_;
    bool initialized_;
    bool enabled_;
    bool voice_active_;
    int silent_tail_count_;
};

AgcController* agc_new(float target_dbfs, float call_interval_ms) {
    AgcController* a = (AgcController*)calloc(1, sizeof(AgcController));
    if (!a) return NULL;
    a->target_dbfs_ = target_dbfs;
    a->target_linear_ = (float)pow(10.0, target_dbfs / 20.0);
    a->gain_min_linear_ = (float)pow(10.0, -30.0 / 20.0);
    a->gain_max_linear_ = (float)pow(10.0, 30.0 / 20.0);
    a->silence_thr_linear_ = (float)pow(10.0, -45.0 / 20.0);
    a->rms_floor_linear_ = (float)pow(10.0, -60.0 / 20.0);
    a->smoothed_gain_linear_ = 1.0f;
    a->initialized_ = false;
    a->enabled_ = false;
    a->voice_active_ = false;
    a->silent_tail_count_ = 0;
    float dt = call_interval_ms / 1000.0f;
    a->attack_alpha_ = 1.0f - (float)exp(-dt / 0.010);
    a->release_alpha_ = 1.0f - (float)exp(-dt / 0.150);
    a->decay_factor_ = (float)pow(0.5, dt);
    a->call_interval_ms_ = call_interval_ms;
    a->attack_ms_ = 10.0f;
    a->release_ms_ = 150.0f;
    a->dead_zone_ = (float)pow(10.0, 0.5 / 20.0);
    a->rms_alpha_ = 1.0f - (float)exp(-dt / 0.200);
    return a;
}

void agc_free(AgcController* a) { free(a); }

void agc_reset(AgcController* a) {
    a->smoothed_gain_linear_ = 1.0f;
    a->rms_ema_ = 0.0f;
    a->initialized_ = false;
    a->voice_active_ = false;
    a->silent_tail_count_ = 0;
}

void agc_update_rms(AgcController* a, float rms_linear) {
    bool is_voice = rms_linear > a->silence_thr_linear_;
    if (is_voice) { a->silent_tail_count_ = 0; a->voice_active_ = true; }
    else {
        a->silent_tail_count_++;
        if (a->silent_tail_count_ >= AGC_SILENT_TAIL_FRAMES) a->voice_active_ = false;
    }
    if (!a->voice_active_) return;
    if (rms_linear <= a->silence_thr_linear_) return;
    if (a->rms_ema_ == 0.0f) a->rms_ema_ = rms_linear;
    else a->rms_ema_ = a->rms_alpha_ * rms_linear + (1.0f - a->rms_alpha_) * a->rms_ema_;
}

float agc_tick(AgcController* a) {
    if (!a->enabled_) return 1.0f;
    if (!a->voice_active_) {
        if (a->smoothed_gain_linear_ > 1.0f) {
            a->smoothed_gain_linear_ *= a->decay_factor_;
            if (a->smoothed_gain_linear_ < 1.0f) a->smoothed_gain_linear_ = 1.0f;
        }
        return a->smoothed_gain_linear_;
    }
    if (a->rms_ema_ == 0.0f) return a->smoothed_gain_linear_;
    float rms = a->rms_ema_;
    if (rms < a->rms_floor_linear_) rms = a->rms_floor_linear_;
    float target_gain = a->target_linear_ / rms;
    if (target_gain < a->gain_min_linear_) target_gain = a->gain_min_linear_;
    if (target_gain > a->gain_max_linear_) target_gain = a->gain_max_linear_;
    if (!a->initialized_) {
        a->initialized_ = true;
        a->smoothed_gain_linear_ = target_gain;
    } else {
        float ratio = target_gain / a->smoothed_gain_linear_;
        if (ratio > (1.0f / a->dead_zone_) && ratio < a->dead_zone_) {
            return a->smoothed_gain_linear_;
        }
        float alpha = (target_gain < a->smoothed_gain_linear_)
                          ? a->attack_alpha_ : a->release_alpha_;
        a->smoothed_gain_linear_ = alpha * target_gain
                                 + (1.0f - alpha) * a->smoothed_gain_linear_;
    }
    return a->smoothed_gain_linear_;
}

float agc_get_current_gain_linear(const AgcController* a) { return a->smoothed_gain_linear_; }
float agc_get_current_gain_db(const AgcController* a) {
    return 20.0f * (float)log10(a->smoothed_gain_linear_ > 0.0f ? a->smoothed_gain_linear_ : 1e-10f);
}
int agc_is_voice_active(const AgcController* a) { return a->voice_active_ ? 1 : 0; }
void agc_set_enabled(AgcController* a, bool enabled, float initial_gain_db) {
    if (enabled && !a->enabled_) {
        a->smoothed_gain_linear_ = (float)pow(10.0, initial_gain_db / 20.0);
        a->rms_ema_ = 0.0f;
        a->initialized_ = false;
        a->voice_active_ = false;
        a->silent_tail_count_ = 0;
    }
    a->enabled_ = enabled;
}
bool agc_is_enabled(const AgcController* a) { return a->enabled_; }
void agc_set_target(AgcController* a, float dbfs) {
    a->target_dbfs_ = dbfs;
    a->target_linear_ = (float)pow(10.0, dbfs / 20.0);
}
float agc_target_dbfs(const AgcController* a) { return a->target_dbfs_; }

void agc_set_attack_ms(AgcController* a, float ms) {
    if (ms < 1.0f) ms = 1.0f;
    if (ms > 500.0f) ms = 500.0f;
    a->attack_ms_ = ms;
    float dt = a->call_interval_ms_ / 1000.0f;
    a->attack_alpha_ = 1.0f - (float)exp(-dt / (ms * 0.001f));
}
float agc_get_attack_ms(const AgcController* a) { return a->attack_ms_; }

void agc_set_release_ms(AgcController* a, float ms) {
    if (ms < 10.0f) ms = 10.0f;
    if (ms > 1000.0f) ms = 1000.0f;
    a->release_ms_ = ms;
    float dt = a->call_interval_ms_ / 1000.0f;
    a->release_alpha_ = 1.0f - (float)exp(-dt / (ms * 0.001f));
}
float agc_get_release_ms(const AgcController* a) { return a->release_ms_; }

/* ---- NoiseFloorTracker ---- */
struct NoiseFloorTracker {
    float floor_linear_;
    float floor_db_;
    float alpha_rise_;
    float alpha_fall_;
    bool initialized_;
    int silence_frames_;
    float min_rms_;
};

NoiseFloorTracker* noise_floor_tracker_new(float call_interval_ms) {
    NoiseFloorTracker* nt = (NoiseFloorTracker*)calloc(1, sizeof(NoiseFloorTracker));
    if (!nt) return NULL;
    float dt = call_interval_ms / 1000.0f;
    nt->floor_linear_ = 0.0f;
    nt->floor_db_ = -60.0f;
    nt->alpha_rise_ = 1.0f - (float)exp(-dt / 2.0f);
    nt->alpha_fall_ = 1.0f - (float)exp(-dt / 0.5f);
    nt->initialized_ = false;
    nt->silence_frames_ = 0;
    nt->min_rms_ = 1e10f;
    return nt;
}

void noise_floor_tracker_free(NoiseFloorTracker* nt) { free(nt); }

void noise_floor_tracker_reset(NoiseFloorTracker* nt) {
    nt->floor_linear_ = 0.0f;
    nt->floor_db_ = -60.0f;
    nt->initialized_ = false;
    nt->silence_frames_ = 0;
    nt->min_rms_ = 1e10f;
}

void noise_floor_tracker_update(NoiseFloorTracker* nt, float rms_linear, bool is_voice_active) {
    if (is_voice_active) {
        nt->silence_frames_ = 0;
        return;
    }
    nt->silence_frames_++;
    if (nt->silence_frames_ < 10) return;
    if (!nt->initialized_) {
        nt->floor_linear_ = rms_linear;
        nt->floor_db_ = 20.0f * (float)log10(rms_linear > 0.0f ? rms_linear : 1e-10f);
        if (nt->floor_db_ < -80.0f) nt->floor_db_ = -80.0f;
        nt->min_rms_ = rms_linear;
        nt->initialized_ = true;
        return;
    }
    if (rms_linear < nt->min_rms_) nt->min_rms_ = rms_linear;
    if (rms_linear < nt->floor_linear_) {
        nt->floor_linear_ = nt->alpha_fall_ * rms_linear + (1.0f - nt->alpha_fall_) * nt->floor_linear_;
    } else {
        nt->floor_linear_ = nt->alpha_rise_ * rms_linear + (1.0f - nt->alpha_rise_) * nt->floor_linear_;
    }
    nt->floor_db_ = 20.0f * (float)log10(nt->floor_linear_ > 0.0f ? nt->floor_linear_ : 1e-10f);
    if (nt->floor_db_ < -80.0f) nt->floor_db_ = -80.0f;
}

float noise_floor_tracker_get_floor_db(const NoiseFloorTracker* nt) {
    return nt->initialized_ ? nt->floor_db_ : -60.0f;
}

float noise_floor_tracker_get_floor_linear(const NoiseFloorTracker* nt) {
    return nt->initialized_ ? nt->floor_linear_ : 0.001f;
}

/* ───────────────────────── Compressor ───────────────────────── */
struct Compressor {
    float threshold_db_;
    float ratio_;
    float knee_db_;
    float makeup_db_;
    float detector_attack_alpha_;
    float detector_attack_ms_;
    float detector_release_alpha_;
    float detector_release_ms_;
    float gain_attack_alpha_;
    float gain_release_alpha_;
    bool enabled_;
    float envelope_;
    float gain_smooth_;
};

static void compressor_set_detector_attack(Compressor* c, float ms, float fs) {
    c->detector_attack_ms_ = ms;
    c->detector_attack_alpha_ = 1.0f - (float)exp(-1.0 / (ms * 0.001f * fs));
}
static void compressor_set_detector_release(Compressor* c, float ms, float fs) {
    c->detector_release_ms_ = ms;
    c->detector_release_alpha_ = 1.0f - (float)exp(-1.0 / (ms * 0.001f * fs));
}
static void compressor_set_gain_attack(Compressor* c, float ms, float fs) {
    c->gain_attack_alpha_ = 1.0f - (float)exp(-1.0 / (ms * 0.001f * fs));
}
static void compressor_set_gain_release(Compressor* c, float ms, float fs) {
    c->gain_release_alpha_ = 1.0f - (float)exp(-1.0 / (ms * 0.001f * fs));
}

Compressor* compressor_new(float threshold_db, float ratio, float attack_ms,
                           float release_ms, float knee_db, float makeup_db, float fs) {
    Compressor* c = (Compressor*)calloc(1, sizeof(Compressor));
    if (!c) return NULL;
    c->threshold_db_ = threshold_db;
    c->ratio_ = ratio;
    c->knee_db_ = knee_db;
    c->makeup_db_ = makeup_db;
    c->enabled_ = false;
    c->envelope_ = 0.0f;
    c->gain_smooth_ = 1.0f;
    compressor_set_detector_attack(c, attack_ms, fs);
    compressor_set_detector_release(c, release_ms, fs);
    compressor_set_gain_attack(c, 25.0f, fs);
    compressor_set_gain_release(c, 220.0f, fs);
    return c;
}

void compressor_free(Compressor* c) { free(c); }
void compressor_set_threshold(Compressor* c, float db) { c->threshold_db_ = db; }
float compressor_get_threshold(const Compressor* c) { return c->threshold_db_; }
void compressor_set_ratio(Compressor* c, float r) { c->ratio_ = (r < 1.0f) ? 1.0f : r; }
float compressor_get_ratio(const Compressor* c) { return c->ratio_; }
void compressor_set_attack_ms(Compressor* c, float ms) { compressor_set_detector_attack(c, ms, 48000.0f); }
float compressor_get_attack_ms(const Compressor* c) { return c->detector_attack_ms_; }
void compressor_set_release_ms(Compressor* c, float ms) { compressor_set_detector_release(c, ms, 48000.0f); }
float compressor_get_release_ms(const Compressor* c) { return c->detector_release_ms_; }
void compressor_set_knee(Compressor* c, float db) { c->knee_db_ = db; }
float compressor_get_knee(const Compressor* c) { return c->knee_db_; }
void compressor_set_makeup(Compressor* c, float db) { c->makeup_db_ = db; }
float compressor_get_makeup(const Compressor* c) { return c->makeup_db_; }
void compressor_set_enabled(Compressor* c, bool en) { c->enabled_ = en; }
bool compressor_is_enabled(const Compressor* c) { return c->enabled_; }
void compressor_reset(Compressor* c) { c->envelope_ = 0.0f; c->gain_smooth_ = 1.0f; }

void compressor_process(Compressor* c, float* data, size_t len) {
    if (!c->enabled_) return;
    for (size_t i = 0; i < len; ++i) {
        float x2 = data[i] * data[i];
        float alpha = (x2 > c->envelope_) ? c->detector_attack_alpha_ : c->detector_release_alpha_;
        c->envelope_ += alpha * (x2 - c->envelope_);
        float env_db = (c->envelope_ > 1e-12f) ? 10.0f * (float)log10(c->envelope_) : -120.0f;
        float over = env_db - c->threshold_db_;
        float gr_db = 0.0f;
        if (over > 0.0f) {
            if (c->knee_db_ > 0.0f && over < c->knee_db_) {
                float t = over / c->knee_db_;
                gr_db = (1.0f / c->ratio_ - 1.0f) * over * t * 0.5f;
            } else {
                gr_db = (1.0f / c->ratio_ - 1.0f) * over;
            }
        }
        float gain_target = (float)pow(10.0, (gr_db + c->makeup_db_) / 20.0);
        if (gain_target < c->gain_smooth_) {
            c->gain_smooth_ = c->gain_attack_alpha_ * gain_target
                            + (1.0f - c->gain_attack_alpha_) * c->gain_smooth_;
        } else {
            c->gain_smooth_ = c->gain_release_alpha_ * gain_target
                            + (1.0f - c->gain_release_alpha_) * c->gain_smooth_;
        }
        data[i] = (float)tanh(data[i] * c->gain_smooth_);
    }
}

/* ------------------------- EQ (61-band peaking) ------------------------- */
#define EQ_BANDS 61
#define EQ_Q 1.414f

static const float EQ_FREQS[EQ_BANDS] = {
    20.0f, 22.4f, 25.0f, 28.0f, 31.5f, 35.5f, 40.0f, 45.0f, 50.0f, 56.0f,
    63.0f, 71.0f, 80.0f, 90.0f, 100.0f, 112.0f, 125.0f, 140.0f, 160.0f, 180.0f,
    200.0f, 224.0f, 250.0f, 280.0f, 315.0f, 355.0f, 400.0f, 450.0f, 500.0f, 560.0f,
    630.0f, 710.0f, 800.0f, 900.0f, 1000.0f, 1120.0f, 1250.0f, 1400.0f, 1600.0f, 1800.0f,
    2000.0f, 2240.0f, 2500.0f, 2800.0f, 3150.0f, 3550.0f, 4000.0f, 4500.0f, 5000.0f, 5600.0f,
    6300.0f, 7100.0f, 8000.0f, 9000.0f, 10000.0f, 11200.0f, 12500.0f, 14000.0f, 16000.0f, 18000.0f,
    20000.0f
};

typedef struct {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
} BiquadCoeff;

static BiquadCoeff design_peaking_eq(float freq, float gain_db, float q, float sample_rate) {
    BiquadCoeff c;
    memset(&c, 0, sizeof(c));
    c.b0 = 1; c.b1 = 0; c.b2 = 0; c.a1 = 0; c.a2 = 0;
    float A = (float)pow(10.0, gain_db / 40.0);
    float w0 = 2.0f * (float)M_PI * freq / sample_rate;
    float cos_w0 = (float)cos(w0);
    float sin_w0 = (float)sin(w0);
    float alpha = sin_w0 / (2.0f * q);
    float a0 = 1.0f + alpha / A;
    c.b0 = (1.0f + alpha * A) / a0;
    c.b1 = (-2.0f * cos_w0) / a0;
    c.b2 = (1.0f - alpha * A) / a0;
    c.a1 = (-2.0f * cos_w0) / a0;
    c.a2 = (1.0f - alpha / A) / a0;
    return c;
}

/* ═══════════════════════════════════════════════════════════════
 * DenoiseProcessor — purevox9 (2048 FFT + Band256, Single STFT)
 *   ONNX: spec [1,1025,1,2], enc_c [1,77106], dec_c [1,53862],
 *         tfa_c [1,1056], inter_c [1,1024]
 *   external interface: 1024-sample blocks (DENOISE_HOP=1024).
 * ═══════════════════════════════════════════════════════════════
 */
#define DENOISE_NFFT 2048
#define DENOISE_HOP 1024
#define DENOISE_FREQ 1025
#define DENOISE_SPEC_SIZE (DENOISE_FREQ * 2)   /* 2050 interleaved [r,i] */
#define DENOISE_ENC_C_SIZE 77106
#define DENOISE_DEC_C_SIZE 53862
#define DENOISE_TFA_C_SIZE 1056
#define DENOISE_INTER_C_SIZE 1024

struct DenoiseProcessor {
    OnnxModel m;
    PFFFT_Setup* fft_plan_;
    float* fft_in_;
    float* fft_out_;
    float* ifft_out_;
    float* window_;            /* sqrt-Hann NFFT */
    float* input_history_;     /* NFFT-HOP = 1024 */
    float* ola_accumulator_;   /* NFFT */
    float* window_sum_;        /* NFFT */
    float* model_spec_;        /* 2050 interleaved */
    size_t enc_c_size_, dec_c_size_, tfa_c_size_, inter_c_size_;
    float* enc_c_;
    float* dec_c_;
    float* tfa_c_;
    float* inter_c_;
    FVec acc_output_;
};

static int denoise_alloc_buffers(DenoiseProcessor* d) {
    d->fft_plan_ = pffft_new_setup(DENOISE_NFFT, PFFFT_REAL);
    d->fft_in_ = (float*)pffft_aligned_malloc(DENOISE_NFFT * sizeof(float));
    d->fft_out_ = (float*)pffft_aligned_malloc(DENOISE_NFFT * sizeof(float));
    d->ifft_out_ = (float*)pffft_aligned_malloc(DENOISE_NFFT * sizeof(float));
    d->window_ = (float*)malloc(DENOISE_NFFT * sizeof(float));
    d->input_history_ = (float*)calloc(DENOISE_NFFT - DENOISE_HOP, sizeof(float));
    d->ola_accumulator_ = (float*)calloc(DENOISE_NFFT, sizeof(float));
    d->window_sum_ = (float*)calloc(DENOISE_NFFT, sizeof(float));
    d->model_spec_ = (float*)calloc(DENOISE_SPEC_SIZE, sizeof(float));
    if (!d->fft_plan_ || !d->fft_in_ || !d->fft_out_ || !d->ifft_out_ || !d->window_ ||
        !d->input_history_ || !d->ola_accumulator_ || !d->window_sum_ || !d->model_spec_)
        return -1;
    make_sqrt_hann(d->window_, DENOISE_NFFT);
    fvec_init(&d->acc_output_);
    return 0;
}

static void denoise_free_buffers(DenoiseProcessor* d) {
    if (d->fft_plan_) pffft_destroy_setup(d->fft_plan_);
    if (d->fft_in_) pffft_aligned_free(d->fft_in_);
    if (d->fft_out_) pffft_aligned_free(d->fft_out_);
    if (d->ifft_out_) pffft_aligned_free(d->ifft_out_);
    free(d->window_);
    free(d->input_history_);
    free(d->ola_accumulator_);
    free(d->window_sum_);
    free(d->model_spec_);
    free(d->enc_c_);
    free(d->dec_c_);
    free(d->tfa_c_);
    free(d->inter_c_);
    fvec_free(&d->acc_output_);
}

/* pure-frequency-domain ONNX: model_spec_ -> infer -> update caches -> model_spec_ = enhanced */
static void denoise_run_onnx(DenoiseProcessor* d) {
    OnnxModel* m = &d->m;
    size_t nin = m->n_inputs, nout = m->n_outputs;
    OrtValue* inputs[16] = {NULL};
    OrtValue* outputs[16] = {NULL};
    int64_t spec_shape[4] = {1, DENOISE_FREQ, 1, 2};
    int64_t s2[2];
    for (size_t i = 0; i < nin && i < 16; ++i) {
        const char* name = m->input_names[i];
        if (!name) continue;
        if (strcmp(name, "spec") == 0)
            inputs[i] = ort_tensor(m, d->model_spec_, DENOISE_SPEC_SIZE, spec_shape, 4);
        else if (strcmp(name, "enc_c") == 0) {
            s2[0] = 1; s2[1] = (int64_t)d->enc_c_size_;
            inputs[i] = ort_tensor(m, d->enc_c_, d->enc_c_size_, s2, 2);
        } else if (strcmp(name, "dec_c") == 0) {
            s2[0] = 1; s2[1] = (int64_t)d->dec_c_size_;
            inputs[i] = ort_tensor(m, d->dec_c_, d->dec_c_size_, s2, 2);
        } else if (strcmp(name, "tfa_c") == 0) {
            s2[0] = 1; s2[1] = (int64_t)d->tfa_c_size_;
            inputs[i] = ort_tensor(m, d->tfa_c_, d->tfa_c_size_, s2, 2);
        } else if (strcmp(name, "inter_c") == 0) {
            s2[0] = 1; s2[1] = (int64_t)d->inter_c_size_;
            inputs[i] = ort_tensor(m, d->inter_c_, d->inter_c_size_, s2, 2);
        } else {
            s2[0] = 1; s2[1] = 0;
            inputs[i] = ort_tensor(m, d->model_spec_, 0, s2, 2);
        }
    }
    ort_run(m, inputs, nin, outputs, nout);
    for (size_t i = 0; i < nout && i < 16; ++i) {
        const char* name = m->output_names[i];
        if (!name || !outputs[i]) continue;
        float* data = ort_value_data(m, outputs[i]);
        size_t total = ort_value_elems(m, outputs[i]);
        if (!data) continue;
        if (strcmp(name, "enhanced_spec") == 0)
            memcpy(d->model_spec_, data, DENOISE_SPEC_SIZE * sizeof(float));
        else if (strcmp(name, "enc_c_out") == 0) {
            size_t n = total < d->enc_c_size_ ? total : d->enc_c_size_;
            memcpy(d->enc_c_, data, n * sizeof(float));
        } else if (strcmp(name, "dec_c_out") == 0) {
            size_t n = total < d->dec_c_size_ ? total : d->dec_c_size_;
            memcpy(d->dec_c_, data, n * sizeof(float));
        } else if (strcmp(name, "tfa_c_out") == 0) {
            size_t n = total < d->tfa_c_size_ ? total : d->tfa_c_size_;
            memcpy(d->tfa_c_, data, n * sizeof(float));
        } else if (strcmp(name, "inter_c_out") == 0) {
            size_t n = total < d->inter_c_size_ ? total : d->inter_c_size_;
            memcpy(d->inter_c_, data, n * sizeof(float));
        }
    }
    for (size_t i = 0; i < nin && i < 16; ++i) if (inputs[i]) m->api->ReleaseValue(inputs[i]);
    for (size_t i = 0; i < nout && i < 16; ++i) if (outputs[i]) m->api->ReleaseValue(outputs[i]);
}

/* operand: FFT -> build model_spec_ -> ONNX */
static void denoise_compute_spec(DenoiseProcessor* d, const float* input_1024) {
    size_t prev_size = DENOISE_NFFT - DENOISE_HOP;
    memcpy(d->fft_in_, d->input_history_, prev_size * sizeof(float));
    memcpy(d->fft_in_ + prev_size, input_1024, DENOISE_HOP * sizeof(float));
    memmove(d->input_history_, d->input_history_ + DENOISE_HOP,
            (prev_size - DENOISE_HOP) * sizeof(float));
    memcpy(d->input_history_ + prev_size - DENOISE_HOP, input_1024,
           DENOISE_HOP * sizeof(float));
    for (int i = 0; i < DENOISE_NFFT; ++i) d->fft_in_[i] *= d->window_[i];
    pffft_transform_ordered(d->fft_plan_, d->fft_in_, d->fft_out_, NULL, PFFFT_FORWARD);
    d->model_spec_[0] = d->fft_out_[0];                       /* DC real */
    d->model_spec_[1] = 0.0f;                                 /* DC imag */
    d->model_spec_[DENOISE_SPEC_SIZE - 2] = d->fft_out_[1];   /* Nyquist real */
    d->model_spec_[DENOISE_SPEC_SIZE - 1] = 0.0f;             /* Nyquist imag */
    for (int k = 1; k < DENOISE_FREQ - 1; ++k) {
        int pidx = 2 + (k - 1) * 2;
        d->model_spec_[k * 2] = d->fft_out_[pidx];
        d->model_spec_[k * 2 + 1] = d->fft_out_[pidx + 1];
    }
}

static void denoise_synth_ola(DenoiseProcessor* d) {
    d->fft_out_[0] = d->model_spec_[0];
    d->fft_out_[1] = d->model_spec_[DENOISE_SPEC_SIZE - 2];
    for (int k = 1; k < DENOISE_FREQ - 1; ++k) {
        int p = 2 + (k - 1) * 2;
        d->fft_out_[p] = d->model_spec_[k * 2];
        d->fft_out_[p + 1] = d->model_spec_[k * 2 + 1];
    }
    pffft_transform_ordered(d->fft_plan_, d->fft_out_, d->ifft_out_, NULL, PFFFT_BACKWARD);
    float scale = 1.0f / DENOISE_NFFT;
    for (int i = 0; i < DENOISE_NFFT; ++i) d->ifft_out_[i] *= scale * d->window_[i];
    for (int i = 0; i < DENOISE_NFFT; ++i) d->ola_accumulator_[i] += d->ifft_out_[i];
    for (int i = 0; i < DENOISE_NFFT; ++i) d->window_sum_[i] += d->window_[i] * d->window_[i];
    for (int i = 0; i < DENOISE_HOP; ++i) {
        float norm = d->window_sum_[i];
        float val = (norm > 1e-6f) ? (d->ola_accumulator_[i] / norm) : d->ola_accumulator_[i];
        fvec_push(&d->acc_output_, val);
    }
    for (int i = 0; i < DENOISE_NFFT - DENOISE_HOP; ++i) {
        d->ola_accumulator_[i] = d->ola_accumulator_[i + DENOISE_HOP];
        d->window_sum_[i] = d->window_sum_[i + DENOISE_HOP];
    }
    for (int i = DENOISE_NFFT - DENOISE_HOP; i < DENOISE_NFFT; ++i) {
        d->ola_accumulator_[i] = 0.0f;
        d->window_sum_[i] = 0.0f;
    }
}

DenoiseProcessor* denoise_new(const char* model_path) {
    if (!model_path || !model_path[0]) return NULL;
    DenoiseProcessor* d = (DenoiseProcessor*)calloc(1, sizeof(DenoiseProcessor));
    if (!d) return NULL;
    if (onnx_model_open(&d->m, "DenoiseProcessor", model_path,
                        ORT_LOGGING_LEVEL_WARNING, ORT_ENABLE_BASIC) != 0) {
        free(d);
        return NULL;
    }
    if (denoise_alloc_buffers(d) != 0) {
        onnx_model_close(&d->m);
        denoise_free_buffers(d);
        free(d);
        return NULL;
    }
    d->enc_c_size_ = onnx_input_total(&d->m, "enc_c", DENOISE_ENC_C_SIZE);
    d->dec_c_size_ = onnx_input_total(&d->m, "dec_c", DENOISE_DEC_C_SIZE);
    d->tfa_c_size_ = onnx_input_total(&d->m, "tfa_c", DENOISE_TFA_C_SIZE);
    d->inter_c_size_ = onnx_input_total(&d->m, "inter_c", DENOISE_INTER_C_SIZE);
    d->enc_c_ = (float*)calloc(d->enc_c_size_, sizeof(float));
    d->dec_c_ = (float*)calloc(d->dec_c_size_, sizeof(float));
    d->tfa_c_ = (float*)calloc(d->tfa_c_size_, sizeof(float));
    d->inter_c_ = (float*)calloc(d->inter_c_size_, sizeof(float));
    if (!d->enc_c_ || !d->dec_c_ || !d->tfa_c_ || !d->inter_c_) {
        onnx_model_close(&d->m);
        denoise_free_buffers(d);
        free(d);
        return NULL;
    }
    /* warm-up: feed 3 silent blocks, discard output */
    float silent[DENOISE_HOP] = {0.0f};
    float dummy[DENOISE_HOP];
    for (int i = 0; i < 3; ++i) denoise_process_chunk(d, silent, dummy);
    fvec_clear(&d->acc_output_);
    return d;
}

void denoise_free(DenoiseProcessor* d) {
    if (!d) return;
    onnx_model_close(&d->m);
    denoise_free_buffers(d);
    free(d);
}

void denoise_process_chunk(DenoiseProcessor* d, const float* in, float* out) {
    denoise_compute_spec(d, in);
    denoise_run_onnx(d);
    denoise_synth_ola(d);
    if (d->acc_output_.len >= DENOISE_HOP) {
        memcpy(out, d->acc_output_.data, DENOISE_HOP * sizeof(float));
        fvec_erase_front(&d->acc_output_, DENOISE_HOP);
    } else {
        memset(out, 0, DENOISE_HOP * sizeof(float));
    }
}

void denoise_process_spec_only(DenoiseProcessor* d, const float* in, float* spec_out) {
    denoise_compute_spec(d, in);
    denoise_run_onnx(d);
    memcpy(spec_out, d->model_spec_, DENOISE_SPEC_SIZE * sizeof(float));
}

void denoise_process_spec_freq(DenoiseProcessor* d, const float* in, float* out) {
    memcpy(d->model_spec_, in, DENOISE_SPEC_SIZE * sizeof(float));
    denoise_run_onnx(d);
    memcpy(out, d->model_spec_, DENOISE_SPEC_SIZE * sizeof(float));
}

void denoise_reset(DenoiseProcessor* d) {
    memset(d->enc_c_, 0, d->enc_c_size_ * sizeof(float));
    memset(d->dec_c_, 0, d->dec_c_size_ * sizeof(float));
    memset(d->tfa_c_, 0, d->tfa_c_size_ * sizeof(float));
    memset(d->inter_c_, 0, d->inter_c_size_ * sizeof(float));
    memset(d->input_history_, 0, (DENOISE_NFFT - DENOISE_HOP) * sizeof(float));
    memset(d->ola_accumulator_, 0, DENOISE_NFFT * sizeof(float));
    memset(d->window_sum_, 0, DENOISE_NFFT * sizeof(float));
    memset(d->model_spec_, 0, DENOISE_SPEC_SIZE * sizeof(float));
    fvec_clear(&d->acc_output_);
}

/* ═══════════════════════════════════════════════════════════════
 * StftProcessor — unified FFT/IFFT/OLA (2048-pt, 1024-hop, 48kHz)
 *   flat spectrum [r0..r1024, i0..i1024] = 2050 floats.
 * ═══════════════════════════════════════════════════════════════
 */
#define STFT_NFFT 2048
#define STFT_HOP 1024
#define STFT_FREQ 1025
#define STFT_SPEC_FLOATS 2050

typedef struct {
    PFFFT_Setup* fft_plan_;
    float* fft_in_;
    float* fft_out_;
    float* ifft_out_;
    float* window_;           /* Hann */
    float* input_history_;    /* NFFT-HOP */
    float* ola_acc_;          /* NFFT */
    float* win_sum_;          /* NFFT */
    bool primed_;
} StftProcessor;

static int stft_init(StftProcessor* s) {
    memset(s, 0, sizeof(*s));
    s->fft_plan_ = pffft_new_setup(STFT_NFFT, PFFFT_REAL);
    s->fft_in_ = (float*)pffft_aligned_malloc(STFT_NFFT * sizeof(float));
    s->fft_out_ = (float*)pffft_aligned_malloc(STFT_NFFT * sizeof(float));
    s->ifft_out_ = (float*)pffft_aligned_malloc(STFT_NFFT * sizeof(float));
    s->window_ = (float*)malloc(STFT_NFFT * sizeof(float));
    s->input_history_ = (float*)calloc(STFT_NFFT - STFT_HOP, sizeof(float));
    s->ola_acc_ = (float*)calloc(STFT_NFFT, sizeof(float));
    s->win_sum_ = (float*)calloc(STFT_NFFT, sizeof(float));
    if (!s->fft_plan_ || !s->fft_in_ || !s->fft_out_ || !s->ifft_out_ || !s->window_ ||
        !s->input_history_ || !s->ola_acc_ || !s->win_sum_) return -1;
    make_sqrt_hann(s->window_, STFT_NFFT);
    for (int i = 0; i < STFT_NFFT; ++i) s->window_[i] *= s->window_[i]; /* Hann */
    return 0;
}

static void stft_free(StftProcessor* s) {
    if (s->fft_plan_) pffft_destroy_setup(s->fft_plan_);
    if (s->fft_in_) pffft_aligned_free(s->fft_in_);
    if (s->fft_out_) pffft_aligned_free(s->fft_out_);
    if (s->ifft_out_) pffft_aligned_free(s->ifft_out_);
    free(s->window_);
    free(s->input_history_);
    free(s->ola_acc_);
    free(s->win_sum_);
    memset(s, 0, sizeof(*s));
}

static void stft_forward(StftProcessor* s, const float* in, float* spec_planar) {
    size_t prev = STFT_NFFT - STFT_HOP;
    memcpy(s->fft_in_, s->input_history_, prev * sizeof(float));
    memcpy(s->fft_in_ + prev, in, STFT_HOP * sizeof(float));
    memmove(s->input_history_, s->input_history_ + STFT_HOP, (prev - STFT_HOP) * sizeof(float));
    memcpy(s->input_history_ + prev - STFT_HOP, in, STFT_HOP * sizeof(float));
    for (int i = 0; i < STFT_NFFT; ++i) s->fft_in_[i] *= s->window_[i];
    pffft_transform_ordered(s->fft_plan_, s->fft_in_, s->fft_out_, NULL, PFFFT_FORWARD);
    float* rp = spec_planar;
    float* ip = spec_planar + STFT_FREQ;
    rp[0] = s->fft_out_[0]; ip[0] = 0.0f;
    rp[STFT_FREQ - 1] = s->fft_out_[1]; ip[STFT_FREQ - 1] = 0.0f;
    for (int k = 1; k < STFT_FREQ - 1; ++k) {
        int p = 2 + (k - 1) * 2;
        rp[k] = s->fft_out_[p]; ip[k] = s->fft_out_[p + 1];
    }
}

static void stft_backward(StftProcessor* s, const float* spec_planar, float* out) {
    s->fft_out_[0] = spec_planar[0];
    s->fft_out_[1] = spec_planar[STFT_FREQ - 1];
    for (int k = 1; k < STFT_FREQ - 1; ++k) {
        int p = 2 + (k - 1) * 2;
        s->fft_out_[p] = spec_planar[k];
        s->fft_out_[p + 1] = spec_planar[STFT_FREQ + k];
    }
    pffft_transform_ordered(s->fft_plan_, s->fft_out_, s->ifft_out_, NULL, PFFFT_BACKWARD);
    float sc = 1.0f / STFT_NFFT;
    for (int i = 0; i < STFT_NFFT; ++i) {
        s->ifft_out_[i] *= sc * s->window_[i];
        s->ola_acc_[i] += s->ifft_out_[i];
        s->win_sum_[i] += s->window_[i] * s->window_[i];
    }
    if (!s->primed_) {
        s->primed_ = true;
        memset(out, 0, STFT_HOP * sizeof(float));
    } else {
        for (int i = 0; i < STFT_HOP; ++i) {
            float n = s->win_sum_[i];
            out[i] = (n > 1e-6f) ? (s->ola_acc_[i] / n) : s->ola_acc_[i];
        }
    }
    for (int i = 0; i < STFT_NFFT - STFT_HOP; ++i) {
        s->ola_acc_[i] = s->ola_acc_[i + STFT_HOP];
        s->win_sum_[i] = s->win_sum_[i + STFT_HOP];
    }
    for (int i = STFT_NFFT - STFT_HOP; i < STFT_NFFT; ++i) {
        s->ola_acc_[i] = 0.0f;
        s->win_sum_[i] = 0.0f;
    }
}

static void stft_reset(StftProcessor* s) {
    memset(s->input_history_, 0, (STFT_NFFT - STFT_HOP) * sizeof(float));
    memset(s->ola_acc_, 0, STFT_NFFT * sizeof(float));
    memset(s->win_sum_, 0, STFT_NFFT * sizeof(float));
    s->primed_ = false;
}

/* ═══════════════════════════════════════════════════════════════
 * TseProcessor — tse15 streaming ONNX (2048 FFT, 1024 HOP, 48kHz, flat cache)
 *   ONNX: spec_frame [1,2,1,1025] + enr_spec [1,2,Te,1025] + cache_in [319040]
 *         → enh_frame [1,2,1,1025] + cache_out [319040]
 * ═══════════════════════════════════════════════════════════════
 */
#define TSE_NFFT 2048
#define TSE_HOP 1024
#define TSE_FREQ 1025
#define TSE_SPEC_FLOATS (2 * TSE_FREQ)          /* 2050 flat [real, imag] */
#define TSE_ENR_CH 2
#define TSE_ENR_FLOATS (TSE_ENR_CH * TSE_FREQ)  /* 2050 */
#define TSE_CACHE_TOTAL 319040

struct TseProcessor {
    OnnxModel m;
    PFFFT_Setup* fft_plan_;
    float* fft_in_;
    float* fft_out_;
    float* ifft_out_;
    float* window_;           /* Hann */
    float* input_history_;    /* NFFT-HOP = 1024 */
    float* ola_accumulator_;  /* NFFT */
    float* window_sum_;       /* NFFT */
    bool primed_;
    int frame_count_;
    bool debug_dump_;
    char debug_dir_[1024];
    float* cache_;            /* CACHE_TOTAL */
    float* enr_buf_;          /* n_frames * 2050 */
    size_t enr_len_;
    size_t enr_frames_;
    float* spec_buf_;         /* 2050 flat [real|imag] */
};

static void dump_bin(const char* path, const float* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
}

static void stat_line(const char* tag, const float* data, size_t n, int frame) {
    float lo = data[0], hi = data[0];
    double sum = 0, sq = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        sum += v;
        sq += (double)v * v;
    }
    float rms = (n > 0) ? (float)sqrt(sq / n) : 0.0f;
    printf("[TSE15 dbg] f%02d %-12s | min=%+.4f max=%+.4f rms=%.4f\n",
           frame, tag, lo, hi, rms);
}

static int tse_alloc_buffers(TseProcessor* t) {
    t->fft_plan_ = pffft_new_setup(TSE_NFFT, PFFFT_REAL);
    t->fft_in_ = (float*)pffft_aligned_malloc(TSE_NFFT * sizeof(float));
    t->fft_out_ = (float*)pffft_aligned_malloc(TSE_NFFT * sizeof(float));
    t->ifft_out_ = (float*)pffft_aligned_malloc(TSE_NFFT * sizeof(float));
    t->window_ = (float*)malloc(TSE_NFFT * sizeof(float));
    t->input_history_ = (float*)calloc(TSE_NFFT - TSE_HOP, sizeof(float));
    t->ola_accumulator_ = (float*)calloc(TSE_NFFT, sizeof(float));
    t->window_sum_ = (float*)calloc(TSE_NFFT, sizeof(float));
    t->spec_buf_ = (float*)calloc(TSE_SPEC_FLOATS, sizeof(float));
    t->cache_ = (float*)calloc(TSE_CACHE_TOTAL, sizeof(float));
    if (!t->fft_plan_ || !t->fft_in_ || !t->fft_out_ || !t->ifft_out_ || !t->window_ ||
        !t->input_history_ || !t->ola_accumulator_ || !t->window_sum_ || !t->spec_buf_)
        return -1;
    make_sqrt_hann(t->window_, TSE_NFFT);
    for (int i = 0; i < TSE_NFFT; ++i) t->window_[i] *= t->window_[i]; /* Hann */
    return 0;
}

static void tse_free_buffers(TseProcessor* t) {
    if (t->fft_plan_) pffft_destroy_setup(t->fft_plan_);
    if (t->fft_in_) pffft_aligned_free(t->fft_in_);
    if (t->fft_out_) pffft_aligned_free(t->fft_out_);
    if (t->ifft_out_) pffft_aligned_free(t->ifft_out_);
    free(t->window_);
    free(t->input_history_);
    free(t->ola_accumulator_);
    free(t->window_sum_);
    free(t->spec_buf_);
    free(t->cache_);
    free(t->enr_buf_);
}

TseProcessor* tse_new(const char* model_path) {
    if (!model_path || !model_path[0]) return NULL;
    TseProcessor* t = (TseProcessor*)calloc(1, sizeof(TseProcessor));
    if (!t) return NULL;
    if (onnx_model_open(&t->m, "TseProcessor", model_path,
                        ORT_LOGGING_LEVEL_WARNING, ORT_ENABLE_ALL) != 0) {
        free(t);
        return NULL;
    }
    if (tse_alloc_buffers(t) != 0) {
        onnx_model_close(&t->m);
        tse_free_buffers(t);
        free(t);
        return NULL;
    }
    return t;
}

void tse_free(TseProcessor* t) {
    if (!t) return;
    onnx_model_close(&t->m);
    tse_free_buffers(t);
    free(t);
}

bool tse_has_reference(const TseProcessor* t) { return t->enr_len_ > 0; }

void tse_set_reference(TseProcessor* t, const float* ref, size_t n) {
    if (n < (size_t)TSE_NFFT) return;
    size_t n_frames = n / TSE_HOP + 1;
    size_t new_len = n_frames * TSE_ENR_FLOATS;
    float* nb = (float*)realloc(t->enr_buf_, new_len * sizeof(float));
    if (!nb) return;
    t->enr_buf_ = nb;
    t->enr_len_ = new_len;
    t->enr_frames_ = n_frames;
    memset(nb, 0, new_len * sizeof(float));

    size_t pad = TSE_NFFT / 2;
    float* padded = (float*)malloc((n + pad * 2) * sizeof(float));
    if (!padded) return;
    for (size_t i = 0; i < pad && i + 1 < n; ++i)
        padded[pad - 1 - i] = ref[i + 1];
    memcpy(padded + pad, ref, n * sizeof(float));
    for (size_t i = 0; i < pad && i + 1 < n; ++i)
        padded[pad + n + i] = ref[n - 2 - i];

    float* real_ptr = nb;
    float* imag_ptr = nb + n_frames * TSE_FREQ;
    for (size_t tt = 0; tt < n_frames; ++tt) {
        size_t off = tt * TSE_HOP;
        for (int i = 0; i < TSE_NFFT; ++i)
            t->fft_in_[i] = padded[off + i] * t->window_[i];
        pffft_transform_ordered(t->fft_plan_, t->fft_in_, t->fft_out_, NULL, PFFFT_FORWARD);
        real_ptr[tt * TSE_FREQ] = t->fft_out_[0];
        imag_ptr[tt * TSE_FREQ] = 0.0f;
        for (int k = 1; k < TSE_FREQ - 1; ++k) {
            int p = 2 + (k - 1) * 2;
            real_ptr[tt * TSE_FREQ + k] = t->fft_out_[p];
            imag_ptr[tt * TSE_FREQ + k] = t->fft_out_[p + 1];
        }
        real_ptr[tt * TSE_FREQ + TSE_FREQ - 1] = t->fft_out_[1];
        imag_ptr[tt * TSE_FREQ + TSE_FREQ - 1] = 0.0f;
    }
    free(padded);
}

void tse_set_debug_dump(TseProcessor* t, bool enable, const char* dir) {
    t->debug_dump_ = enable;
    if (enable && dir && dir[0]) {
        snprintf(t->debug_dir_, sizeof(t->debug_dir_), "%s", dir);
        if (t->enr_len_ > 0) {
            char p[1024];
            snprintf(p, sizeof(p), "%s/debug_enr_spec.bin", t->debug_dir_);
            dump_bin(p, t->enr_buf_, t->enr_len_);
            printf("[TSE15 dbg] enr_spec dumped: %zu floats, ref_frames=%zu\n",
                   t->enr_len_, t->enr_frames_);
        }
    }
}

/* pure-frequency ONNX: spec_buf_ -> infer -> spec_buf_ = enh_frame, cache_ update */
static void tse_run_onnx(TseProcessor* t) {
    OnnxModel* m = &t->m;
    size_t nin = m->n_inputs, nout = m->n_outputs;
    OrtValue* inputs[8] = {NULL};
    OrtValue* outputs[8] = {NULL};
    int64_t ss[4] = {1, 2, 1, TSE_FREQ};
    int64_t es[4];
    int64_t cs[1] = {TSE_CACHE_TOTAL};
    for (size_t i = 0; i < nin && i < 8; ++i) {
        const char* name = m->input_names[i];
        if (!name) continue;
        if (strcmp(name, "spec_frame") == 0) {
            inputs[i] = ort_tensor(m, t->spec_buf_, TSE_SPEC_FLOATS, ss, 4);
        } else if (strcmp(name, "enr_spec") == 0) {
            es[0] = 1; es[1] = TSE_ENR_CH; es[2] = (int64_t)t->enr_frames_; es[3] = TSE_FREQ;
            inputs[i] = ort_tensor(m, t->enr_buf_, t->enr_len_, es, 4);
        } else if (strcmp(name, "cache_in") == 0) {
            inputs[i] = ort_tensor(m, t->cache_, TSE_CACHE_TOTAL, cs, 1);
        } else {
            int64_t zero_shape[1] = {0};
            inputs[i] = ort_tensor(m, t->spec_buf_, 0, zero_shape, 1);
        }
    }
    ort_run(m, inputs, nin, outputs, nout);
    for (size_t i = 0; i < nout && i < 8; ++i) {
        const char* name = m->output_names[i];
        if (!name || !outputs[i]) continue;
        float* data = ort_value_data(m, outputs[i]);
        size_t total = ort_value_elems(m, outputs[i]);
        if (!data) continue;
        if (strcmp(name, "enh_frame") == 0) {
            size_t n = total < TSE_SPEC_FLOATS ? total : TSE_SPEC_FLOATS;
            memcpy(t->spec_buf_, data, n * sizeof(float));
        } else if (strcmp(name, "cache_out") == 0) {
            size_t n = total < TSE_CACHE_TOTAL ? total : TSE_CACHE_TOTAL;
            memcpy(t->cache_, data, n * sizeof(float));
        }
    }
    for (size_t i = 0; i < nin && i < 8; ++i) if (inputs[i]) m->api->ReleaseValue(inputs[i]);
    for (size_t i = 0; i < nout && i < 8; ++i) if (outputs[i]) m->api->ReleaseValue(outputs[i]);
}

/* flat complex -> pffft packed -> IFFT -> OLA (shared tail of process_chunk/from_spec) */
static void tse_synth_ola(TseProcessor* t, const float* in_1024, float* out_1024) {
    t->fft_out_[0] = t->spec_buf_[0];
    t->fft_out_[1] = t->spec_buf_[TSE_FREQ - 1];
    for (int k = 1; k < TSE_FREQ - 1; ++k) {
        int p = 2 + (k - 1) * 2;
        t->fft_out_[p] = t->spec_buf_[k];
        t->fft_out_[p + 1] = t->spec_buf_[TSE_FREQ + k];
    }
    pffft_transform_ordered(t->fft_plan_, t->fft_out_, t->ifft_out_, NULL, PFFFT_BACKWARD);
    float scale = 1.0f / TSE_NFFT;
    for (int i = 0; i < TSE_NFFT; ++i) t->ifft_out_[i] *= scale * t->window_[i];
    for (int i = 0; i < TSE_NFFT; ++i) t->ola_accumulator_[i] += t->ifft_out_[i];
    for (int i = 0; i < TSE_NFFT; ++i) t->window_sum_[i] += t->window_[i] * t->window_[i];
    if (!t->primed_) {
        t->primed_ = true;
        memcpy(out_1024, in_1024, TSE_HOP * sizeof(float));
    } else {
        for (int i = 0; i < TSE_HOP; ++i) {
            float norm = t->window_sum_[i];
            out_1024[i] = (norm > 1e-6f) ? (t->ola_accumulator_[i] / norm)
                                         : t->ola_accumulator_[i];
        }
    }
    for (int i = 0; i < TSE_NFFT - TSE_HOP; ++i) {
        t->ola_accumulator_[i] = t->ola_accumulator_[i + TSE_HOP];
        t->window_sum_[i] = t->window_sum_[i + TSE_HOP];
    }
    for (int i = TSE_NFFT - TSE_HOP; i < TSE_NFFT; ++i) {
        t->ola_accumulator_[i] = 0.0f;
        t->window_sum_[i] = 0.0f;
    }
}

void tse_process_chunk(TseProcessor* t, const float* in, float* out) {
    if (t->enr_len_ == 0) {
        memcpy(out, in, TSE_HOP * sizeof(float));
        return;
    }
    memcpy(t->fft_in_, t->input_history_, (TSE_NFFT - TSE_HOP) * sizeof(float));
    memcpy(t->fft_in_ + TSE_NFFT - TSE_HOP, in, TSE_HOP * sizeof(float));
    memmove(t->input_history_, t->input_history_ + TSE_HOP,
            (TSE_NFFT - 2 * TSE_HOP) * sizeof(float));
    memcpy(t->input_history_ + TSE_NFFT - 2 * TSE_HOP, in, TSE_HOP * sizeof(float));

    for (int i = 0; i < TSE_NFFT; ++i) t->fft_in_[i] *= t->window_[i];
    pffft_transform_ordered(t->fft_plan_, t->fft_in_, t->fft_out_, NULL, PFFFT_FORWARD);

    float* rp = t->spec_buf_;
    float* ip = t->spec_buf_ + TSE_FREQ;
    rp[0] = t->fft_out_[0]; ip[0] = 0.0f;
    for (int k = 1; k < TSE_FREQ - 1; ++k) {
        int p = 2 + (k - 1) * 2;
        rp[k] = t->fft_out_[p];
        ip[k] = t->fft_out_[p + 1];
    }
    rp[TSE_FREQ - 1] = t->fft_out_[1]; ip[TSE_FREQ - 1] = 0.0f;

    int fc = t->frame_count_++;
    if (t->debug_dump_) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/f%02d_in.bin", t->debug_dir_, fc);
        dump_bin(path, in, TSE_HOP);
        snprintf(path, sizeof(path), "%s/f%02d_fft.bin", t->debug_dir_, fc);
        dump_bin(path, t->fft_in_, TSE_NFFT);
        snprintf(path, sizeof(path), "%s/f%02d_spec.bin", t->debug_dir_, fc);
        dump_bin(path, t->spec_buf_, TSE_SPEC_FLOATS);
    }

    tse_run_onnx(t);

    if (t->debug_dump_) stat_line("spec_raw", t->spec_buf_, TSE_SPEC_FLOATS, fc);

    tse_synth_ola(t, in, out);

    if (t->debug_dump_) {
        stat_line("output", out, TSE_HOP, fc);
        char path[1024];
        snprintf(path, sizeof(path), "%s/f%02d_out.bin", t->debug_dir_, fc);
        dump_bin(path, out, TSE_HOP);
    }
}

void tse_process_spec_freq(TseProcessor* t, const float* in, float* out) {
    memcpy(t->spec_buf_, in, TSE_SPEC_FLOATS * sizeof(float));
    t->frame_count_++;
    tse_run_onnx(t);
    memcpy(out, t->spec_buf_, TSE_SPEC_FLOATS * sizeof(float));
}

void tse_process_from_spec(TseProcessor* t, const float* spec, float* out) {
    if (t->enr_len_ == 0) { memset(out, 0, TSE_HOP * sizeof(float)); return; }
    memcpy(t->spec_buf_, spec, TSE_SPEC_FLOATS * sizeof(float));
    t->frame_count_++;
    tse_run_onnx(t);
    tse_synth_ola(t, t->spec_buf_, out);
}

void tse_reset(TseProcessor* t) {
    memset(t->cache_, 0, TSE_CACHE_TOTAL * sizeof(float));
    memset(t->input_history_, 0, (TSE_NFFT - TSE_HOP) * sizeof(float));
    memset(t->ola_accumulator_, 0, TSE_NFFT * sizeof(float));
    memset(t->window_sum_, 0, TSE_NFFT * sizeof(float));
    t->primed_ = false;
    t->frame_count_ = 0;
}

/* ═══════════════════════════════════════════════════════════════
 * AecProcessor - aec9 streaming ONNX (NFFT=2048, HOP=1024, FREQ=1025,
 *   Mel-256, delay line, complex mask). External interface 1024-sample blocks.
 * ═══════════════════════════════════════════════════════════════
 */
#define AEC_NFFT 2048
#define AEC_HOP 1024
#define AEC_FREQ 1025
#define AEC_SPEC_SIZE (AEC_FREQ * 2)              /* 2050 planar [real,imag] */
#define AEC_RES_ENC_CONV_SIZE 108544
#define AEC_RES_ENC_TFA_SIZE 248
#define AEC_MIC_ENC_CONV_SIZE 108544
#define AEC_MIC_ENC_TFA_SIZE 248
#define AEC_DEEP_ENC_TFA_SIZE 432
#define AEC_DEC_CONV_SIZE 10752
#define AEC_DEC_TFA_SIZE 496
#define AEC_INTER_SIZE 6144
#define AEC_PREV_SIZE 256
#define AEC_DELAY_BUF_SIZE 39936

struct AecProcessor {
    OnnxModel m;
    PFFFT_Setup* fft_plan_;
    float* fft_in_;
    float* fft_out_;
    float* ifft_out_;
    float* window_;             /* sqrt-Hann */
    float* mic_history_;        /* NFFT */
    float* far_history_;        /* NFFT */
    float* ola_accumulator_;    /* NFFT */
    float* window_sum_;         /* NFFT */
    float* mic_onnx_;           /* 2050 planar */
    float* far_onnx_;           /* 2050 planar */
    size_t res_enc_conv_sz_, res_enc_tfa_sz_, mic_enc_conv_sz_, mic_enc_tfa_sz_;
    size_t deep_enc_tfa_sz_, dec_conv_sz_, dec_tfa_sz_, inter_sz_, prev_sz_, delay_buf_sz_;
    float* res_enc_conv_; float* res_enc_tfa_; float* mic_enc_conv_; float* mic_enc_tfa_;
    float* deep_enc_tfa_; float* dec_conv_; float* dec_tfa_; float* inter_;
    float* res_prev1_; float* res_prev2_; float* mic_prev1_; float* mic_prev2_;
    float* delay_buf_;
    FVec out_acc_;
    size_t out_acc_pos_;
};

static int aec_alloc_buffers(AecProcessor* a) {
    a->fft_plan_ = pffft_new_setup(AEC_NFFT, PFFFT_REAL);
    a->fft_in_ = (float*)pffft_aligned_malloc(AEC_NFFT * sizeof(float));
    a->fft_out_ = (float*)pffft_aligned_malloc(AEC_NFFT * sizeof(float));
    a->ifft_out_ = (float*)pffft_aligned_malloc(AEC_NFFT * sizeof(float));
    a->window_ = (float*)malloc(AEC_NFFT * sizeof(float));
    a->mic_history_ = (float*)calloc(AEC_NFFT, sizeof(float));
    a->far_history_ = (float*)calloc(AEC_NFFT, sizeof(float));
    a->ola_accumulator_ = (float*)calloc(AEC_NFFT, sizeof(float));
    a->window_sum_ = (float*)calloc(AEC_NFFT, sizeof(float));
    a->mic_onnx_ = (float*)calloc(AEC_SPEC_SIZE, sizeof(float));
    a->far_onnx_ = (float*)calloc(AEC_SPEC_SIZE, sizeof(float));
    if (!a->fft_plan_ || !a->fft_in_ || !a->fft_out_ || !a->ifft_out_ || !a->window_ ||
        !a->mic_history_ || !a->far_history_ || !a->ola_accumulator_ || !a->window_sum_ ||
        !a->mic_onnx_ || !a->far_onnx_) return -1;
    for (int i = 0; i < AEC_NFFT; ++i) {
        float hann = 0.5f * (1.0f - (float)cos(2.0 * M_PI * i / AEC_NFFT));
        a->window_[i] = (float)sqrt(hann);
    }
    fvec_init(&a->out_acc_);
    return 0;
}

static void aec_free_buffers(AecProcessor* a) {
    if (a->fft_plan_) pffft_destroy_setup(a->fft_plan_);
    if (a->fft_in_) pffft_aligned_free(a->fft_in_);
    if (a->fft_out_) pffft_aligned_free(a->fft_out_);
    if (a->ifft_out_) pffft_aligned_free(a->ifft_out_);
    free(a->window_);
    free(a->mic_history_);
    free(a->far_history_);
    free(a->ola_accumulator_);
    free(a->window_sum_);
    free(a->mic_onnx_);
    free(a->far_onnx_);
    free(a->res_enc_conv_);
    free(a->res_enc_tfa_);
    free(a->mic_enc_conv_);
    free(a->mic_enc_tfa_);
    free(a->deep_enc_tfa_);
    free(a->dec_conv_);
    free(a->dec_tfa_);
    free(a->inter_);
    free(a->res_prev1_);
    free(a->res_prev2_);
    free(a->mic_prev1_);
    free(a->mic_prev2_);
    free(a->delay_buf_);
    fvec_free(&a->out_acc_);
}

static void aec_compute_stft_frame(AecProcessor* a, const float* input_nfft, float* onnx_spec) {
    for (int i = 0; i < AEC_NFFT; ++i) a->fft_in_[i] = input_nfft[i] * a->window_[i];
    pffft_transform_ordered(a->fft_plan_, a->fft_in_, a->fft_out_, NULL, PFFFT_FORWARD);
    onnx_spec[0] = a->fft_out_[0];
    onnx_spec[AEC_FREQ] = 0.0f;
    for (int k = 1; k < AEC_FREQ - 1; ++k) {
        int p = 2 + (k - 1) * 2;
        onnx_spec[k] = a->fft_out_[p];
        onnx_spec[AEC_FREQ + k] = a->fft_out_[p + 1];
    }
    onnx_spec[AEC_FREQ - 1] = a->fft_out_[1];
    onnx_spec[AEC_FREQ + AEC_FREQ - 1] = 0.0f;
}

static void aec_run_onnx(AecProcessor* a) {
    OnnxModel* m = &a->m;
    size_t nin = m->n_inputs, nout = m->n_outputs;
    OrtValue* inputs[16] = {NULL};
    OrtValue* outputs[16] = {NULL};
    int64_t frame_shape[4] = {1, 2, 1, AEC_FREQ};
    int64_t s2[2];
    int64_t prev_shape[4] = {1, 1, 1, (int64_t)a->prev_sz_};
    int64_t delay_shape[4] = {1, 3, 52, 256};
    for (size_t i = 0; i < nin && i < 16; ++i) {
        const char* name = m->input_names[i];
        if (!name) continue;
        if (strcmp(name, "mic_frame") == 0) {
            inputs[i] = ort_tensor(m, a->mic_onnx_, AEC_SPEC_SIZE, frame_shape, 4);
        } else if (strcmp(name, "far_frame") == 0) {
            inputs[i] = ort_tensor(m, a->far_onnx_, AEC_SPEC_SIZE, frame_shape, 4);
        } else if (strcmp(name, "res_enc_conv") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->res_enc_conv_sz_;
            inputs[i] = ort_tensor(m, a->res_enc_conv_, a->res_enc_conv_sz_, s2, 2);
        } else if (strcmp(name, "res_enc_tfa") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->res_enc_tfa_sz_;
            inputs[i] = ort_tensor(m, a->res_enc_tfa_, a->res_enc_tfa_sz_, s2, 2);
        } else if (strcmp(name, "mic_enc_conv") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->mic_enc_conv_sz_;
            inputs[i] = ort_tensor(m, a->mic_enc_conv_, a->mic_enc_conv_sz_, s2, 2);
        } else if (strcmp(name, "mic_enc_tfa") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->mic_enc_tfa_sz_;
            inputs[i] = ort_tensor(m, a->mic_enc_tfa_, a->mic_enc_tfa_sz_, s2, 2);
        } else if (strcmp(name, "deep_enc_conv") == 0) {
            s2[0] = 1; s2[1] = 0;
            inputs[i] = ort_tensor(m, a->deep_enc_tfa_, 0, s2, 2);
        } else if (strcmp(name, "deep_enc_tfa") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->deep_enc_tfa_sz_;
            inputs[i] = ort_tensor(m, a->deep_enc_tfa_, a->deep_enc_tfa_sz_, s2, 2);
        } else if (strcmp(name, "dec_conv") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->dec_conv_sz_;
            inputs[i] = ort_tensor(m, a->dec_conv_, a->dec_conv_sz_, s2, 2);
        } else if (strcmp(name, "dec_tfa") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->dec_tfa_sz_;
            inputs[i] = ort_tensor(m, a->dec_tfa_, a->dec_tfa_sz_, s2, 2);
        } else if (strcmp(name, "inter") == 0) {
            s2[0] = 1; s2[1] = (int64_t)a->inter_sz_;
            inputs[i] = ort_tensor(m, a->inter_, a->inter_sz_, s2, 2);
        } else if (strcmp(name, "res_prev1") == 0) {
            inputs[i] = ort_tensor(m, a->res_prev1_, a->prev_sz_, prev_shape, 4);
        } else if (strcmp(name, "res_prev2") == 0) {
            inputs[i] = ort_tensor(m, a->res_prev2_, a->prev_sz_, prev_shape, 4);
        } else if (strcmp(name, "mic_prev1") == 0) {
            inputs[i] = ort_tensor(m, a->mic_prev1_, a->prev_sz_, prev_shape, 4);
        } else if (strcmp(name, "mic_prev2") == 0) {
            inputs[i] = ort_tensor(m, a->mic_prev2_, a->prev_sz_, prev_shape, 4);
        } else if (strcmp(name, "delay_buf") == 0) {
            inputs[i] = ort_tensor(m, a->delay_buf_, a->delay_buf_sz_, delay_shape, 4);
        } else {
            s2[0] = 1; s2[1] = 0;
            inputs[i] = ort_tensor(m, a->res_enc_conv_, 0, s2, 2);
        }
    }
    ort_run(m, inputs, nin, outputs, nout);

    for (size_t i = 0; i < nout && i < 16; ++i) {
        const char* name = m->output_names[i];
        if (!name || !outputs[i]) continue;
        float* data = ort_value_data(m, outputs[i]);
        size_t total = ort_value_elems(m, outputs[i]);
        if (!data) continue;
        size_t n;
        if (strcmp(name, "enhanced_frame") == 0) {
            a->fft_out_[0] = data[0];
            a->fft_out_[1] = data[AEC_FREQ - 1];
            for (int k = 1; k < AEC_FREQ - 1; ++k) {
                int p = 2 + (k - 1) * 2;
                a->fft_out_[p] = data[k];
                a->fft_out_[p + 1] = data[AEC_FREQ + k];
            }
        } else if (strcmp(name, "res_enc_conv") == 0 || strcmp(name, "res_enc_conv_o") == 0) {
            n = total < a->res_enc_conv_sz_ ? total : a->res_enc_conv_sz_;
            memcpy(a->res_enc_conv_, data, n * sizeof(float));
        } else if (strcmp(name, "res_enc_tfa") == 0 || strcmp(name, "res_enc_tfa_o") == 0) {
            n = total < a->res_enc_tfa_sz_ ? total : a->res_enc_tfa_sz_;
            memcpy(a->res_enc_tfa_, data, n * sizeof(float));
        } else if (strcmp(name, "mic_enc_conv") == 0 || strcmp(name, "mic_enc_conv_o") == 0) {
            n = total < a->mic_enc_conv_sz_ ? total : a->mic_enc_conv_sz_;
            memcpy(a->mic_enc_conv_, data, n * sizeof(float));
        } else if (strcmp(name, "mic_enc_tfa") == 0 || strcmp(name, "mic_enc_tfa_o") == 0) {
            n = total < a->mic_enc_tfa_sz_ ? total : a->mic_enc_tfa_sz_;
            memcpy(a->mic_enc_tfa_, data, n * sizeof(float));
        } else if (strcmp(name, "deep_enc_tfa") == 0 || strcmp(name, "deep_enc_tfa_o") == 0) {
            n = total < a->deep_enc_tfa_sz_ ? total : a->deep_enc_tfa_sz_;
            memcpy(a->deep_enc_tfa_, data, n * sizeof(float));
        } else if (strcmp(name, "dec_conv") == 0 || strcmp(name, "dec_conv_o") == 0) {
            n = total < a->dec_conv_sz_ ? total : a->dec_conv_sz_;
            memcpy(a->dec_conv_, data, n * sizeof(float));
        } else if (strcmp(name, "dec_tfa") == 0 || strcmp(name, "dec_tfa_o") == 0) {
            n = total < a->dec_tfa_sz_ ? total : a->dec_tfa_sz_;
            memcpy(a->dec_tfa_, data, n * sizeof(float));
        } else if (strcmp(name, "inter") == 0 || strcmp(name, "inter_o") == 0) {
            n = total < a->inter_sz_ ? total : a->inter_sz_;
            memcpy(a->inter_, data, n * sizeof(float));
        } else if (strcmp(name, "res_prev1") == 0 || strcmp(name, "res_prev1_o") == 0) {
            n = total < a->prev_sz_ ? total : a->prev_sz_;
            memcpy(a->res_prev1_, data, n * sizeof(float));
        } else if (strcmp(name, "res_prev2") == 0 || strcmp(name, "res_prev2_o") == 0) {
            n = total < a->prev_sz_ ? total : a->prev_sz_;
            memcpy(a->res_prev2_, data, n * sizeof(float));
        } else if (strcmp(name, "mic_prev1") == 0 || strcmp(name, "mic_prev1_o") == 0) {
            n = total < a->prev_sz_ ? total : a->prev_sz_;
            memcpy(a->mic_prev1_, data, n * sizeof(float));
        } else if (strcmp(name, "mic_prev2") == 0 || strcmp(name, "mic_prev2_o") == 0) {
            n = total < a->prev_sz_ ? total : a->prev_sz_;
            memcpy(a->mic_prev2_, data, n * sizeof(float));
        } else if (strcmp(name, "delay_buf") == 0 || strcmp(name, "delay_buf_o") == 0) {
            n = total < a->delay_buf_sz_ ? total : a->delay_buf_sz_;
            memcpy(a->delay_buf_, data, n * sizeof(float));
        }
    }
    for (size_t i = 0; i < nin && i < 16; ++i) if (inputs[i]) m->api->ReleaseValue(inputs[i]);
    for (size_t i = 0; i < nout && i < 16; ++i) if (outputs[i]) m->api->ReleaseValue(outputs[i]);
}

static void aec_process_one_frame(AecProcessor* a, const float* mic_1024, const float* far_1024) {
    memmove(a->mic_history_, a->mic_history_ + AEC_HOP,
            (AEC_NFFT - AEC_HOP) * sizeof(float));
    memcpy(a->mic_history_ + AEC_NFFT - AEC_HOP, mic_1024, AEC_HOP * sizeof(float));
    memmove(a->far_history_, a->far_history_ + AEC_HOP,
            (AEC_NFFT - AEC_HOP) * sizeof(float));
    memcpy(a->far_history_ + AEC_NFFT - AEC_HOP, far_1024, AEC_HOP * sizeof(float));

    aec_compute_stft_frame(a, a->mic_history_, a->mic_onnx_);
    aec_compute_stft_frame(a, a->far_history_, a->far_onnx_);

    aec_run_onnx(a);

    /* enhanced frame already written to fft_out_ (packed) -> IFFT -> OLA */
    pffft_transform_ordered(a->fft_plan_, a->fft_out_, a->ifft_out_, NULL, PFFFT_BACKWARD);
    float scale = 1.0f / AEC_NFFT;
    for (int i = 0; i < AEC_NFFT; ++i) a->ifft_out_[i] *= scale * a->window_[i];
    for (int i = 0; i < AEC_NFFT; ++i) a->ola_accumulator_[i] += a->ifft_out_[i];
    for (int i = 0; i < AEC_NFFT; ++i) a->window_sum_[i] += a->window_[i] * a->window_[i];

    for (int i = 0; i < AEC_HOP; ++i) {
        float norm = a->window_sum_[i];
        float val = (norm > 1e-6f) ? (a->ola_accumulator_[i] / norm) : a->ola_accumulator_[i];
        fvec_push(&a->out_acc_, val);
    }
    for (int i = 0; i < AEC_NFFT - AEC_HOP; ++i) {
        a->ola_accumulator_[i] = a->ola_accumulator_[i + AEC_HOP];
        a->window_sum_[i] = a->window_sum_[i + AEC_HOP];
    }
    for (int i = AEC_NFFT - AEC_HOP; i < AEC_NFFT; ++i) {
        a->ola_accumulator_[i] = 0.0f;
        a->window_sum_[i] = 0.0f;
    }
}

void aec_process_frame(AecProcessor* a, const float* mic, const float* far, float* out) {
    aec_process_one_frame(a, mic, far);
    if (a->out_acc_.len - a->out_acc_pos_ >= (size_t)AEC_HOP) {
        memcpy(out, a->out_acc_.data + a->out_acc_pos_, AEC_HOP * sizeof(float));
        a->out_acc_pos_ += AEC_HOP;
    } else {
        memset(out, 0, AEC_HOP * sizeof(float));
    }
    if (a->out_acc_pos_ >= (size_t)AEC_HOP && a->out_acc_pos_ <= a->out_acc_.len) {
        fvec_erase_front(&a->out_acc_, a->out_acc_pos_);
        a->out_acc_pos_ = 0;
    }
}

void aec_reset(AecProcessor* a) {
    size_t n = a->res_enc_conv_sz_;
    memset(a->res_enc_conv_, 0, n * sizeof(float));
    memset(a->res_enc_tfa_, 0, a->res_enc_tfa_sz_ * sizeof(float));
    memset(a->mic_enc_conv_, 0, a->mic_enc_conv_sz_ * sizeof(float));
    memset(a->mic_enc_tfa_, 0, a->mic_enc_tfa_sz_ * sizeof(float));
    memset(a->deep_enc_tfa_, 0, a->deep_enc_tfa_sz_ * sizeof(float));
    memset(a->dec_conv_, 0, a->dec_conv_sz_ * sizeof(float));
    memset(a->dec_tfa_, 0, a->dec_tfa_sz_ * sizeof(float));
    memset(a->inter_, 0, a->inter_sz_ * sizeof(float));
    memset(a->res_prev1_, 0, a->prev_sz_ * sizeof(float));
    memset(a->res_prev2_, 0, a->prev_sz_ * sizeof(float));
    memset(a->mic_prev1_, 0, a->prev_sz_ * sizeof(float));
    memset(a->mic_prev2_, 0, a->prev_sz_ * sizeof(float));
    memset(a->delay_buf_, 0, a->delay_buf_sz_ * sizeof(float));
    memset(a->mic_history_, 0, AEC_NFFT * sizeof(float));
    memset(a->far_history_, 0, AEC_NFFT * sizeof(float));
    memset(a->ola_accumulator_, 0, AEC_NFFT * sizeof(float));
    memset(a->window_sum_, 0, AEC_NFFT * sizeof(float));
    fvec_clear(&a->out_acc_);
    a->out_acc_pos_ = 0;
}

AecProcessor* aec_new(const char* model_path) {
    if (!model_path || !model_path[0]) return NULL;
    AecProcessor* a = (AecProcessor*)calloc(1, sizeof(AecProcessor));
    if (!a) return NULL;
    if (onnx_model_open(&a->m, "AecProcessor", model_path,
                        ORT_LOGGING_LEVEL_WARNING, ORT_ENABLE_BASIC) != 0) {
        free(a);
        return NULL;
    }
    if (aec_alloc_buffers(a) != 0) {
        onnx_model_close(&a->m);
        aec_free_buffers(a);
        free(a);
        return NULL;
    }
    a->res_enc_conv_sz_ = onnx_input_total(&a->m, "res_enc_conv", AEC_RES_ENC_CONV_SIZE);
    a->res_enc_tfa_sz_  = onnx_input_total(&a->m, "res_enc_tfa", AEC_RES_ENC_TFA_SIZE);
    a->mic_enc_conv_sz_ = onnx_input_total(&a->m, "mic_enc_conv", AEC_MIC_ENC_CONV_SIZE);
    a->mic_enc_tfa_sz_  = onnx_input_total(&a->m, "mic_enc_tfa", AEC_MIC_ENC_TFA_SIZE);
    a->deep_enc_tfa_sz_ = onnx_input_total(&a->m, "deep_enc_tfa", AEC_DEEP_ENC_TFA_SIZE);
    a->dec_conv_sz_     = onnx_input_total(&a->m, "dec_conv", AEC_DEC_CONV_SIZE);
    a->dec_tfa_sz_      = onnx_input_total(&a->m, "dec_tfa", AEC_DEC_TFA_SIZE);
    a->inter_sz_        = onnx_input_total(&a->m, "inter", AEC_INTER_SIZE);
    a->prev_sz_         = onnx_input_total(&a->m, "res_prev1", AEC_PREV_SIZE);
    a->delay_buf_sz_    = onnx_input_total(&a->m, "delay_buf", AEC_DELAY_BUF_SIZE);

    a->res_enc_conv_ = (float*)calloc(a->res_enc_conv_sz_, sizeof(float));
    a->res_enc_tfa_  = (float*)calloc(a->res_enc_tfa_sz_, sizeof(float));
    a->mic_enc_conv_ = (float*)calloc(a->mic_enc_conv_sz_, sizeof(float));
    a->mic_enc_tfa_  = (float*)calloc(a->mic_enc_tfa_sz_, sizeof(float));
    a->deep_enc_tfa_ = (float*)calloc(a->deep_enc_tfa_sz_, sizeof(float));
    a->dec_conv_     = (float*)calloc(a->dec_conv_sz_, sizeof(float));
    a->dec_tfa_      = (float*)calloc(a->dec_tfa_sz_, sizeof(float));
    a->inter_        = (float*)calloc(a->inter_sz_, sizeof(float));
    a->res_prev1_    = (float*)calloc(a->prev_sz_, sizeof(float));
    a->res_prev2_    = (float*)calloc(a->prev_sz_, sizeof(float));
    a->mic_prev1_    = (float*)calloc(a->prev_sz_, sizeof(float));
    a->mic_prev2_    = (float*)calloc(a->prev_sz_, sizeof(float));
    a->delay_buf_    = (float*)calloc(a->delay_buf_sz_, sizeof(float));
    if (!a->res_enc_conv_ || !a->res_enc_tfa_ || !a->mic_enc_conv_ || !a->mic_enc_tfa_ ||
        !a->deep_enc_tfa_ || !a->dec_conv_ || !a->dec_tfa_ || !a->inter_ ||
        !a->res_prev1_ || !a->res_prev2_ || !a->mic_prev1_ || !a->mic_prev2_ || !a->delay_buf_) {
        onnx_model_close(&a->m);
        aec_free_buffers(a);
        free(a);
        return NULL;
    }
    return a;
}

void aec_free(AecProcessor* a) {
    if (!a) return;
    onnx_model_close(&a->m);
    aec_free_buffers(a);
    free(a);
}

/* ═══════════════════════════════════════════════════════════════
 * Resampler - libsamplerate wrapper
 * ═══════════════════════════════════════════════════════════════
 */
struct Resampler {
    SRC_STATE* state;
    double current_ratio;
    FVec out;
};

int resampler_src_sinc_fastest(void) { return SRC_SINC_FASTEST; }

Resampler* resampler_new(int converter_type) {
    int error;
    SRC_STATE* st = src_new(converter_type, 1, &error);
    if (!st) {
        fprintf(stderr, "aimic: Resampler init failed: %s\n", src_strerror(error));
        return NULL;
    }
    Resampler* r = (Resampler*)calloc(1, sizeof(Resampler));
    if (!r) { src_delete(st); return NULL; }
    r->state = st;
    r->current_ratio = 1.0;
    fvec_init(&r->out);
    return r;
}

void resampler_free(Resampler* r) {
    if (!r) return;
    if (r->state) src_delete(r->state);
    fvec_free(&r->out);
    free(r);
}

size_t resampler_run(Resampler* r, const float* in, size_t n, double ratio, bool eof) {
    if (n == 0 && !eof) return r->out.len;
    if (ratio <= 0.0) ratio = 1.0;
    r->current_ratio = ratio;
    const float* ptr = in;
    size_t remaining = n;
    float tmp[4096];
    SRC_DATA data;
    while (remaining > 0) {
        memset(&data, 0, sizeof(data));
        data.data_in = ptr;
        data.input_frames = (long)remaining;
        data.data_out = tmp;
        data.output_frames = 4096;
        data.src_ratio = ratio;
        data.end_of_input = 0;
        int err = src_process(r->state, &data);
        if (err) { fprintf(stderr, "aimic: resampler: %s\n", src_strerror(err)); break; }
        fvec_append(&r->out, tmp, (size_t)data.output_frames_gen);
        remaining -= (size_t)data.input_frames_used;
        ptr += data.input_frames_used;
    }
    if (eof) {
        for (;;) {
            memset(&data, 0, sizeof(data));
            data.data_in = NULL;
            data.input_frames = 0;
            data.data_out = tmp;
            data.output_frames = 4096;
            data.src_ratio = ratio;
            data.end_of_input = 1;
            int err = src_process(r->state, &data);
            if (err) { fprintf(stderr, "aimic: resampler flush: %s\n", src_strerror(err)); break; }
            if (data.output_frames_gen == 0) break;
            fvec_append(&r->out, tmp, (size_t)data.output_frames_gen);
        }
    }
    return r->out.len;
}

size_t resampler_take(Resampler* r, float* out, size_t cap) {
    size_t n = r->out.len < cap ? r->out.len : cap;
    memcpy(out, r->out.data, n * sizeof(float));
    fvec_erase_front(&r->out, n);
    return n;
}

void resampler_reset(Resampler* r) {
    src_reset(r->state);
    fvec_clear(&r->out);
}

/* ═══════════════════════════════════════════════════════════════
 * AudioProcessor - unified processing chain:
 *   pre_gain → EQ → clip → [passthrough|denoise|aec|tse] → compressor
 *   → clip → VAD → AGC
 * ═══════════════════════════════════════════════════════════════
 */
struct AudioProcessor {
    float pre_gain_;
    BiquadCoeff* eq_filters_;        /* EQ_BANDS */
    int mode_;
    bool eq_active_;
    DenoiseProcessor* denoise_;
    TseProcessor* tse_;
    AecProcessor* aec_;
    StftProcessor stft_;
    int far_sample_rate_;
    Resampler* far_resampler_;
    float far_rms_target_;
    int io_in_sr_, io_out_sr_;
    FVec io_in_acc_;
    FVec io_out_acc_;
    FVec viz_in_48k_;
    FVec viz_out_48k_;
    VadGate* vad_gate_;
    bool vad_enabled_;
    AgcController* agc_;
    bool agc_enabled_;
    Compressor* compressor_;
    bool compressor_enabled_;
    FVec tse_recording_buffer_;
    bool recording_enabled_;
    int backend_effective_;
    int backend_reason_;
    NoiseFloorTracker* noise_tracker_;
    float noise_gate_offset_db_;
    bool noise_gate_enabled_;
};

static void ap_apply_eq(AudioProcessor* ap, float* data, size_t len) {
    for (size_t nd = 0; nd < len; ++nd) {
        float y = data[nd];
        for (int b = 0; b < EQ_BANDS; ++b) {
            BiquadCoeff* f = &ap->eq_filters_[b];
            float x = y;
            y = f->b0 * x + f->b1 * f->x1 + f->b2 * f->x2
                - f->a1 * f->y1 - f->a2 * f->y2;
            f->x2 = f->x1;
            f->x1 = x;
            f->y2 = f->y1;
            f->y1 = y;
        }
        data[nd] = y;
    }
}

static void ap_apply_pre_gain(AudioProcessor* ap, float* buf) {
    float g = ap->agc_enabled_ ? agc_tick(ap->agc_) : ap->pre_gain_;
    for (size_t i = 0; i < HOP_LENGTH; ++i) buf[i] *= g;
}

static void ap_apply_eq_clip(AudioProcessor* ap, float* buf) {
    if (ap->eq_active_) ap_apply_eq(ap, buf, HOP_LENGTH);
    clip_buffer(buf, HOP_LENGTH);
}

static void ap_measure_agc_rms(AudioProcessor* ap, float* out) {
    float sq = 0.0f;
    for (size_t i = 0; i < HOP_LENGTH; ++i) sq += out[i] * out[i];
    float rms = (float)sqrt(sq / (float)HOP_LENGTH);
    agc_update_rms(ap->agc_, rms);
}

static void ap_aec_step(AudioProcessor* ap, const float* buf, const float* far,
                        size_t far_n, float* out) {
    if (!ap->aec_ || !far || far_n == 0) {
        memcpy(out, buf, HOP_LENGTH * sizeof(float));
        return;
    }
    if (ap->far_resampler_ && ap->far_sample_rate_ != 48000) {
        double ratio = 48000.0 / (double)ap->far_sample_rate_;
        resampler_run(ap->far_resampler_, far, far_n, ratio, false);
        float far2[HOP_LENGTH];
        if (resampler_take(ap->far_resampler_, far2, HOP_LENGTH) == HOP_LENGTH) {
            aec_process_frame(ap->aec_, buf, far2, out);
            return;
        }
    } else if (far_n >= HOP_LENGTH) {
        aec_process_frame(ap->aec_, buf, far, out);
        return;
    }
    memcpy(out, buf, HOP_LENGTH * sizeof(float));
}

static void ap_tse_step(AudioProcessor* ap, const float* buf, float* out) {
    if (!ap->tse_ || !tse_has_reference(ap->tse_)) {
        memcpy(out, buf, HOP_LENGTH * sizeof(float));
        return;
    }
    float spec[STFT_SPEC_FLOATS];
    float denoised[HOP_LENGTH];
    if (ap->denoise_) {
        denoise_process_chunk(ap->denoise_, buf, denoised);
        stft_forward(&ap->stft_, denoised, spec);
    } else {
        stft_forward(&ap->stft_, buf, spec);
    }
    tse_process_spec_freq(ap->tse_, spec, spec);
    stft_backward(&ap->stft_, spec, out);
}

AudioProcessor* audio_processor_new(float pre_gain_db, const char* denoise_model_path,
                                    const char* tse_model_path, const char* aec_model_path) {
    AudioProcessor* ap = (AudioProcessor*)calloc(1, sizeof(AudioProcessor));
    if (!ap) return NULL;
    ap->pre_gain_ = (float)pow(10.0, pre_gain_db / 20.0);
    ap->mode_ = AIMIC_MODE_DENOISE;
    ap->eq_active_ = false;
    ap->vad_enabled_ = false;
    ap->agc_enabled_ = false;
    ap->compressor_enabled_ = false;
    ap->recording_enabled_ = false;
    ap->far_sample_rate_ = 48000;
    ap->far_rms_target_ = 0.05f;
    ap->io_in_sr_ = 48000;
    ap->io_out_sr_ = 48000;
    ap->backend_effective_ = AIMIC_BACKEND_AVX;
    ap->backend_reason_ = AIMIC_BACKEND_REASON_OK;

    ap->eq_filters_ = (BiquadCoeff*)calloc(EQ_BANDS, sizeof(BiquadCoeff));
    if (!ap->eq_filters_) { free(ap); return NULL; }
    for (int i = 0; i < EQ_BANDS; ++i)
        ap->eq_filters_[i] = design_peaking_eq(EQ_FREQS[i], 0.0f, EQ_Q, SAMPLE_RATE);

    fvec_init(&ap->io_in_acc_);
    fvec_init(&ap->io_out_acc_);
    fvec_init(&ap->viz_in_48k_);
    fvec_init(&ap->viz_out_48k_);
    fvec_init(&ap->tse_recording_buffer_);

    ap->vad_gate_ = vad_new(-45.0f, 20.0f, 250.0f, 48000.0f, 480);
    ap->agc_ = agc_new(-20.0f, 10.0f);
    ap->compressor_ = compressor_new(-20.0f, 3.0f, 15.0f, 180.0f, 8.0f, 4.0f, 48000.0f);
    ap->noise_tracker_ = noise_floor_tracker_new(10.0f);
    ap->noise_gate_offset_db_ = 3.0f;
    ap->noise_gate_enabled_ = false;
    if (stft_init(&ap->stft_) != 0) {
        audio_processor_free(ap);
        return NULL;
    }

    if (tse_model_path && tse_model_path[0]) ap->tse_ = tse_new(tse_model_path);
    if (aec_model_path && aec_model_path[0]) ap->aec_ = aec_new(aec_model_path);
    if (denoise_model_path && denoise_model_path[0]) ap->denoise_ = denoise_new(denoise_model_path);
    /* Report the effective backend / fallback reason (all models share the
     * same backend, so use whichever one is loaded). */
    if (ap->denoise_) {
        ap->backend_effective_ = ap->denoise_->m.effective_backend_;
        ap->backend_reason_ = ap->denoise_->m.backend_reason_;
    } else if (ap->tse_) {
        ap->backend_effective_ = ap->tse_->m.effective_backend_;
        ap->backend_reason_ = ap->tse_->m.backend_reason_;
    } else if (ap->aec_) {
        ap->backend_effective_ = ap->aec_->m.effective_backend_;
        ap->backend_reason_ = ap->aec_->m.backend_reason_;
    }
    return ap;
}

void audio_processor_cleanup(AudioProcessor* ap) {
    if (!ap) return;
    if (ap->denoise_) { denoise_free(ap->denoise_); ap->denoise_ = NULL; }
    if (ap->tse_) { tse_free(ap->tse_); ap->tse_ = NULL; }
    if (ap->aec_) { aec_free(ap->aec_); ap->aec_ = NULL; }
}

int audio_processor_backend_effective(const AudioProcessor* ap) {
    return ap ? ap->backend_effective_ : AIMIC_BACKEND_AVX;
}

int audio_processor_backend_reason(const AudioProcessor* ap) {
    return ap ? ap->backend_reason_ : AIMIC_BACKEND_REASON_OK;
}

void audio_processor_free(AudioProcessor* ap) {
    if (!ap) return;
    audio_processor_cleanup(ap);
    if (ap->far_resampler_) { resampler_free(ap->far_resampler_); ap->far_resampler_ = NULL; }
    if (ap->vad_gate_) vad_free(ap->vad_gate_);
    if (ap->agc_) agc_free(ap->agc_);
    if (ap->compressor_) compressor_free(ap->compressor_);
    free(ap->eq_filters_);
    stft_free(&ap->stft_);
    fvec_free(&ap->io_in_acc_);
    fvec_free(&ap->io_out_acc_);
    fvec_free(&ap->viz_in_48k_);
    fvec_free(&ap->viz_out_48k_);
    fvec_free(&ap->tse_recording_buffer_);
    free(ap);
}

/* -- AudioProcessor: EQ / gain -- */
void audio_processor_set_eq_gains(AudioProcessor* ap, const float* gains, size_t n) {
    bool any_nonzero = false;
    for (int i = 0; i < EQ_BANDS; ++i) {
        float g = (i < (int)n) ? gains[i] : 0.0f;
        if (g != 0.0f) any_nonzero = true;
        ap->eq_filters_[i] = design_peaking_eq(EQ_FREQS[i], g, EQ_Q, SAMPLE_RATE);
    }
    ap->eq_active_ = any_nonzero;
}

void audio_processor_get_eq_freqs(AudioProcessor* ap, float* freqs) {
    (void)ap;
    memcpy(freqs, EQ_FREQS, EQ_BANDS * sizeof(float));
}

int audio_processor_get_eq_band_count(AudioProcessor* ap) { (void)ap; return EQ_BANDS; }

size_t audio_processor_process_eq_only(AudioProcessor* ap, const float* in, size_t n, float* out) {
    memcpy(out, in, n * sizeof(float));
    float g = ap->agc_enabled_ ? agc_tick(ap->agc_) : ap->pre_gain_;
    for (size_t i = 0; i < n; ++i) out[i] *= g;
    if (ap->eq_active_) ap_apply_eq(ap, out, n);
    return n;
}

void audio_processor_set_pre_gain(AudioProcessor* ap, float gain_db) {
    ap->pre_gain_ = (float)pow(10.0, gain_db / 20.0);
}

/* -- AudioProcessor: mode -- */
void audio_processor_set_mode(AudioProcessor* ap, int mode) {
    ap->mode_ = mode;
    stft_reset(&ap->stft_);
    if (mode == AIMIC_MODE_TSE && ap->tse_) tse_reset(ap->tse_);
}

int audio_processor_get_mode(AudioProcessor* ap) { return ap->mode_; }

void audio_processor_set_tse_enabled(AudioProcessor* ap, bool en) {
    if (en) audio_processor_set_mode(ap, AIMIC_MODE_TSE);
    else if (ap->mode_ == AIMIC_MODE_TSE) audio_processor_set_mode(ap, AIMIC_MODE_PASSTHROUGH);
}

void audio_processor_set_aec_enabled(AudioProcessor* ap, bool en) {
    if (en) audio_processor_set_mode(ap, AIMIC_MODE_AEC);
    else if (ap->mode_ == AIMIC_MODE_AEC) audio_processor_set_mode(ap, AIMIC_MODE_PASSTHROUGH);
}

void audio_processor_set_aec_far_sample_rate(AudioProcessor* ap, int sr) {
    if (sr <= 0) sr = 48000;
    ap->far_sample_rate_ = sr;
    if (sr != 48000 && ap->aec_) {
        if (ap->far_resampler_) resampler_free(ap->far_resampler_);
        ap->far_resampler_ = resampler_new(resampler_src_sinc_fastest());
        double ratio = 48000.0 / (double)sr;
        float* silence = (float*)calloc(HOP_LENGTH, sizeof(float));
        if (!silence) return;
        resampler_run(ap->far_resampler_, silence, HOP_LENGTH, ratio, false);
        free(silence);
    } else {
        if (ap->far_resampler_) { resampler_free(ap->far_resampler_); ap->far_resampler_ = NULL; }
    }
}

int audio_processor_get_aec_far_sample_rate(AudioProcessor* ap) { return ap->far_sample_rate_; }

void audio_processor_set_aec_far_rms_target(AudioProcessor* ap, float rms) {
    ap->far_rms_target_ = (rms > 0.0f) ? rms : 0.05f;
}

float audio_processor_get_aec_far_rms_target(AudioProcessor* ap) { return ap->far_rms_target_; }

bool audio_processor_is_aec_available(AudioProcessor* ap) { return ap->aec_ != NULL; }

/* ── AudioProcessor：TSE ── */
void audio_processor_set_tse_reference(AudioProcessor* ap, const float* data, size_t n) {
    if (ap->tse_) tse_set_reference(ap->tse_, data, n);
}

bool audio_processor_is_tse_reference_loaded(AudioProcessor* ap) {
    return ap->tse_ && tse_has_reference(ap->tse_);
}

bool audio_processor_is_tse_available(AudioProcessor* ap) { return ap->tse_ != NULL; }

size_t audio_processor_get_tse_recording_audio_size(AudioProcessor* ap) {
    return ap->tse_recording_buffer_.len;
}

void audio_processor_get_tse_recording_audio(AudioProcessor* ap, float* out) {
    memcpy(out, ap->tse_recording_buffer_.data,
           ap->tse_recording_buffer_.len * sizeof(float));
}

/* -- AudioProcessor: VAD / AGC / recording -- */
void audio_processor_set_vad_enabled(AudioProcessor* ap, bool enabled) {
    if (enabled && !ap->vad_enabled_) vad_reset(ap->vad_gate_);
    ap->vad_enabled_ = enabled;
}

bool audio_processor_is_vad_enabled(AudioProcessor* ap) { return ap->vad_enabled_; }
bool audio_processor_is_vad_active(AudioProcessor* ap) { return vad_is_active(ap->vad_gate_); }
void audio_processor_set_vad_threshold(AudioProcessor* ap, float dbfs) {
    vad_set_threshold(ap->vad_gate_, dbfs);
}
float audio_processor_get_vad_threshold(AudioProcessor* ap) { return vad_threshold_dbfs(ap->vad_gate_); }

void audio_processor_set_agc_enabled(AudioProcessor* ap, bool enabled, float initial_gain_db) {
    agc_set_enabled(ap->agc_, enabled, initial_gain_db);
    ap->agc_enabled_ = enabled;
}

bool audio_processor_is_agc_enabled(AudioProcessor* ap) { return ap->agc_enabled_; }
bool audio_processor_is_agc_voice_active(AudioProcessor* ap) { return agc_is_voice_active(ap->agc_); }
float audio_processor_get_agc_gain_db(AudioProcessor* ap) { return agc_get_current_gain_db(ap->agc_); }
void audio_processor_set_agc_target(AudioProcessor* ap, float dbfs) { agc_set_target(ap->agc_, dbfs); }
float audio_processor_get_agc_target(AudioProcessor* ap) { return agc_target_dbfs(ap->agc_); }

void audio_processor_set_agc_attack_ms(AudioProcessor* ap, float ms) {
    if (ap->agc_) agc_set_attack_ms(ap->agc_, ms);
}
float audio_processor_get_agc_attack_ms(AudioProcessor* ap) {
    return ap->agc_ ? agc_get_attack_ms(ap->agc_) : 10.0f;
}

void audio_processor_set_agc_release_ms(AudioProcessor* ap, float ms) {
    if (ap->agc_) agc_set_release_ms(ap->agc_, ms);
}
float audio_processor_get_agc_release_ms(AudioProcessor* ap) {
    return ap->agc_ ? agc_get_release_ms(ap->agc_) : 150.0f;
}

void audio_processor_set_noise_gate_enabled(AudioProcessor* ap, bool en) {
    ap->noise_gate_enabled_ = en;
}
bool audio_processor_is_noise_gate_enabled(AudioProcessor* ap) {
    return ap->noise_gate_enabled_;
}

void audio_processor_set_noise_gate_offset_db(AudioProcessor* ap, float db) {
    if (db < 0.0f) db = 0.0f;
    if (db > 30.0f) db = 30.0f;
    ap->noise_gate_offset_db_ = db;
}
float audio_processor_get_noise_gate_offset_db(AudioProcessor* ap) {
    return ap->noise_gate_offset_db_;
}

float audio_processor_get_noise_floor_db(AudioProcessor* ap) {
    return ap->noise_tracker_ ? noise_floor_tracker_get_floor_db(ap->noise_tracker_) : -60.0f;
}

void audio_processor_set_recording_enabled(AudioProcessor* ap, bool enabled) {
    ap->recording_enabled_ = enabled;
}

bool audio_processor_is_recording_enabled(AudioProcessor* ap) { return ap->recording_enabled_; }

/* ── AudioProcessor：Compressor ── */
void audio_processor_set_compressor_enabled(AudioProcessor* ap, bool enabled) {
    ap->compressor_enabled_ = enabled;
    compressor_set_enabled(ap->compressor_, enabled);
}
bool audio_processor_is_compressor_enabled(AudioProcessor* ap) { return ap->compressor_enabled_; }
void audio_processor_set_compressor_threshold(AudioProcessor* ap, float db) { compressor_set_threshold(ap->compressor_, db); }
float audio_processor_get_compressor_threshold(AudioProcessor* ap) { return compressor_get_threshold(ap->compressor_); }
void audio_processor_set_compressor_ratio(AudioProcessor* ap, float r) { compressor_set_ratio(ap->compressor_, r); }
float audio_processor_get_compressor_ratio(AudioProcessor* ap) { return compressor_get_ratio(ap->compressor_); }
void audio_processor_set_compressor_attack(AudioProcessor* ap, float ms) { compressor_set_attack_ms(ap->compressor_, ms); }
float audio_processor_get_compressor_attack(AudioProcessor* ap) { return compressor_get_attack_ms(ap->compressor_); }
void audio_processor_set_compressor_release(AudioProcessor* ap, float ms) { compressor_set_release_ms(ap->compressor_, ms); }
float audio_processor_get_compressor_release(AudioProcessor* ap) { return compressor_get_release_ms(ap->compressor_); }
void audio_processor_set_compressor_makeup(AudioProcessor* ap, float db) { compressor_set_makeup(ap->compressor_, db); }
float audio_processor_get_compressor_makeup(AudioProcessor* ap) { return compressor_get_makeup(ap->compressor_); }
void audio_processor_set_compressor_knee(AudioProcessor* ap, float db) { compressor_set_knee(ap->compressor_, db); }
float audio_processor_get_compressor_knee(AudioProcessor* ap) { return compressor_get_knee(ap->compressor_); }

/* ── AudioProcessor：process / pipeline ── */
size_t audio_processor_process(AudioProcessor* ap, const float* in, size_t n,
                               const float* far, size_t far_n, float* out) {
    if (n != HOP_LENGTH) return 0;
    float buf[HOP_LENGTH];
    memcpy(buf, in, HOP_LENGTH * sizeof(float));

    ap_apply_pre_gain(ap, buf);
    ap_apply_eq_clip(ap, buf);

    float out_buf[HOP_LENGTH];
    switch (ap->mode_) {
    case AIMIC_MODE_PASSTHROUGH:
        memcpy(out_buf, buf, HOP_LENGTH * sizeof(float));
        break;
    case AIMIC_MODE_DENOISE:
        if (ap->denoise_) {
            denoise_process_chunk(ap->denoise_, buf, out_buf);
        } else {
            memcpy(out_buf, buf, HOP_LENGTH * sizeof(float));
        }
        break;
    case AIMIC_MODE_AEC:
        if (ap->denoise_) {
            float denoised[HOP_LENGTH];
            denoise_process_chunk(ap->denoise_, buf, denoised);
            memcpy(buf, denoised, HOP_LENGTH * sizeof(float));
        }
        ap_aec_step(ap, buf, far, far_n, out_buf);
        break;
    case AIMIC_MODE_TSE:
        ap_tse_step(ap, buf, out_buf);
        break;
    default:
        memcpy(out_buf, buf, HOP_LENGTH * sizeof(float));
        break;
    }

    if (ap->compressor_enabled_) compressor_process(ap->compressor_, out_buf, HOP_LENGTH);
    if (ap->recording_enabled_ && ap->mode_ != AIMIC_MODE_TSE) {
        fvec_clear(&ap->tse_recording_buffer_);
        fvec_append(&ap->tse_recording_buffer_, out_buf, HOP_LENGTH);
    } else if (!ap->recording_enabled_) {
        fvec_clear(&ap->tse_recording_buffer_);
    }
    clip_buffer(out_buf, HOP_LENGTH);
    if (ap->vad_enabled_) vad_process(ap->vad_gate_, out_buf, HOP_LENGTH);
    if (ap->agc_enabled_) ap_measure_agc_rms(ap, out_buf);
    memcpy(out, out_buf, HOP_LENGTH * sizeof(float));
    return HOP_LENGTH;
}

void audio_processor_set_io_sample_rates(AudioProcessor* ap, int in_sr, int out_sr) {
    ap->io_in_sr_ = in_sr;
    ap->io_out_sr_ = out_sr;
    fvec_clear(&ap->io_in_acc_);
    fvec_clear(&ap->io_out_acc_);
}

size_t audio_processor_process_pipeline(AudioProcessor* ap, const float* in, size_t n,
                                        const float* far, size_t far_n) {
    if (n == 0) return ap->io_out_acc_.len;
    fvec_append(&ap->io_in_acc_, in, n);
    while (ap->io_in_acc_.len >= HOP_LENGTH) {
        float chunk[HOP_LENGTH];
        memcpy(chunk, ap->io_in_acc_.data, HOP_LENGTH * sizeof(float));
        fvec_erase_front(&ap->io_in_acc_, HOP_LENGTH);
        fvec_append(&ap->viz_in_48k_, chunk, HOP_LENGTH);
        float p[HOP_LENGTH];
        audio_processor_process(ap, chunk, HOP_LENGTH, far, far_n, p);
        fvec_append(&ap->viz_out_48k_, p, HOP_LENGTH);
        fvec_append(&ap->io_out_acc_, p, HOP_LENGTH);
    }
    if (ap->io_in_acc_.len >= HOP_LENGTH * 3 / 4) {
        size_t orig_sz = ap->io_in_acc_.len;
        fvec_fill(&ap->io_in_acc_, HOP_LENGTH, 0.0f);
        fvec_append(&ap->viz_in_48k_, ap->io_in_acc_.data, orig_sz);
        float p[HOP_LENGTH];
        audio_processor_process(ap, ap->io_in_acc_.data, HOP_LENGTH, far, far_n, p);
        fvec_append(&ap->viz_out_48k_, p, HOP_LENGTH);
        fvec_append(&ap->io_out_acc_, p, HOP_LENGTH);
        fvec_clear(&ap->io_in_acc_);
    }
    return ap->io_out_acc_.len;
}

size_t audio_processor_pipeline_take(AudioProcessor* ap, float* out, size_t cap) {
    size_t n = ap->io_out_acc_.len < cap ? ap->io_out_acc_.len : cap;
    if (n) memcpy(out, ap->io_out_acc_.data, n * sizeof(float));
    fvec_erase_front(&ap->io_out_acc_, n);
    return n;
}

size_t audio_processor_viz_input_take(AudioProcessor* ap, float* out, size_t cap) {
    size_t n = ap->viz_in_48k_.len < cap ? ap->viz_in_48k_.len : cap;
    if (n) memcpy(out, ap->viz_in_48k_.data, n * sizeof(float));
    fvec_erase_front(&ap->viz_in_48k_, n);
    return n;
}

size_t audio_processor_viz_output_take(AudioProcessor* ap, float* out, size_t cap) {
    size_t n = ap->viz_out_48k_.len < cap ? ap->viz_out_48k_.len : cap;
    if (n) memcpy(out, ap->viz_out_48k_.data, n * sizeof(float));
    fvec_erase_front(&ap->viz_out_48k_, n);
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * spectrum - 128-band Mel, matching human perception
 * ═══════════════════════════════════════════════════════════════
 */
#define SPECTRUM_FFT_SIZE 2048
#define SPECTRUM_SAMPLE_RATE 48000.0f
#define MEL_LOW_FREQ 20.0f
#define MEL_HIGH_FREQ 20000.0f

static float* mel_filterbank_weights = NULL;
static int* mel_filterbank_starts = NULL;
static int* mel_filterbank_ends = NULL;
static int mel_num_bins = 0;
static bool mel_initialized = false;

static inline float hz_to_mel(float hz) {
    return 2595.0f * (float)log10(1.0f + hz / 700.0f);
}
static inline float mel_to_hz(float mel) {
    return 700.0f * ((float)pow(10.0, mel / 2595.0f) - 1.0f);
}

static void init_mel_filterbank(void) {
    if (mel_initialized) return;
    int n_fft = SPECTRUM_FFT_SIZE;
    mel_num_bins = n_fft / 2 + 1;
    float f_max = SPECTRUM_SAMPLE_RATE / 2.0f;
    float mel_max = hz_to_mel(MEL_HIGH_FREQ);
    float mel_min = hz_to_mel(MEL_LOW_FREQ);

    float* center_freqs = (float*)malloc((AIMIC_SPECTRUM_NUM_BANDS + 2) * sizeof(float));
    if (!center_freqs) return;
    for (int i = 0; i < AIMIC_SPECTRUM_NUM_BANDS + 2; ++i) {
        float melv = mel_min + (mel_max - mel_min) * i / (AIMIC_SPECTRUM_NUM_BANDS + 1);
        center_freqs[i] = mel_to_hz(melv);
    }

    /* each FFT bin -> (band, weight) sparse table */
    float* weights = NULL;
    int* starts = NULL;
    int* ends = NULL;
    int* cnt = NULL;
    /* first use count array to determine entries per bin */
    starts = (int*)calloc(mel_num_bins + 1, sizeof(int));
    cnt = (int*)calloc(mel_num_bins, sizeof(int));
    if (!starts || !cnt) goto fail;
    for (int b = 0; b < AIMIC_SPECTRUM_NUM_BANDS; ++b) {
        float fl = center_freqs[b];
        float fc = center_freqs[b + 1];
        float fr = center_freqs[b + 2];
        for (int k = 0; k < mel_num_bins; ++k) {
            float freq = (float)k * SPECTRUM_SAMPLE_RATE / n_fft;
            int active = 0;
            if (freq >= fl && freq <= fc && fc > fl) active = 1;
            else if (freq > fc && freq <= fr && fr > fc) active = 1;
            if (active) cnt[k]++;
        }
    }
    int total = 0;
    for (int k = 0; k < mel_num_bins; ++k) { starts[k] = total; total += cnt[k]; }
    starts[mel_num_bins] = total;
    weights = (float*)calloc(total ? total : 1, sizeof(float));
    ends = (int*)calloc(total ? total : 1, sizeof(int));
    if (!weights || !ends) goto fail;
    /* fill in */
    /* use per-bin position-mark array */
    {
        int* pos = (int*)calloc(mel_num_bins, sizeof(int));
        if (!pos) goto fail;
        for (int b = 0; b < AIMIC_SPECTRUM_NUM_BANDS; ++b) {
            float fl = center_freqs[b], fc = center_freqs[b + 1], fr = center_freqs[b + 2];
            for (int k = 0; k < mel_num_bins; ++k) {
                float freq = (float)k * SPECTRUM_SAMPLE_RATE / n_fft;
                float w = 0.0f;
                if (freq >= fl && freq <= fc && fc > fl) w = (freq - fl) / (fc - fl);
                else if (freq > fc && freq <= fr && fr > fc) w = (fr - freq) / (fr - fc);
                if (w > 0.0f) {
                    int idx = starts[k] + pos[k];
                    weights[idx] = w;
                    ends[idx] = b;
                    pos[k]++;
                }
            }
        }
        free(pos);
    }
    free(cnt);
    free(center_freqs);
    mel_filterbank_weights = weights;
    mel_filterbank_starts = starts;
    mel_filterbank_ends = ends;
    mel_initialized = true;
    return;

fail:
    free(center_freqs);
    free(starts);
    free(cnt);
    free(weights);
    free(ends);
}

size_t compute_spectrum(const float* samples, size_t n, float* out) {
    init_mel_filterbank();
    if (!mel_initialized) {
        for (int i = 0; i < AIMIC_SPECTRUM_NUM_BANDS; ++i) out[i] = -90.0f;
        return AIMIC_SPECTRUM_NUM_BANDS;
    }

    static PFFFT_Setup* spectrum_setup = NULL;
    static float* fft_in = NULL;
    static float* fft_out = NULL;
    static float* buf = NULL;
    static const int nfft = SPECTRUM_FFT_SIZE;
    if (!spectrum_setup) {
        spectrum_setup = pffft_new_setup(nfft, PFFFT_REAL);
        if (!spectrum_setup) goto fail;
        fft_in = (float*)pffft_aligned_malloc(nfft * sizeof(float));
        fft_out = (float*)pffft_aligned_malloc(nfft * sizeof(float));
        buf = (float*)pffft_aligned_malloc(nfft * sizeof(float));
        if (!fft_in || !fft_out || !buf) goto fail;
    }

    int copy_len = (int)(n < (size_t)nfft ? n : (size_t)nfft);
    memset(buf, 0, nfft * sizeof(float));
    if (copy_len > 0) {
        int start = (int)n - copy_len;
        memcpy(buf, samples + start, copy_len * sizeof(float));
    }
    for (int i = 0; i < nfft; ++i) {
        float w = 0.5f - 0.5f * (float)cos(2.0 * M_PI * i / nfft);
        buf[i] *= w;
    }
    memcpy(fft_in, buf, nfft * sizeof(float));
    pffft_transform_ordered(spectrum_setup, fft_in, fft_out, NULL, PFFFT_FORWARD);

    int num_bins = nfft / 2 + 1;
    float scale = 1.0f / (float)(nfft * nfft);
    float* power = (float*)malloc(num_bins * sizeof(float));
    if (!power) goto fail;
    for (int i = 0; i < num_bins; ++i) power[i] = 0.0f;
    power[0] = fft_out[0] * fft_out[0] * scale;
    for (int k = 1; k < nfft / 2; ++k) {
        float re = fft_out[2 * k];
        float im = fft_out[2 * k + 1];
        power[k] = (re * re + im * im) * scale;
    }
    power[nfft / 2] = fft_out[1] * fft_out[1] * scale;

    float* mel_energy = (float*)calloc(AIMIC_SPECTRUM_NUM_BANDS, sizeof(float));
    if (!mel_energy) { free(power); goto fail; }
    for (int k = 0; k < mel_num_bins; ++k) {
        int start_idx = mel_filterbank_starts[k];
        int end_idx = mel_filterbank_starts[k + 1];
        for (int j = start_idx; j < end_idx; ++j) {
            int band = mel_filterbank_ends[j];
            mel_energy[band] += power[k] * mel_filterbank_weights[j];
        }
    }
    for (int i = 0; i < AIMIC_SPECTRUM_NUM_BANDS; ++i) {
        out[i] = -90.0f;
        if (mel_energy[i] > 1e-12f) {
            float db = 10.0f * (float)log10(mel_energy[i]);
            out[i] = fmaxf(-90.0f, fminf(-20.0f, db));
        }
    }
    free(power);
    free(mel_energy);
    return AIMIC_SPECTRUM_NUM_BANDS;

fail:
    for (int i = 0; i < AIMIC_SPECTRUM_NUM_BANDS; ++i) out[i] = -90.0f;
    return 0;
}

void spectrum_warmup(void) {
    init_mel_filterbank();
    static const int n = SPECTRUM_FFT_SIZE;
    static PFFFT_Setup* setup = NULL;
    static float *in = NULL, *out = NULL;
    if (!setup) {
        setup = pffft_new_setup(n, PFFFT_REAL);
        if (setup) {
            in = (float*)pffft_aligned_malloc(n * sizeof(float));
            out = (float*)pffft_aligned_malloc(n * sizeof(float));
            if (!in || !out) {
                pffft_aligned_free(in);
                pffft_aligned_free(out);
                in = out = NULL;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * RingBuffer - thread-safe FIFO (high-frequency read/write from Python)
 * ═══════════════════════════════════════════════════════════════
 */
struct RingBuffer {
    pv_mutex_t mutex_;
    float* buffer_;
    size_t capacity_;
    size_t write_pos_;
    size_t read_pos_;
    size_t count_;
};

RingBuffer* ringbuffer_new(size_t capacity) {
    RingBuffer* rb = (RingBuffer*)calloc(1, sizeof(RingBuffer));
    if (!rb) return NULL;
    rb->buffer_ = (float*)calloc(capacity ? capacity : 1, sizeof(float));
    if (!rb->buffer_) { free(rb); return NULL; }
    rb->capacity_ = capacity;
    pv_mutex_init(&rb->mutex_);
    return rb;
}

void ringbuffer_free(RingBuffer* rb) {
    if (!rb) return;
    pv_mutex_destroy(&rb->mutex_);
    free(rb->buffer_);
    free(rb);
}

void ringbuffer_write(RingBuffer* rb, const float* data, size_t n) {
    pv_mutex_lock(&rb->mutex_);
    if (n > rb->capacity_) n = rb->capacity_;
    for (size_t i = 0; i < n; ++i) {
        rb->buffer_[rb->write_pos_++] = data[i];
        if (rb->write_pos_ >= rb->capacity_) rb->write_pos_ = 0;
        if (rb->count_ < rb->capacity_) rb->count_++;
        else rb->read_pos_ = rb->write_pos_;
    }
    pv_mutex_unlock(&rb->mutex_);
}

size_t ringbuffer_read(RingBuffer* rb, float* dest, size_t n) {
    pv_mutex_lock(&rb->mutex_);
    if (n > rb->count_) n = rb->count_;
    for (size_t i = 0; i < n; ++i) {
        dest[i] = rb->buffer_[rb->read_pos_++];
        if (rb->read_pos_ >= rb->capacity_) rb->read_pos_ = 0;
        rb->count_--;
    }
    pv_mutex_unlock(&rb->mutex_);
    return n;
}

size_t ringbuffer_available(const RingBuffer* rb) {
    size_t c;
    pv_mutex_lock((pv_mutex_t*)&rb->mutex_);
    c = rb->count_;
    pv_mutex_unlock((pv_mutex_t*)&rb->mutex_);
    return c;
}

void ringbuffer_clear(RingBuffer* rb) {
    pv_mutex_lock(&rb->mutex_);
    rb->write_pos_ = rb->read_pos_ = rb->count_ = 0;
    memset(rb->buffer_, 0, rb->capacity_ * sizeof(float));
    pv_mutex_unlock(&rb->mutex_);
}
