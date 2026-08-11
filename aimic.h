/* PureVox — AI 麦克风降噪工具
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
 * PureVox 的纯 C 音频引擎公共 API。核心实现见 aimic.c，ctypes 绑定见 aimic.py。
 */

#ifndef AIMIC_H
#define AIMIC_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AIMIC_HOP_LENGTH 1024
#define AIMIC_EQ_BANDS 61
#define AIMIC_SPECTRUM_NUM_BANDS 128

/* ── 模式 ── */
#define AIMIC_MODE_PASSTHROUGH 0
#define AIMIC_MODE_DENOISE 1
#define AIMIC_MODE_AEC 2
#define AIMIC_MODE_TSE 3

/* ── inference backend (auto-selected) ── */
#define AIMIC_BACKEND_AVX 0  /* default CPU EP, ORT picks best ISA (AVX if CPU has it) */
#define AIMIC_BACKEND_SSE 1  /* default CPU EP on CPUs without AVX */
#define AIMIC_BACKEND_NPU 2  /* NPU execution provider active */

/* Effective backend status: reason says why NPU is not in use (0 = NPU active) */
#define AIMIC_BACKEND_REASON_OK              0
#define AIMIC_BACKEND_REASON_NPU_UNAVAILABLE 1  /* NPU EP append failed (not compiled into runtime) */
#define AIMIC_BACKEND_REASON_NPU_NO_ENTRY    2  /* no NPU EP entry point on this platform */

/* 不透明类型 */
typedef struct VadGate        VadGate;
typedef struct AgcController  AgcController;
typedef struct Compressor     Compressor;
typedef struct DenoiseProcessor DenoiseProcessor;
typedef struct TseProcessor   TseProcessor;
typedef struct AecProcessor   AecProcessor;
typedef struct AudioProcessor AudioProcessor;
typedef struct NoiseFloorTracker NoiseFloorTracker;
typedef struct Resampler      Resampler;
typedef struct RingBuffer     RingBuffer;

/* ─────────────────────── VAD ─────────────────────── */
VadGate*       vad_new(float threshold_dbfs, float onset_ms, float hang_ms, float fs, int hop);
void           vad_free(VadGate*);
void           vad_reset(VadGate*);
int            vad_process(VadGate*, float* samples, size_t n);
int            vad_is_active(const VadGate*);
void           vad_set_threshold(VadGate*, float dbfs);
float          vad_threshold_dbfs(const VadGate*);

/* ─────────────────────── AGC ─────────────────────── */
AgcController* agc_new(float target_dbfs, float call_interval_ms);
void           agc_free(AgcController*);
void           agc_reset(AgcController*);
void           agc_update_rms(AgcController*, float rms_linear);
float          agc_tick(AgcController*);
float          agc_get_current_gain_linear(const AgcController*);
float          agc_get_current_gain_db(const AgcController*);
int            agc_is_voice_active(const AgcController*);
void           agc_set_enabled(AgcController*, bool enabled, float initial_gain_db);
bool           agc_is_enabled(const AgcController*);
void           agc_set_target(AgcController*, float dbfs);
float          agc_target_dbfs(const AgcController*);
void           agc_set_attack_ms(AgcController*, float ms);
float          agc_get_attack_ms(const AgcController*);
void           agc_set_release_ms(AgcController*, float ms);
float          agc_get_release_ms(const AgcController*);

/* ─────────────────────── Compressor ─────────────────────── */
Compressor*    compressor_new(float threshold_db, float ratio, float attack_ms,
                              float release_ms, float knee_db, float makeup_db, float fs);
void           compressor_free(Compressor*);
void           compressor_set_threshold(Compressor*, float db);
float          compressor_get_threshold(const Compressor*);
void           compressor_set_ratio(Compressor*, float r);
float          compressor_get_ratio(const Compressor*);
void           compressor_set_attack_ms(Compressor*, float ms);
float          compressor_get_attack_ms(const Compressor*);
void           compressor_set_release_ms(Compressor*, float ms);
float          compressor_get_release_ms(const Compressor*);
void           compressor_set_knee(Compressor*, float db);
float          compressor_get_knee(const Compressor*);
void           compressor_set_makeup(Compressor*, float db);
float          compressor_get_makeup(const Compressor*);
void           compressor_set_enabled(Compressor*, bool en);
bool           compressor_is_enabled(const Compressor*);
void           compressor_reset(Compressor*);
void           compressor_process(Compressor*, float* data, size_t len);

/* ─────────────────────── Denoise ─────────────────────── */
DenoiseProcessor* denoise_new(const char* model_path);
void           denoise_free(DenoiseProcessor*);
void           denoise_process_chunk(DenoiseProcessor*, const float* in, float* out);
void           denoise_process_spec_only(DenoiseProcessor*, const float* in, float* spec_out);
void           denoise_process_spec_freq(DenoiseProcessor*, const float* in, float* out);
void           denoise_reset(DenoiseProcessor*);

/* ─────────────────────── TSE ─────────────────────── */
TseProcessor*  tse_new(const char* model_path);
void           tse_free(TseProcessor*);
void           tse_set_reference(TseProcessor*, const float* data, size_t n);
bool           tse_has_reference(const TseProcessor*);
void           tse_process_chunk(TseProcessor*, const float* in, float* out);
void           tse_process_spec_freq(TseProcessor*, const float* in, float* out);
void           tse_process_from_spec(TseProcessor*, const float* spec, float* out);
void           tse_reset(TseProcessor*);
void           tse_set_debug_dump(TseProcessor*, bool enable, const char* dir);

/* ─────────────────────── AEC ─────────────────────── */
AecProcessor*  aec_new(const char* model_path);
void           aec_free(AecProcessor*);
void           aec_process_frame(AecProcessor*, const float* mic, const float* far, float* out);
void           aec_reset(AecProcessor*);

/* ─────────────────────── Resampler（libsamplerate） ─────────────────────── */
Resampler*      resampler_new(int converter_type);
void            resampler_free(Resampler*);
int             resampler_src_sinc_fastest(void);       /* SRC_SINC_FASTEST 枚举值 */
size_t          resampler_run(Resampler*, const float* in, size_t n, double src_ratio, bool end_of_input);
size_t          resampler_take(Resampler*, float* out, size_t cap);
void            resampler_reset(Resampler*);

/* ─────────────────────── AudioProcessor ─────────────────────── */
AudioProcessor* audio_processor_new(float pre_gain_db, const char* denoise_model_path,
                                     const char* tse_model_path, const char* aec_model_path);
void            audio_processor_free(AudioProcessor*);
void            audio_processor_cleanup(AudioProcessor*);
int             audio_processor_backend_effective(const AudioProcessor*);
int             audio_processor_backend_reason(const AudioProcessor*);
void            audio_processor_set_eq_gains(AudioProcessor*, const float* gains, size_t n);
void            audio_processor_get_eq_freqs(AudioProcessor*, float* freqs);
int             audio_processor_get_eq_band_count(AudioProcessor*);
size_t          audio_processor_process_eq_only(AudioProcessor*, const float* in, size_t n, float* out);
void            audio_processor_set_pre_gain(AudioProcessor*, float gain_db);
void            audio_processor_set_mode(AudioProcessor*, int mode);
int             audio_processor_get_mode(AudioProcessor*);
void            audio_processor_set_tse_enabled(AudioProcessor*, bool en);
size_t          audio_processor_get_tse_recording_audio_size(AudioProcessor*);
void            audio_processor_get_tse_recording_audio(AudioProcessor*, float* out);
void            audio_processor_set_tse_reference(AudioProcessor*, const float* data, size_t n);
bool            audio_processor_is_tse_reference_loaded(AudioProcessor*);
bool            audio_processor_is_tse_available(AudioProcessor*);
void            audio_processor_set_vad_enabled(AudioProcessor*, bool);
bool            audio_processor_is_vad_enabled(AudioProcessor*);
bool            audio_processor_is_vad_active(AudioProcessor*);
void            audio_processor_set_vad_threshold(AudioProcessor*, float);
float           audio_processor_get_vad_threshold(AudioProcessor*);
void            audio_processor_set_agc_enabled(AudioProcessor*, bool, float);
bool            audio_processor_is_agc_enabled(AudioProcessor*);
bool            audio_processor_is_agc_voice_active(AudioProcessor*);
void            audio_processor_set_recording_enabled(AudioProcessor*, bool);
bool            audio_processor_is_recording_enabled(AudioProcessor*);
float           audio_processor_get_agc_gain_db(AudioProcessor*);
void            audio_processor_set_agc_target(AudioProcessor*, float);
float           audio_processor_get_agc_target(AudioProcessor*);
void            audio_processor_set_agc_attack_ms(AudioProcessor*, float);
float           audio_processor_get_agc_attack_ms(AudioProcessor*);
void            audio_processor_set_agc_release_ms(AudioProcessor*, float);
float           audio_processor_get_agc_release_ms(AudioProcessor*);
void            audio_processor_set_noise_gate_enabled(AudioProcessor*, bool);
bool            audio_processor_is_noise_gate_enabled(AudioProcessor*);
void            audio_processor_set_noise_gate_offset_db(AudioProcessor*, float);
float           audio_processor_get_noise_gate_offset_db(AudioProcessor*);
float           audio_processor_get_noise_floor_db(AudioProcessor*);
void            audio_processor_set_compressor_enabled(AudioProcessor*, bool);
bool            audio_processor_is_compressor_enabled(AudioProcessor*);
void            audio_processor_set_compressor_threshold(AudioProcessor*, float);
float           audio_processor_get_compressor_threshold(AudioProcessor*);
void            audio_processor_set_compressor_ratio(AudioProcessor*, float);
float           audio_processor_get_compressor_ratio(AudioProcessor*);
void            audio_processor_set_compressor_attack(AudioProcessor*, float);
float           audio_processor_get_compressor_attack(AudioProcessor*);
void            audio_processor_set_compressor_release(AudioProcessor*, float);
float           audio_processor_get_compressor_release(AudioProcessor*);
void            audio_processor_set_compressor_makeup(AudioProcessor*, float);
float           audio_processor_get_compressor_makeup(AudioProcessor*);
void            audio_processor_set_compressor_knee(AudioProcessor*, float);
float           audio_processor_get_compressor_knee(AudioProcessor*);
void            audio_processor_set_aec_enabled(AudioProcessor*, bool);
bool            audio_processor_is_aec_available(AudioProcessor*);
void            audio_processor_set_aec_far_sample_rate(AudioProcessor*, int);
int             audio_processor_get_aec_far_sample_rate(AudioProcessor*);
void            audio_processor_set_aec_far_rms_target(AudioProcessor*, float);
float           audio_processor_get_aec_far_rms_target(AudioProcessor*);
/* process: 输入 1024 采样 → 输出 out（容量 ≥1024）。far/far_n 可空用于 AEC。返回输出采样数。 */
size_t          audio_processor_process(AudioProcessor*, const float* in, size_t n,
                                        const float* far, size_t far_n, float* out);
void            audio_processor_set_io_sample_rates(AudioProcessor*, int in_sr, int out_sr);
/* pipeline: 喂入任意长输入，输出累积到内部缓冲。返回累计可读采样数。 */
size_t          audio_processor_process_pipeline(AudioProcessor*, const float* in, size_t n,
                                                 const float* far, size_t far_n);
size_t          audio_processor_pipeline_take(AudioProcessor*, float* out, size_t cap);
size_t          audio_processor_viz_input_take(AudioProcessor*, float* out, size_t cap);
size_t          audio_processor_viz_output_take(AudioProcessor*, float* out, size_t cap);

/* ─────────────────────── 频谱 ─────────────────────── */
size_t          compute_spectrum(const float* samples, size_t n, float* out);
void            spectrum_warmup(void);

/* ─────────────────────── RingBuffer（线程安全 FIFO）─────────────────────── */
RingBuffer*     ringbuffer_new(size_t capacity);
void            ringbuffer_free(RingBuffer*);
void            ringbuffer_write(RingBuffer*, const float* data, size_t n);
size_t          ringbuffer_read(RingBuffer*, float* dest, size_t n);
size_t          ringbuffer_available(const RingBuffer*);
void            ringbuffer_clear(RingBuffer*);

#ifdef __cplusplus
}
#endif

#endif /* AIMIC_H */