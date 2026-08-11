# PureVox — AI 麦克风降噪工具
# Copyright (C) 2024-2026 a2heng <752848283@qq.com>
#
# PureVox is licensed under the GNU General Public License v3.0 or
# later (GPL-3.0-or-later).  See LICENSE for details.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The built-in AI models are NOT covered by the GPL; they are the
# property of a2heng and may only be used with PureVox under
# authorization.  See MODEL-LICENSE.md for details.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# aimic — 纯 C 音频引擎（aimic.c → libaimic.so）的 ctypes 绑定。
#
# 取代旧 aimic_bind.cpp（pybind11）：不再有任何 C++，Python 只管类型搬运，
# 全部 DSP 留在 C 侧。类名/方法名与原 pybind11 绑定完全一致：
#   AudioProcessor / TseProcessor / AecProcessor / Resampler / RingBuffer
#   + compute_spectrum / spectrum_warmup / SRC_SINC_FASTEST / SPECTRUM_NUM_BANDS
#
# 加载策略：Linux 下先按发行包路径预加载捆绑的 libonnxruntime.so（1.11.1），
# 再加载 libaimic.so（同目录优先，其次 LD_LIBRARY_PATH）。

import ctypes as _ct
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
IS_LINUX = sys.platform.startswith("linux")
IS_WIN = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"

_HOP_LENGTH = 1024
_SPECTRUM_BANDS = 128
_EQ_BANDS = 61

_MODE_PASSTHROUGH = 0
_MODE_DENOISE = 1
_MODE_AEC = 2
_MODE_TSE = 3

_BACKEND_AVX = 0
_BACKEND_SSE = 1
_BACKEND_NPU = 2


def _preload_onnxruntime():
    """预加载捆绑的 onnxruntime（满足 libaimic.so 的 DT_NEEDED）。

    返回预加载的库路径；找不到时返回 ""（此时仍需 LD_LIBRARY_PATH 兜底）。
    """
    if IS_LINUX:
        prefix, pattern = "libonnxruntime", "libonnxruntime.so"
    elif IS_WIN:
        prefix, pattern = "onnxruntime", "onnxruntime.dll"
    else:
        prefix, pattern = "libonnxruntime", "libonnxruntime.dylib"
    if IS_WIN:
        dirs = (os.path.join(_HERE, "packages", "onnxruntime-win-x64-1.11.1", "lib"), _HERE)
    elif IS_LINUX:
        dirs = (os.path.join(_HERE, "packages", "onnxruntime-linux-x64-1.11.1", "lib"), _HERE)
    else:
        dirs = (_HERE,)
    for d in dirs:
        try:
            entries = [f for f in os.listdir(d) if f.startswith(pattern)]
        except OSError:
            entries = []
        if not entries:
            continue
        entries.sort()
        for name in entries:
            path = os.path.join(d, name)
            try:
                if hasattr(_ct, "RTLD_GLOBAL"):
                    mode = getattr(_ct, "RTLD_GLOBAL", 0) | getattr(_ct, "RTLD_NOW", 0) or _ct.DEFAULT_MODE
                else:
                    mode = _ct.DEFAULT_MODE
                _ct.CDLL(path, mode=mode)
                return path
            except OSError:
                continue
    return ""


def _load_lib():
    _preload_onnxruntime()
    _lib = None
    if IS_LINUX:
        names = ("libaimic.so",)
    elif IS_WIN:
        names = ("aimic.dll",)
    elif IS_MACOS:
        names = ("libaimic.dylib",)
    else:
        names = ()
    errors = []
    for name in names:
        for base in (_HERE, ""):
            path = os.path.join(base, name) if base else name
            try:
                _lib = _ct.CDLL(path)
                break
            except OSError as e:
                errors.append("%s: %s" % (path, e))
        if _lib is not None:
            break
    if _lib is None:
        raise ImportError("无法加载 aimic 共享库请先 python setup.py build_ext --inplace\n  %s"
                          % "; ".join(errors))
    return _lib


_lib = _load_lib()

# ── C API 原型 ─────────────────────────────────────────────────────────
_c_void_p = _ct.c_void_p
_c_float = _ct.c_float
_c_size = _ct.c_size_t
_c_int = _ct.c_int
_c_bool = _ct.c_bool
_c_char_p = _ct.c_char_p
_c_char = _ct.c_char
_c_float_p = _ct.POINTER(_ct.c_float)
_c_void_p_p = _ct.c_void_p


def _fn(name, restype, *argtypes):
    f = getattr(_lib, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


_rb_new = _fn("ringbuffer_new", _c_void_p, _c_size)
_rb_free = _fn("ringbuffer_free", None, _c_void_p)
_rb_write = _fn("ringbuffer_write", None, _c_void_p, _c_float_p, _c_size)
_rb_read = _fn("ringbuffer_read", _c_size, _c_void_p, _c_float_p, _c_size)
_rb_available = _fn("ringbuffer_available", _c_size, _c_void_p)
_rb_clear = _fn("ringbuffer_clear", None, _c_void_p)

_resampler_src_fast = _fn("resampler_src_sinc_fastest", _c_int)
_resampler_new = _fn("resampler_new", _c_void_p, _c_int)
_resampler_free = _fn("resampler_free", None, _c_void_p)
_resampler_run = _fn("resampler_run", _c_size, _c_void_p, _c_float_p, _c_size,
                     _ct.c_double, _c_bool)
_resampler_take = _fn("resampler_take", _c_size, _c_void_p, _c_float_p, _c_size)
_resampler_reset = _fn("resampler_reset", None, _c_void_p)

_aec_new = _fn("aec_new", _c_void_p, _c_char_p)
_aec_free = _fn("aec_free", None, _c_void_p)
_aec_process_frame = _fn("aec_process_frame", None, _c_void_p, _c_float_p, _c_float_p, _c_float_p)
_aec_reset = _fn("aec_reset", None, _c_void_p)

_tse_new = _fn("tse_new", _c_void_p, _c_char_p)
_tse_free = _fn("tse_free", None, _c_void_p)
_tse_set_reference = _fn("tse_set_reference", None, _c_void_p, _c_float_p, _c_size)
_tse_has_reference = _fn("tse_has_reference", _c_bool, _c_void_p)
_tse_process_chunk = _fn("tse_process_chunk", None, _c_void_p, _c_float_p, _c_float_p)
_tse_reset = _fn("tse_reset", None, _c_void_p)
_tse_set_debug_dump = _fn("tse_set_debug_dump", None, _c_void_p, _c_bool, _c_char_p)

_ap_new = _fn("audio_processor_new", _c_void_p, _c_float, _c_char_p, _c_char_p, _c_char_p)
_ap_free = _fn("audio_processor_free", None, _c_void_p)
_ap_cleanup = _fn("audio_processor_cleanup", None, _c_void_p)
_ap_backend_effective = _fn("audio_processor_backend_effective", _c_int, _c_void_p)
_ap_backend_reason = _fn("audio_processor_backend_reason", _c_int, _c_void_p)
_ap_set_eq_gains = _fn("audio_processor_set_eq_gains", None, _c_void_p, _c_float_p, _c_size)
_ap_get_eq_freqs = _fn("audio_processor_get_eq_freqs", None, _c_void_p, _c_float_p)
_ap_get_eq_band_count = _fn("audio_processor_get_eq_band_count", _c_int, _c_void_p)
_ap_process_eq_only = _fn("audio_processor_process_eq_only", _c_size, _c_void_p, _c_float_p, _c_size, _c_float_p)
_ap_set_pre_gain = _fn("audio_processor_set_pre_gain", None, _c_void_p, _c_float)
_ap_set_mode = _fn("audio_processor_set_mode", None, _c_void_p, _c_int)
_ap_get_mode = _fn("audio_processor_get_mode", _c_int, _c_void_p)
_ap_set_tse_enabled = _fn("audio_processor_set_tse_enabled", None, _c_void_p, _c_bool)
_ap_get_tse_rec_audio_size = _fn("audio_processor_get_tse_recording_audio_size", _c_size, _c_void_p)
_ap_get_tse_rec_audio = _fn("audio_processor_get_tse_recording_audio", None, _c_void_p, _c_float_p)
_ap_set_tse_reference = _fn("audio_processor_set_tse_reference", None, _c_void_p, _c_float_p, _c_size)
_ap_is_tse_ref_loaded = _fn("audio_processor_is_tse_reference_loaded", _c_bool, _c_void_p)
_ap_is_tse_available = _fn("audio_processor_is_tse_available", _c_bool, _c_void_p)
_ap_set_vad_enabled = _fn("audio_processor_set_vad_enabled", None, _c_void_p, _c_bool)
_ap_is_vad_enabled = _fn("audio_processor_is_vad_enabled", _c_bool, _c_void_p)
_ap_is_vad_active = _fn("audio_processor_is_vad_active", _c_bool, _c_void_p)
_ap_set_vad_threshold = _fn("audio_processor_set_vad_threshold", None, _c_void_p, _c_float)
_ap_get_vad_threshold = _fn("audio_processor_get_vad_threshold", _c_float, _c_void_p)
_ap_set_agc_enabled = _fn("audio_processor_set_agc_enabled", None, _c_void_p, _c_bool, _c_float)
_ap_is_agc_enabled = _fn("audio_processor_is_agc_enabled", _c_bool, _c_void_p)
_ap_is_agc_voice_active = _fn("audio_processor_is_agc_voice_active", _c_bool, _c_void_p)
_ap_set_recording_enabled = _fn("audio_processor_set_recording_enabled", None, _c_void_p, _c_bool)
_ap_is_recording_enabled = _fn("audio_processor_is_recording_enabled", _c_bool, _c_void_p)
_ap_get_agc_gain_db = _fn("audio_processor_get_agc_gain_db", _c_float, _c_void_p)
_ap_set_agc_target = _fn("audio_processor_set_agc_target", None, _c_void_p, _c_float)
_ap_get_agc_target = _fn("audio_processor_get_agc_target", _c_float, _c_void_p)
_ap_set_agc_attack_ms = _fn("audio_processor_set_agc_attack_ms", None, _c_void_p, _c_float)
_ap_get_agc_attack_ms = _fn("audio_processor_get_agc_attack_ms", _c_float, _c_void_p)
_ap_set_agc_release_ms = _fn("audio_processor_set_agc_release_ms", None, _c_void_p, _c_float)
_ap_get_agc_release_ms = _fn("audio_processor_get_agc_release_ms", _c_float, _c_void_p)
_ap_set_noise_gate_enabled = _fn("audio_processor_set_noise_gate_enabled", None, _c_void_p, _c_bool)
_ap_is_noise_gate_enabled = _fn("audio_processor_is_noise_gate_enabled", _c_bool, _c_void_p)
_ap_set_noise_gate_offset_db = _fn("audio_processor_set_noise_gate_offset_db", None, _c_void_p, _c_float)
_ap_get_noise_gate_offset_db = _fn("audio_processor_get_noise_gate_offset_db", _c_float, _c_void_p)
_ap_get_noise_floor_db = _fn("audio_processor_get_noise_floor_db", _c_float, _c_void_p)
_ap_set_agc_attack_ms = _fn("audio_processor_set_agc_attack_ms", None, _c_void_p, _c_float)
_ap_get_agc_attack_ms = _fn("audio_processor_get_agc_attack_ms", _c_float, _c_void_p)
_ap_set_agc_release_ms = _fn("audio_processor_set_agc_release_ms", None, _c_void_p, _c_float)
_ap_get_agc_release_ms = _fn("audio_processor_get_agc_release_ms", _c_float, _c_void_p)
_ap_set_noise_gate_enabled = _fn("audio_processor_set_noise_gate_enabled", None, _c_void_p, _c_bool)
_ap_is_noise_gate_enabled = _fn("audio_processor_is_noise_gate_enabled", _c_bool, _c_void_p)
_ap_set_noise_gate_offset_db = _fn("audio_processor_set_noise_gate_offset_db", None, _c_void_p, _c_float)
_ap_get_noise_gate_offset_db = _fn("audio_processor_get_noise_gate_offset_db", _c_float, _c_void_p)
_ap_get_noise_floor_db = _fn("audio_processor_get_noise_floor_db", _c_float, _c_void_p)
_ap_set_comp_enabled = _fn("audio_processor_set_compressor_enabled", None, _c_void_p, _c_bool)
_ap_is_comp_enabled = _fn("audio_processor_is_compressor_enabled", _c_bool, _c_void_p)
_ap_set_comp_threshold = _fn("audio_processor_set_compressor_threshold", None, _c_void_p, _c_float)
_ap_get_comp_threshold = _fn("audio_processor_get_compressor_threshold", _c_float, _c_void_p)
_ap_set_comp_ratio = _fn("audio_processor_set_compressor_ratio", None, _c_void_p, _c_float)
_ap_get_comp_ratio = _fn("audio_processor_get_compressor_ratio", _c_float, _c_void_p)
_ap_set_comp_attack = _fn("audio_processor_set_compressor_attack", None, _c_void_p, _c_float)
_ap_get_comp_attack = _fn("audio_processor_get_compressor_attack", _c_float, _c_void_p)
_ap_set_comp_release = _fn("audio_processor_set_compressor_release", None, _c_void_p, _c_float)
_ap_get_comp_release = _fn("audio_processor_get_compressor_release", _c_float, _c_void_p)
_ap_set_comp_makeup = _fn("audio_processor_set_compressor_makeup", None, _c_void_p, _c_float)
_ap_get_comp_makeup = _fn("audio_processor_get_compressor_makeup", _c_float, _c_void_p)
_ap_set_comp_knee = _fn("audio_processor_set_compressor_knee", None, _c_void_p, _c_float)
_ap_get_comp_knee = _fn("audio_processor_get_compressor_knee", _c_float, _c_void_p)
_ap_set_aec_enabled = _fn("audio_processor_set_aec_enabled", None, _c_void_p, _c_bool)
_ap_is_aec_available = _fn("audio_processor_is_aec_available", _c_bool, _c_void_p)
_ap_set_aec_far_sr = _fn("audio_processor_set_aec_far_sample_rate", None, _c_void_p, _c_int)
_ap_get_aec_far_sr = _fn("audio_processor_get_aec_far_sample_rate", _c_int, _c_void_p)
_ap_set_aec_far_rms = _fn("audio_processor_set_aec_far_rms_target", None, _c_void_p, _c_float)
_ap_get_aec_far_rms = _fn("audio_processor_get_aec_far_rms_target", _c_float, _c_void_p)
_ap_process = _fn("audio_processor_process", _c_size, _c_void_p, _c_float_p, _c_size,
                  _c_float_p, _c_size, _c_float_p)
_ap_set_io_sr = _fn("audio_processor_set_io_sample_rates", None, _c_void_p, _c_int, _c_int)
_ap_process_pipeline = _fn("audio_processor_process_pipeline", _c_size, _c_void_p,
                           _c_float_p, _c_size, _c_float_p, _c_size)
_ap_pipeline_take = _fn("audio_processor_pipeline_take", _c_size, _c_void_p, _c_float_p, _c_size)
_ap_viz_input_take = _fn("audio_processor_viz_input_take", _c_size, _c_void_p, _c_float_p, _c_size)
_ap_viz_output_take = _fn("audio_processor_viz_output_take", _c_size, _c_void_p, _c_float_p, _c_size)

_compute_spectrum = _fn("compute_spectrum", _c_size, _c_float_p, _c_size, _c_float_p)
_spectrum_warmup = _fn("spectrum_warmup", None)

# ── 模块级常量与函数 ────────────────────────────────────────────────────

SPECTRUM_NUM_BANDS = _SPECTRUM_BANDS
HOP_LENGTH = _HOP_LENGTH
SRC_SINC_FASTEST = _resampler_src_fast()
MODE_PASSTHROUGH = _MODE_PASSTHROUGH
MODE_DENOISE = _MODE_DENOISE
MODE_AEC = _MODE_AEC
MODE_TSE = _MODE_TSE

BACKEND_AVX = _BACKEND_AVX
BACKEND_SSE = _BACKEND_SSE
BACKEND_NPU = _BACKEND_NPU

BACKEND_REASON_OK = 0
BACKEND_REASON_NPU_UNAVAILABLE = 1
BACKEND_REASON_NPU_NO_ENTRY = 2


def compute_spectrum(samples):
    """计算 128 频段的 Mel 频谱（返回 dB 值 list）。"""
    if not samples:
        return []
    n = len(samples)
    arr = (_c_float * n)(*samples)
    out = (_c_float * _SPECTRUM_BANDS)()
    got = _compute_spectrum(arr, n, out)
    return list(out[:got])


def spectrum_warmup():
    """预初始化频谱 FFT 与 Mel filterbank（启动时调用一次）。"""
    _spectrum_warmup()


# ══════════════════════════════════════════════════════════════════════
#  RingBuffer
# ══════════════════════════════════════════════════════════════════════

class RingBuffer:
    """纯 C 线程安全 FIFO。满时丢弃最旧数据。"""

    def __init__(self, capacity):
        self._p = _rb_new(int(capacity))
        self._free = _rb_free
        if not self._p:
            raise RuntimeError("aimic: RingBuffer alloc failed")

    def __del__(self):
        try:
            if getattr(self, "_p", None):
                self._free(self._p)
                self._p = None
        except Exception:
            pass

    def write(self, data):
        if not data:
            return
        n = len(data)
        arr = (_c_float * n)(*data)
        _rb_write(self._p, arr, n)

    def read(self, n):
        n = int(n)
        arr = (_c_float * max(n, 1))()
        got = _rb_read(self._p, arr, n)
        return list(arr[:got])

    def available(self):
        return int(_rb_available(self._p))

    def clear(self):
        _rb_clear(self._p)


# ══════════════════════════════════════════════════════════════════════
#  Resampler
# ══════════════════════════════════════════════════════════════════════

class Resampler:
    """libsamplerate 重采样器（默认 SRC_SINC_FASTEST）。"""

    def __init__(self, converter_type=None):
        if converter_type is None:
            converter_type = SRC_SINC_FASTEST
        self._p = _resampler_new(int(converter_type))
        self._free = _resampler_free
        if not self._p:
            raise RuntimeError("aimic: Resampler alloc failed")

    def __del__(self):
        try:
            if getattr(self, "_p", None):
                self._free(self._p)
                self._p = None
        except Exception:
            pass

    def process(self, input, src_ratio, end_of_input=False):
        """src_ratio = 目标采样率 / 输入采样率。返回重采样后样本列表。"""
        if not input and not end_of_input:
            return []
        if src_ratio <= 0.0:
            raise ValueError("Resampler: src_ratio must be positive")
        in_arr = None
        if input:
            in_arr = (_c_float * len(input))(*input)
        n = _resampler_run(self._p, in_arr, len(input), float(src_ratio), bool(end_of_input))
        if n == 0:
            return []
        arr = (_c_float * n)()
        _resampler_take(self._p, arr, n)
        return list(arr)

    def reset(self):
        _resampler_reset(self._p)


# ══════════════════════════════════════════════════════════════════════
#  AecProcessor
# ══════════════════════════════════════════════════════════════════════

class AecProcessor:
    """流式 AEC（1024 采样/块，48kHz，2048 NFFT）。"""

    def __init__(self, model_path):
        self._p = _aec_new(model_path.encode("utf-8"))
        self._free = _aec_free
        if not self._p:
            raise RuntimeError("aimic: AEC processor alloc failed")

    def __del__(self):
        try:
            if getattr(self, "_p", None):
                self._free(self._p)
                self._p = None
        except Exception:
            pass

    def process_frame(self, mic, far):
        if len(mic) != _HOP_LENGTH:
            raise ValueError(
                "Input audio chunk length must be equal to hop length (1024)")
        m = (_c_float * len(mic))(*mic)
        f = (_c_float * len(far))(*far) if far else None
        out = (_c_float * _HOP_LENGTH)()
        _aec_process_frame(self._p, m, f, out)
        return list(out)

    def reset(self):
        _aec_reset(self._p)


# ══════════════════════════════════════════════════════════════════════
#  TseProcessor
# ══════════════════════════════════════════════════════════════════════

class TseProcessor:
    """流式 TSE（目标说话人提取）。2048 FFT / 1024 HOP。"""

    def __init__(self, model_path):
        self._p = _tse_new(model_path.encode("utf-8"))
        self._free = _tse_free
        if not self._p:
            raise RuntimeError("aimic: TseProcessor alloc failed")

    def __del__(self):
        try:
            if getattr(self, "_p", None):
                self._free(self._p)
                self._p = None
        except Exception:
            pass

    def set_reference(self, ref):
        if not ref:
            _tse_set_reference(self._p, None, 0)
            return
        arr = (_c_float * len(ref))(*ref)
        _tse_set_reference(self._p, arr, len(ref))

    def has_reference(self):
        return bool(_tse_has_reference(self._p))

    def process_chunk(self, in_samples):
        if len(in_samples) != _HOP_LENGTH:
            raise RuntimeError(
                "Input audio chunk length must be equal to hop length (1024)")
        arr = (_c_float * len(in_samples))(*in_samples)
        out = (_c_float * _HOP_LENGTH)()
        _tse_process_chunk(self._p, arr, out)
        return list(out)

    def reset(self):
        _tse_reset(self._p)

    def set_debug_dump(self, enable, dir_path):
        _tse_set_debug_dump(self._p, bool(enable), dir_path.encode("utf-8"))


# ══════════════════════════════════════════════════════════════════════
#  AudioProcessor
# ══════════════════════════════════════════════════════════════════════

class AudioProcessor:
    """全链路音频处理器（EQ/VAD/AGC/压缩器/降噪/AEC/TSE）。"""

    def __init__(self, pre_gain_db, denoise_model_path, tse_model_path="", aec_model_path=""):
        self._p = _ap_new(float(pre_gain_db),
                          denoise_model_path.encode("utf-8"),
                          tse_model_path.encode("utf-8"),
                          aec_model_path.encode("utf-8"))
        self._free = _ap_free
        if not self._p:
            raise RuntimeError("aimic: AudioProcessor alloc failed")

    def __del__(self):
        try:
            if getattr(self, "_p", None):
                self._free(self._p)
                self._p = None
        except Exception:
            pass

    # ── 推理后端（自动选择：NPU → AVX/SSE；查询实际生效情况）──
    def backend_effective(self):
        """返回实际生效的推理后端（aimic.BACKEND_*）。"""
        return int(_ap_backend_effective(self._p))

    def backend_reason(self):
        """返回 NPU 未生效的原因码（aimic.BACKEND_REASON_*；0 表示 NPU 已生效）。"""
        return int(_ap_backend_reason(self._p))

    def backend_info(self):
        """返回 (实际生效后端, NPU 未生效原因码)。"""
        return (self.backend_effective(), self.backend_reason())

    # ── 基础控制 ──
    def cleanup(self):
        _ap_cleanup(self._p)

    def set_pre_gain(self, db):
        _ap_set_pre_gain(self._p, float(db))

    def set_mode(self, mode):
        _ap_set_mode(self._p, int(mode))

    def get_mode(self):
        return int(_ap_get_mode(self._p))

    def set_io_sample_rates(self, in_sr, out_sr):
        _ap_set_io_sr(self._p, int(in_sr), int(out_sr))

    # ── EQ ──
    def set_eq_gains(self, gains):
        if not gains:
            return
        arr = (_c_float * len(gains))(*gains)
        _ap_set_eq_gains(self._p, arr, len(gains))

    def get_eq_freqs(self):
        arr = (_c_float * _EQ_BANDS)()
        _ap_get_eq_freqs(self._p, arr)
        return list(arr)

    def get_eq_band_count(self):
        return int(_ap_get_eq_band_count(self._p))

    def process_eq_only(self, in_samples):
        out = (_c_float * len(in_samples))()
        if in_samples:
            arr = (_c_float * len(in_samples))(*in_samples)
            _ap_process_eq_only(self._p, arr, len(in_samples), out)
        return list(out)

    # ── 降噪主链路 ──
    def process(self, mic):
        """输入 1024 采样 → 输出（最多 1024）。"""
        if len(mic) != _HOP_LENGTH:
            raise RuntimeError(
                "Input audio chunk length must be equal to hop length (1024)")
        m = (_c_float * len(mic))(*mic)
        out = (_c_float * _HOP_LENGTH)()
        got = _ap_process(self._p, m, len(mic), None, 0, out)
        return list(out[:got])

    def process_with_far(self, mic, far_end):
        """带 far-end（AEC）的处理。"""
        if len(mic) != _HOP_LENGTH:
            raise RuntimeError(
                "Input audio chunk length must be equal to hop length (1024)")
        m = (_c_float * len(mic))(*mic)
        f = (_c_float * len(far_end))(*far_end) if far_end else None
        out = (_c_float * _HOP_LENGTH)()
        got = _ap_process(self._p, m, len(mic), f, len(far_end) if far_end else 0, out)
        return list(out[:got])

    def set_aec_enabled(self, enabled):
        _ap_set_aec_enabled(self._p, bool(enabled))

    def is_aec_available(self):
        return bool(_ap_is_aec_available(self._p))

    def set_aec_far_sample_rate(self, sr):
        _ap_set_aec_far_sr(self._p, int(sr))

    def get_aec_far_sample_rate(self):
        return int(_ap_get_aec_far_sr(self._p))

    def set_aec_far_rms_target(self, v):
        _ap_set_aec_far_rms(self._p, float(v))

    def get_aec_far_rms_target(self):
        return float(_ap_get_aec_far_rms(self._p))

    # ── 流式 pipeline ──
    def process_pipeline(self, raw_input, far_end=None):
        if not raw_input:
            return []
        m = (_c_float * len(raw_input))(*raw_input)
        f = (_c_float * len(far_end))(*far_end) if far_end else None
        n = _ap_process_pipeline(self._p, m, len(raw_input), f,
                                 len(far_end) if far_end else 0)
        if n == 0:
            return []
        out = (_c_float * n)()
        _ap_pipeline_take(self._p, out, n)
        return list(out)

    # ── 频谱可视化 ──
    def get_and_clear_viz_input(self):
        arr = (_c_float * (1 << 16))()
        n = _ap_viz_input_take(self._p, arr, 1 << 16)
        return list(arr[:n])

    def get_and_clear_viz_output(self):
        arr = (_c_float * (1 << 16))()
        n = _ap_viz_output_take(self._p, arr, 1 << 16)
        return list(arr[:n])

    # ── TSE ──
    def set_tse_enabled(self, enabled):
        _ap_set_tse_enabled(self._p, bool(enabled))

    def get_tse_recording_audio(self):
        n = _ap_get_tse_rec_audio_size(self._p)
        if n == 0:
            return []
        arr = (_c_float * n)()
        _ap_get_tse_rec_audio(self._p, arr)
        return list(arr)

    def set_tse_reference(self, ref):
        if not ref:
            _ap_set_tse_reference(self._p, None, 0)
            return
        arr = (_c_float * len(ref))(*ref)
        _ap_set_tse_reference(self._p, arr, len(ref))

    def is_tse_reference_loaded(self):
        return bool(_ap_is_tse_ref_loaded(self._p))

    def is_tse_available(self):
        return bool(_ap_is_tse_available(self._p))

    # ── VAD ──
    def set_vad_enabled(self, enabled):
        _ap_set_vad_enabled(self._p, bool(enabled))

    def is_vad_enabled(self):
        return bool(_ap_is_vad_enabled(self._p))

    def is_vad_active(self):
        return bool(_ap_is_vad_active(self._p))

    def set_vad_threshold(self, db):
        _ap_set_vad_threshold(self._p, float(db))

    def get_vad_threshold(self):
        return float(_ap_get_vad_threshold(self._p))

    # ── AGC ──
    def set_agc_enabled(self, enabled, initial_gain_db=0.0):
        _ap_set_agc_enabled(self._p, bool(enabled), float(initial_gain_db))

    def is_agc_enabled(self):
        return bool(_ap_is_agc_enabled(self._p))

    def is_agc_voice_active(self):
        return bool(_ap_is_agc_voice_active(self._p))

    def set_recording_enabled(self, enabled):
        _ap_set_recording_enabled(self._p, bool(enabled))

    def is_recording_enabled(self):
        return bool(_ap_is_recording_enabled(self._p))

    def get_agc_gain_db(self):
        return float(_ap_get_agc_gain_db(self._p))

    def set_agc_target(self, db):
        _ap_set_agc_target(self._p, float(db))

    def get_agc_target(self):
        return float(_ap_get_agc_target(self._p))

    # -- AGC time parameters --
    def set_agc_attack_ms(self, ms):
        _ap_set_agc_attack_ms(self._p, float(ms))

    def get_agc_attack_ms(self):
        return float(_ap_get_agc_attack_ms(self._p))

    def set_agc_release_ms(self, ms):
        _ap_set_agc_release_ms(self._p, float(ms))

    def get_agc_release_ms(self):
        return float(_ap_get_agc_release_ms(self._p))

    # -- Noise gate --
    def set_noise_gate_enabled(self, en):
        _ap_set_noise_gate_enabled(self._p, bool(en))

    def is_noise_gate_enabled(self):
        return bool(_ap_is_noise_gate_enabled(self._p))

    def set_noise_gate_offset_db(self, db):
        _ap_set_noise_gate_offset_db(self._p, float(db))

    def get_noise_gate_offset_db(self):
        return float(_ap_get_noise_gate_offset_db(self._p))

    def get_noise_floor_db(self):
        return float(_ap_get_noise_floor_db(self._p))

    # ── AGC time parameters ──
    def set_agc_attack_ms(self, ms):
        _ap_set_agc_attack_ms(self._p, float(ms))

    def get_agc_attack_ms(self):
        return float(_ap_get_agc_attack_ms(self._p))

    def set_agc_release_ms(self, ms):
        _ap_set_agc_release_ms(self._p, float(ms))

    def get_agc_release_ms(self):
        return float(_ap_get_agc_release_ms(self._p))

    # ── Noise gate ──
    def set_noise_gate_enabled(self, en):
        _ap_set_noise_gate_enabled(self._p, bool(en))

    def is_noise_gate_enabled(self):
        return bool(_ap_is_noise_gate_enabled(self._p))

    def set_noise_gate_offset_db(self, db):
        _ap_set_noise_gate_offset_db(self._p, float(db))

    def get_noise_gate_offset_db(self):
        return float(_ap_get_noise_gate_offset_db(self._p))

    def get_noise_floor_db(self):
        return float(_ap_get_noise_floor_db(self._p))

    # ── Compressor ──
    def set_compressor_enabled(self, enabled):
        _ap_set_comp_enabled(self._p, bool(enabled))

    def is_compressor_enabled(self):
        return bool(_ap_is_comp_enabled(self._p))

    def set_compressor_threshold(self, db):
        _ap_set_comp_threshold(self._p, float(db))

    def get_compressor_threshold(self):
        return float(_ap_get_comp_threshold(self._p))

    def set_compressor_ratio(self, r):
        _ap_set_comp_ratio(self._p, float(r))

    def get_compressor_ratio(self):
        return float(_ap_get_comp_ratio(self._p))

    def set_compressor_attack(self, ms):
        _ap_set_comp_attack(self._p, float(ms))

    def get_compressor_attack(self):
        return float(_ap_get_comp_attack(self._p))

    def set_compressor_release(self, ms):
        _ap_set_comp_release(self._p, float(ms))

    def get_compressor_release(self):
        return float(_ap_get_comp_release(self._p))

    def set_compressor_makeup(self, db):
        _ap_set_comp_makeup(self._p, float(db))

    def get_compressor_makeup(self):
        return float(_ap_get_comp_makeup(self._p))

    def set_compressor_knee(self, db):
        _ap_set_comp_knee(self._p, float(db))

    def get_compressor_knee(self):
        return float(_ap_get_comp_knee(self._p))


spectrum_warmup()