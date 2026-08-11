# PureVox

实时 AI 音频降噪工具 —— 降噪 / 目标说话人提取 / 回声消除，支持本地麦克风与远程网络推流。

## 概述

PureVox 是一款低延迟 AI 音频处理桌面工具，通过本地麦克风或远程推流（手机/浏览器）输入音频，经过 AI 模型实时处理后输出到扬声器或虚拟麦克风。

## 功能

- **AI 降噪** — 48kHz 实时降噪，消除键盘、风扇、空调等环境噪音
- **TSE 目标提取** — 录制参考语音后，从混合音频中分离目标人声
- **AEC 回声消除** — 消除扬声器回声
- **31 段均衡器** — 8 个插槽 + 内置预设
- **AGC / VAD** — 自动增益控制 / 语音活动检测
- **远程麦克风** — 手机浏览器或 Android APK 通过局域网 WSS 推流到 PC 处理
- **双平台** — Windows (WASAPI) 和 Linux (原生 PipeWire)

## 远程推流

手机/浏览器 → WSS(Opus) → PC 服务器 → AI 处理链路 → 扬声器/虚拟麦克风

```
手机 → https://<PC的IP>:59123（mDNS 广播 _purevox._tcp.local.）→ 降噪 → 输出
```

- 浏览器：手机与 PC 同局域网，访问 `https://<PC的IP>:59123`，信任自签名证书后点击麦克风推流
- APK：打开自动搜索局域网服务器，发现即自动连接推流

## 项目结构

```
run_pyside6.py         # 启动入口（单实例锁）
ui_pyside6.py          # 主 UI（PySide6）
audio_processor.py     # 核心音频引擎
aimic.c + aimic.py     # C 音频核心 → aimic.dll / libaimic.so
config_manager.py      # JSON 配置管理
dialog_eq.py           # 均衡器对话框
dialog_about.py        # 关于对话框
server/                # 远程麦克风 HTTPS/WSS 服务器
html/                  # 浏览器推流前端（AudioWorklet + Opus WASM）
android/               # Android 客户端（Kotlin）
```

## 技术栈

| 组件 | 技术 |
|---|---|
| 桌面 GUI | Python + PySide6 |
| 音频处理 | 纯 C 共享库 + ONNX Runtime C API |
| Linux 音频 | 原生 PipeWire |
| Windows 音频 | WASAPI 全双工 |
| 服务器 | Python aiohttp + zeroconf + cryptography |
| 音频编码 | Opus |
| Android | Kotlin + OkHttp + NsdManager |

## 内置模型禁止商用，谢谢理解
