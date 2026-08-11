# PureVox

实时 AI 音频降噪工具 —— 降噪 / 目标说话人提取 / 回声消除，支持本地麦克风与远程网络推流。

[English](README_EN.md)

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

## 快速开始

### 下载发行版（推荐）

从 [Releases](https://github.com/Simon404Error/PureVox/releases) 下载最新版，解压即用，无需安装 Python。

Windows 用户还需安装 [VB-CABLE](https://vb-audio.com/Cable/) 虚拟声卡（首次运行 PureVox 会弹出安装指引）。

### 从源码运行

```bash
# 克隆仓库
git clone https://github.com/Simon404Error/PureVox.git
cd PureVox

# 安装依赖
pip install -r requirements.txt
# Windows 需追加
pip install -r requirements-win.txt

# 编译 C 扩展（gcc）
python setup.py build_ext --inplace --force

# 运行
python run_pyside6.py
```

> 也可使用项目自带的嵌入式 Python 3.8（独立于系统环境）：
> - Windows: `powershell -File bootstrap_python38.ps1`
> - Linux: `./bootstrap_python38.sh`

## 打包

```powershell
# Windows（生成 dist/PureVox/ 目录）
powershell -ExecutionPolicy Bypass -File build_win.ps1
```

```bash
# Linux（生成 deb / rpm / AppImage）
bash pack_deb.sh      # Debian/Ubuntu
bash pack_rpm.sh      # Fedora/RHEL
bash pack_appimage.sh # 通用 AppImage
```

```bash
# Android APK
cd android && ./gradlew assembleDebug
```

CI 会在推送 tag `v<yyyy.MM.dd.HHmm>` 时自动构建并发布 Release。

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

## 许可

- **源代码**：[GPL-3.0](LICENSE)
- **内置 AI 模型**：不属 GPL 授权，归作者 a2heng 所有，仅可在 PureVox 内经授权使用 — 详见 [MODEL-LICENSE.md](MODEL-LICENSE.md)
- **第三方组件**（PySide6、ONNX Runtime、Opus 等）使用各自许可，详见 [LICENSE-THIRD-PARTY.txt](LICENSE-THIRD-PARTY.txt)

作者的开源模型仓库（较早版本，可自由使用）：
- <https://github.com/a2heng/lightweight-denoise-48k>
- <https://github.com/a2heng/lightweight-aec-48k>

## 联系

- GitHub: <https://a2heng.github.io/>
- Bilibili: <https://space.bilibili.com/10850943>
