# 语音识别助手 - 本地版

基于 Whisper + ECAPA-TDNN + BERT 的全本地语音识别与身份认证系统。

## 项目简介

本项目实现了一个 **100% 本地运行** 的语音助手，集成语音识别、声纹验证、意图分类和系统控制四大核心能力，彻底阻断云端依赖，保障用户隐私数据零上传。

## 系统架构

```
麦克风采集 → 预加重 → 谱减法降噪 → 能量VAD → 分帧加窗
    ↓
Whisper 推理 (INT8量化) → 文本转写
    ↓
NLU意图分类 (BERT / 关键词匹配) → 指令解析
    ↓
系统指令执行 (12种操作)
    ↓
ECAPA-TDNN 声纹验证 (注册/登录)
```

## 核心功能

| 功能 | 技术实现 | 说明 |
|------|---------|------|
| 语音识别 | Whisper (faster-whisper / OpenAI Whisper) | 支持 INT8/FP16/FP32 量化，RTF < 0.5 |
| 声纹验证 | ECAPA-TDNN (SpeechBrain) | 注册/验证/EER计算，余弦相似度阈值判定 |
| 意图分类 | BERT-base-chinese + 分类头 | 13类意图，46条训练样本，同义表达泛化 |
| 音频预处理 | 自实现谱减法 + Wiener滤波 + 能量VAD | 预加重α=0.97，25ms帧/10ms移，SNR提升1.6dB |
| 系统控制 | Windows API (pycaw/pyautogui/ctypes) | 12种指令：记事本/浏览器/音量/截屏/锁屏等 |
| 图形界面 | PyQt5 + QThread 异步 | 3标签页：语音控制/声纹注册/系统信息 |

## 项目结构

```
├── main.py                    # 命令行交互入口
├── gui.py                     # PyQt5 图形界面
├── global_config.py           # 全局配置
├── evaluator.py               # 评估脚本 (CER/EER/延迟/鲁棒性)
├── benchmark.py               # 基准测试 (量化对比/谱减法效果)
├── requirements.txt           # 依赖清单
├── audio/
│   ├── audio_capture.py       # 麦克风采集
│   └── audio_preprocess.py    # 音频预处理 (谱减法/VAD/Mel)
├── asr/
│   └── speech_recognizer.py   # 语音识别 (Whisper多后端)
├── speaker/
│   └── speaker_verifier.py    # 声纹验证 (ECAPA-TDNN)
├── nlu/
│   └── intent_classifier.py   # 意图分类 (BERT)
└── controller/
    ├── command_parser.py       # 指令解析 (三级匹配)
    └── system_controller.py    # 系统控制 (12种指令)
```

## 安装与运行

### 环境要求
- Python 3.11+
- Windows 10/11
- 麦克风设备

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行命令行版
```bash
python main.py
```

### 运行图形界面版
```bash
python gui.py
```

### 运行基准测试
```bash
python benchmark.py          # 快速测试
python benchmark.py --full   # 完整测试 (需要较长时间)
```

## 命令行交互

```
# 声纹注册
enroll <用户名>

# 声纹验证
login <用户名>

# 直接输入文字执行指令
打开浏览器

# 退出
quit
```

## 支持的语音指令

| 指令 | 操作 |
|------|------|
| 打开记事本 / 启动记事本 | 打开 Notepad |
| 打开浏览器 / 打开百度 | 打开默认浏览器 |
| 音量调大 / 声音大一点 | 系统音量 +10% |
| 音量调小 / 声音小一点 | 系统音量 -10% |
| 打开计算器 | 打开 Calculator |
| 截屏 / 截图 | 保存屏幕截图 |
| 锁屏 | 锁定工作站 |
| 关闭窗口 / 关闭当前窗口 | Alt+F4 |
| 打开任务管理器 | 打开 Task Manager |
| 打开设置 | 打开系统设置 |
| 打开命令行 / 打开终端 | 打开 CMD |

## 评估指标

| 指标 | 说明 | 实测值 |
|------|------|--------|
| CER | 字符错误率 (Character Error Rate) | 基于Levenshtein距离 |
| RTF | 实时因子 (Real-Time Factor) | int8: 0.460 |
| EER | 等错误率 (Equal Error Rate) | 基于FAR/FRR扫描 |
| SNR增益 | 谱减法降噪效果 | +1.6 dB |

## 技术亮点

1. **100%本地推理**：零云端依赖，隐私数据零上传
2. **自实现谱减法**：非调库，基于Boll 1979算法的完整实现
3. **Whisper INT8量化**：CTranslate2后端，CPU端RTF < 0.5
4. **多维融合认证**：声纹嵌入 + 余弦相似度阈值判定
5. **BERT意图分类**：支持同义表达泛化，带关键词回退
6. **三级指令匹配**：NLU → 最长关键词 → 语义组合

## 参考文献

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2023)
- Desplanques et al., "ECAPA-TDNN" (INTERSPEECH 2020)
- Boll, "Suppression of Acoustic Noise in Speech Using Spectral Subtraction" (IEEE TASSP 1979)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)

## 许可证

本项目仅供学术用途。
