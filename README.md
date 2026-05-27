# 语音识别助手 - 本地版

基于 Whisper + ECAPA-TDNN + 拼音模糊匹配的全本地语音识别与身份认证系统，配备毛玻璃效果的现代深色 GUI 界面。

## 项目简介

本项目实现了一个 **100% 本地运行** 的语音助手，集成语音识别、声纹验证、指令解析和系统控制四大核心能力。

- 语音输入 → Whisper 转写 → 拼音模糊匹配 + 正则解析 → 系统操作
- 声纹登录 / 声纹验证 → 指令拦截 → 权限控制
- 动态应用白名单 → 扫描桌面/开始菜单快捷方式 → 语音打开任意已安装应用

## 系统架构

`
麦克风采集 → AGC 自动增益 → 谱减法降噪 → 严格 VAD 端点检测
    ↓
Whisper small (INT8量化) → 繁→简转换 → 幻觉过滤
    → 6层指令解析: 精确匹配 → 正则模式 → 滑动窗口拼音 → 子串匹配 → 模糊匹配 → 语义组合
    ↓
ECAPA-TDNN 声纹验证 → 指令拦截门 (protected commands)
    ↓
智能应用启动器: app_path_map → 快捷方式扫描 → os.startfile
`

## 核心功能

| 功能 | 技术实现 | 说明 |
|------|---------|------|
| 语音识别 | Whisper small (CTranslate2 / OpenAI Whisper) | INT8量化，beam_size=5，repetition_penalty=1.2 |
| 声纹验证 | ECAPA-TDNN (SpeechBrain) | 注册/验证，余弦相似度阈值=0.50 |
| 音频预处理 | AGC + 谱减法 + 自适应 VAD | tanh软限幅，目标 RMS=0.1 |
| 指令解析 | 6层策略 + 拼音模糊匹配 | pypinyin + 滑动窗口 + get_close_matches |
| 应用启动 | 快捷方式扫描 + APP_PATH_MAP | 桌面/开始菜单全量扫描，支持 250+ 应用 |
| 图形界面 | PyQt5 + 毛玻璃设计 | 无边框窗口 + 流体背景 + 脉冲呼吸球 + 实时波形 |
| 声纹登录 | LoginWindow + QStackedWidget | 语音验证解锁 → 进入主界面 |
| 指令拦截 | verification_gate.py | 敏感指令需声纹验证通过 |

## 项目结构

`
├── main.py                        # 命令行交互入口
├── gui.py                         # PyQt5 主界面 (无边框 + 流体背景)
├── gui_login.py                   # 声纹登录界面
├── gui_theme.py                   # 深色主题 + QSS 样式表
├── gui_waveform.py                # 实时音频波形控件
├── gui_widgets.py                 # 毛玻璃卡片 + 脉冲呼吸球
├── global_config.py               # 全局配置常量
├── config.yaml                    # YAML 配置文件
├── CHANGELOG.txt                  # 更新日志
├── audio/
│   ├── audio_capture.py           # 麦克风采集 + 文件加载
│   └── audio_preprocess.py        # AGC + 谱减法 + VAD
├── asr/
│   └── speech_recognizer.py       # Whisper 多后端 + 幻觉过滤
├── speaker/
│   └── speaker_verifier.py        # ECAPA-TDNN 声纹验证
├── controller/
│   ├── command_parser.py          # 6层指令解析
│   ├── system_controller.py       # 25+ 指令 + 快捷方式扫描
│   ├── verification_gate.py       # 声纹验证拦截器
│   ├── command_history.py         # 指令历史
│   └── speech_feedback.py         # TTS 语音反馈
├── speaker_db/                     # 声纹特征库
└── tests/
    └── test_all.py
`

## 安装与运行

### 环境要求
- Python 3.11+
- Windows 10/11
- 麦克风设备

### 安装依赖
`ash
pip install -r requirements.txt
`

### 运行图形界面版
`ash
python gui.py
`

### 运行命令行版
`ash
python main.py
`

### 查看系统状态
`ash
python main.py --status
`

## GUI 界面

### 声纹登录页
- 脉冲呼吸球动效 + 流体背景
- 语音输入验证身份 → 自动跳转主界面
- 支持跳过验证 (访客模式)

### 主界面
- **左侧**: 语音控制 + 实时波形 + 快捷按钮 + 身份验证
- **右侧**: 执行日志 + 指令历史 + 系统设置
- **标题栏**: 状态指示器 + 用户信息 + 窗口控制

## 支持的语音指令

| 类别 | 指令示例 | 操作 |
|------|---------|------|
| 应用启动 | 打开记事本 / 打开网易云音乐 | 快捷方式扫描 + 智能启动 |
| 应用关闭 | 关闭窗口 / 退出微信 | 终止进程 |
| 音量控制 | 音量调大 / 音量调小 / 静音 | 系统音量 ±10% |
| 系统操作 | 截屏 / 锁屏 / 任务管理器 | Windows API |
| 搜索 | 搜索XXX / 百度一下 | 打开浏览器搜索 |
| 窗口管理 | 最大化 / 最小化 / 切换 | pyautogui |
| 文件操作 | 新建文件夹 / 清空回收站 | 系统命令 |

## 指令解析策略 (6层)

1. **精确匹配**: COMMAND_MAP 关键词直接映射
2. **正则模式**: 提取参数 + 应用白名单校验
3. **滑动窗口拼音**: pypinyin + 编辑距离，容忍冗余前后缀
4. **子串匹配**: Whisper截断时，"网易" → "网易云音乐"
5. **模糊匹配**: difflib.get_close_matches，"鸡死本" → "记事本"
6. **语义组合**: 关键词意图推断

## 智能应用启动器

程序启动时自动扫描以下目录的 .lnk 快捷方式：
- 用户桌面 (~\Desktop)
- 公共桌面 (%PUBLIC%\Desktop)
- 用户开始菜单 (AppData\Roaming\...\Start Menu\Programs)
- 全局开始菜单 (%PROGRAMDATA%\...\Start Menu\Programs)

扫描结果存入 app_path_map 字典，语音说“打开XXX”时直接 os.startfile() 打开对应快捷方式。

无需硬编码路径，新安装的软件只要桌面有快捷方式即可语音打开。

## 声纹验证流程

1. 注册: 录制 3 段 3秒音频 → ECAPA-TDNN 提取特征 → 存入 speaker_db
2. 验证: 录音 → strict_vad 检测 → 特征提取 → 余弦相似度 → 通过/拒绝
3. 拦截: 敏感指令 (open_app/shutdown 等) 自动触发声纹验证

## Whisper 幻觉过滤

| 检测项 | 规则 | 示例 |
|------|------|------|
| 重复检测 | 片段重复率 > 70% → 丢弃 | “嗯嗯嗯...” |
| 单字重复 | 5+ 次连续重复 → 丢弃 | “嗯嗯嗯嗯嗯嗯” |
| 最短长度 | 无动词: < 4字 → 丢弃; 有动词: < 2字 → 丢弃 | “技术” |
| 动词检测 | 无意图动词 → 丢弃 | “拔开近适的。” |
| 过长截断 | > 40字 → 截取第一句 | 连续幻觉输出 |
| 叠词去重 | 重复模式压缩 | “打开打开打开” |

## 自适应拼音阈值

| 目标应用名长度 | 相似度阈值 | 示例 |
|------|------|------|
| ≤ 2字 | ≥ 0.85 | 豆包 |
| 3字 | ≥ 0.75 | 记事本 |
| ≥ 4字 | ≥ 0.60 | 网易云音乐 |

降级策略: 拼音匹配失败 → 子串匹配 → get_close_matches 文字模糊

## 环境变量

程序自动设置以下环境变量 (gui.py 最顶部):
`python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
`

## 技术栈

| 组件 | 技术 |
|------|------|
| ASR | Whisper small (CTranslate2 INT8 / OpenAI Whisper) |
| 声纹 | ECAPA-TDNN (SpeechBrain) |
| 拼音 | pypinyin + difflib |
| GUI | PyQt5 + 毛玻璃设计 |
| 音频 | PyAudio + numpy + AGC + 谱减法 |
| 控制 | pycaw / pyautogui / ctypes / subprocess |

## 更新日志

详见 [CHANGELOG.txt](CHANGELOG.txt)

## 参考文献

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2023)
- Desplanques et al., "ECAPA-TDNN" (INTERSPEECH 2020)
- Boll, "Suppression of Acoustic Noise in Speech Using Spectral Subtraction" (IEEE TASSP 1979)

## 许可证

本项目仅供学术用途。
