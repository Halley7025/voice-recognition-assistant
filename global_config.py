import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = 16
CHUNK = 1024

PRE_EMPHASIS_COEFF = 0.97
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
FFT_SIZE = 512
N_MELS = 80

VAD_AGGRESSIVENESS = 2
VAD_FRAME_DURATION_MS = 30
VAD_ENERGY_THRESHOLD = 0.01
VAD_SPEECH_RATIO = 0.3

WHISPER_MODEL_SIZE = "base"
WHISPER_LANGUAGE = "zh"
WHISPER_BEAM_SIZE = 5
WHISPER_TEMPERATURE = 0.0
WHISPER_COMPUTE_TYPE = "int8"

SPEAKER_EMBEDDING_DIM = 192
SPEAKER_SIMILARITY_THRESHOLD = 0.65
SPEAKER_ENROLL_SAMPLES = 3

INTENT_MODEL_NAME = "bert-base-chinese"
INTENT_MAX_LENGTH = 64
INTENT_CONFIDENCE_THRESHOLD = 0.5

COMMAND_MAP = {
    "打开记事本": "open_notepad",
    "打开记事本应用": "open_notepad",
    "启动记事本": "open_notepad",
    "帮我开记事本": "open_notepad",
    "我要用记事本": "open_notepad",
    "打开浏览器": "open_browser",
    "打开浏览器应用": "open_browser",
    "启动浏览器": "open_browser",
    "打开百度": "open_browser",
    "帮我开浏览器": "open_browser",
    "我要上网": "open_browser",
    "音量调大": "volume_up",
    "增大音量": "volume_up",
    "声音大一点": "volume_up",
    "音量增大": "volume_up",
    "音量调高": "volume_up",
    "声音调大": "volume_up",
    "大声一点": "volume_up",
    "把音量调大": "volume_up",
    "音量调小": "volume_down",
    "减小音量": "volume_down",
    "声音小一点": "volume_down",
    "音量减小": "volume_down",
    "音量调低": "volume_down",
    "声音调小": "volume_down",
    "小声一点": "volume_down",
    "把音量调小": "volume_down",
    "打开计算器": "open_calculator",
    "启动计算器": "open_calculator",
    "帮我算一下": "open_calculator",
    "打开文件管理器": "open_explorer",
    "打开资源管理器": "open_explorer",
    "浏览文件": "open_explorer",
    "截屏": "screenshot",
    "截图": "screenshot",
    "截个图": "screenshot",
    "屏幕截图": "screenshot",
    "锁屏": "lock_screen",
    "锁定屏幕": "lock_screen",
    "锁住电脑": "lock_screen",
    "关闭当前窗口": "close_window",
    "关闭窗口": "close_window",
    "把窗口关了": "close_window",
    "打开任务管理器": "open_task_manager",
    "查看任务管理器": "open_task_manager",
    "打开设置": "open_settings",
    "系统设置": "open_settings",
    "打开命令行": "open_cmd",
    "打开终端": "open_cmd",
    "打开cmd": "open_cmd",
    "打开画图": "open_paint",
    "画图工具": "open_paint",
    "打开word": "open_word",
    "打开文档": "open_word",
    "打开excel": "open_excel",
    "打开表格": "open_excel",
    "打开ppt": "open_ppt",
    "打开演示文稿": "open_ppt",
    "打开微信": "open_wechat",
    "启动微信": "open_wechat",
    "打开qq": "open_qq",
    "启动qq": "open_qq",
    "静音": "mute",
    "取消静音": "unmute",
    "最大化窗口": "maximize_window",
    "窗口最大化": "maximize_window",
    "最小化窗口": "minimize_window",
    "窗口最小化": "minimize_window",
    "切换窗口": "open_taskbar",
    "新建文件夹": "new_folder",
    "清空回收站": "empty_recycle",
}

KEYWORD_INTENT_MAP = {
    "记事本": "open_notepad",
    "浏览器": "open_browser",
    "百度": "open_browser",
    "上网": "open_browser",
    "音量": None,
    "声音": None,
    "大": "volume_up",
    "小": "volume_down",
    "高": "volume_up",
    "低": "volume_down",
    "增": "volume_up",
    "减": "volume_down",
    "计算器": "open_calculator",
    "文件管理": "open_explorer",
    "资源管理": "open_explorer",
    "截屏": "screenshot",
    "截图": "screenshot",
    "锁屏": "lock_screen",
    "锁定": "lock_screen",
    "关闭": "close_window",
    "任务管理": "open_task_manager",
    "设置": "open_settings",
    "命令行": "open_cmd",
    "终端": "open_cmd",
    "画图": "open_paint",
    "word": "open_word",
    "文档": "open_word",
    "excel": "open_excel",
    "表格": "open_excel",
    "ppt": "open_ppt",
    "演示": "open_ppt",
    "微信": "open_wechat",
    "qq": "open_qq",
    "静音": "mute",
    "最大化": "maximize_window",
    "最小化": "minimize_window",
    "切换": "open_taskbar",
    "文件夹": "new_folder",
    "回收站": "empty_recycle",
}

INTENT_LABELS = [
    "open_notepad", "open_browser", "volume_up", "volume_down",
    "open_calculator", "open_explorer", "screenshot", "lock_screen",
    "close_window", "open_task_manager", "open_settings", "open_cmd",
    "open_paint", "open_word", "open_excel", "open_ppt",
    "open_wechat", "open_qq", "mute", "unmute",
    "maximize_window", "minimize_window", "open_taskbar",
    "new_folder", "empty_recycle", "unknown"
]

INTENT_TRAIN_DATA = [
    ("打开记事本", "open_notepad"), ("启动记事本", "open_notepad"), ("帮我开记事本", "open_notepad"),
    ("记事本打开", "open_notepad"), ("我要用记事本", "open_notepad"),
    ("打开浏览器", "open_browser"), ("启动浏览器", "open_browser"), ("帮我开浏览器", "open_browser"),
    ("打开百度", "open_browser"), ("上网", "open_browser"), ("我要上网", "open_browser"),
    ("音量调大", "volume_up"), ("增大音量", "volume_up"), ("声音大一点", "volume_up"),
    ("把音量调高", "volume_up"), ("大声一点", "volume_up"), ("音量增加", "volume_up"),
    ("音量调小", "volume_down"), ("减小音量", "volume_down"), ("声音小一点", "volume_down"),
    ("把音量调低", "volume_down"), ("小声一点", "volume_down"), ("音量降低", "volume_down"),
    ("打开计算器", "open_calculator"), ("启动计算器", "open_calculator"), ("帮我算一下", "open_calculator"),
    ("打开文件管理器", "open_explorer"), ("打开资源管理器", "open_explorer"), ("浏览文件", "open_explorer"),
    ("截屏", "screenshot"), ("截图", "screenshot"), ("截个图", "screenshot"), ("屏幕截图", "screenshot"),
    ("锁屏", "lock_screen"), ("锁定屏幕", "lock_screen"), ("锁住电脑", "lock_screen"),
    ("关闭当前窗口", "close_window"), ("关闭窗口", "close_window"), ("把窗口关了", "close_window"),
    ("打开任务管理器", "open_task_manager"), ("查看任务管理器", "open_task_manager"),
    ("打开设置", "open_settings"), ("系统设置", "open_settings"), ("打开命令行", "open_cmd"),
    ("打开终端", "open_cmd"), ("打开cmd", "open_cmd"),

    ("打开画图", "open_paint"), ("画图工具", "open_paint"), ("打开Word", "open_word"),
    ("打开文档", "open_word"), ("打开Excel", "open_excel"), ("打开表格", "open_excel"),
    ("打开PPT", "open_ppt"), ("打开演示", "open_ppt"), ("打开微信", "open_wechat"),
    ("启动微信", "open_wechat"), ("打开QQ", "open_qq"), ("启动QQ", "open_qq"),
    ("静音", "mute"), ("取消静音", "unmute"), ("最大化窗口", "maximize_window"),
    ("最小化窗口", "minimize_window"), ("切换窗口", "open_taskbar"),
    ("新建文件夹", "new_folder"), ("清空回收站", "empty_recycle"),
    ("今天天气怎么样", "unknown"), ("播放音乐", "unknown"), ("你好", "unknown"),
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
SPEAKER_DB_DIR = os.path.join(os.path.dirname(__file__), "speaker_db")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPEAKER_DB_DIR, exist_ok=True)
