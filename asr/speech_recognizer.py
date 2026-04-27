from faster_whisper import WhisperModel
import os

# 在代码里强制配置国内镜像，避免终端环境变量不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class SpeechRecognizer:
    def __init__(self):
        print("✅ 测试模式：跳过模型加载")

    def transcribe(self, audio):
        # 模拟识别结果，直接返回预设指令
        return "打开记事本"