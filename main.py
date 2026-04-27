from audio.audio_capture import AudioCapture
from audio.audio_preprocess import AudioPreprocessor
from asr.speech_recognizer import SpeechRecognizer
from controller.command_parser import CommandParser
from controller.system_controller import SystemController
import time

print("语音助手启动中...")

# 分步初始化，加打印定位
print("1/5 正在初始化 AudioCapture（麦克风）...")
cap = AudioCapture()
print("✅ AudioCapture 初始化完成")

print("2/5 正在初始化 AudioPreprocessor（预处理）...")
pre = AudioPreprocessor()
print("✅ AudioPreprocessor 初始化完成")

print("3/5 正在初始化 SpeechRecognizer（语音识别）...")
rec = SpeechRecognizer()
print("✅ SpeechRecognizer 初始化完成")

print("4/5 正在初始化 CommandParser（指令解析）...")
par = CommandParser()
print("✅ CommandParser 初始化完成")

print("5/5 正在初始化 SystemController（系统控制）...")
ctl = SystemController()
print("✅ SystemController 初始化完成")

print("准备就绪！请说话（2秒）")

while True:
    print("\n--- 开始录音 ---")
    audio = cap.record_seconds(2)
    audio = pre.process(audio)
    text = rec.transcribe(audio)
    
    print(f"识别结果：{text}")
    cmd = par.parse(text)
    if cmd:
        ctl.run(cmd)
    time.sleep(0.5)