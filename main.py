import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audio.audio_capture import AudioCapture
from audio.audio_preprocess import AudioPreprocessor
from asr.speech_recognizer import SpeechRecognizer
from speaker.speaker_verifier import SpeakerVerifier
from controller.command_parser import CommandParser
from controller.system_controller import SystemController


class VoiceAssistant:
    def __init__(self):
        print("=" * 50)
        print("语音识别助手 - 系统初始化中...")
        print("=" * 50)

        print("[1/6] 初始化音频采集...")
        self.capture = AudioCapture()

        print("[2/6] 初始化音频预处理器...")
        self.preprocessor = AudioPreprocessor()

        print("[3/6] 初始化语音识别引擎 (Whisper)...")
        self.recognizer = SpeechRecognizer()

        print("[4/6] 初始化声纹验证模块 (ECAPA-TDNN)...")
        self.verifier = SpeakerVerifier()
        self.verifier._load_db()

        print("[5/6] 初始化指令解析器...")
        self.parser = CommandParser(use_nlu=False)

        print("[6/6] 初始化系统控制器...")
        self.controller = SystemController()

        self.current_user = None
        self.is_authenticated = False
        self.running = False
        print("=" * 50)
        print("系统初始化完成！")
        print("=" * 50)

    def enroll_user(self, user_id):
        print(f"\n开始注册用户: {user_id}")
        print(f"请朗读任意内容，共需 {self.verifier.enroll_samples_required} 次采样...")
        samples = []
        for i in range(self.verifier.enroll_samples_required):
            print(f"\n采样 {i+1}/{self.verifier.enroll_samples_required} - 请说话...")
            raw = self.capture.record_seconds(3)
            if len(raw) == 0:
                print("未检测到音频输入，请重试")
                continue
            processed = self.preprocessor.process(raw)
            if len(processed) > SAMPLE_RATE * 0.5:
                samples.append(processed)
                print(f"采样 {i+1} 完成 (时长: {len(processed)/16000:.1f}s)")
            else:
                print("采样无效，请重试")
                i -= 1
        success = self.verifier.register_speaker(user_id, samples)
        return success

    def authenticate_user(self, user_id):
        if user_id not in self.verifier.list_users():
            print(f"用户 '{user_id}' 未注册，请先注册")
            return False
        print(f"\n身份验证: {user_id}")
        print("请说话进行声纹验证...")
        raw = self.capture.record_seconds(3)
        processed = self.preprocessor.process(raw)
        is_match, similarity = self.verifier.verify(user_id, processed)
        if is_match:
            self.current_user = user_id
            self.is_authenticated = True
            print(f"验证通过！欢迎, {user_id}!")
            return True
        else:
            print(f"验证失败 (相似度: {similarity:.4f})")
            return False

    def process_command(self, text):
        if not text or not text.strip():
            return None, "未检测到有效语音"
        cmd = self.parser.parse(text)
        if cmd:
            success, result = self.controller.run(cmd)
            return cmd, result
        return None, f"未理解指令: '{text}'"

    def listen_and_execute(self):
        print("\n请说话...")
        raw = self.capture.record_seconds(3)
        if len(raw) == 0:
            return None, "未检测到音频"
        processed = self.preprocessor.process(raw)
        if len(processed) < 16000 * 0.3:
            return None, "语音过短，已忽略"
        text = self.recognizer.transcribe(processed)
        if not text:
            return None, "未识别到有效语音"
        return self.process_command(text)

    def run_interactive(self):
        self.running = True
        print("\n" + "=" * 50)
        print("语音助手已就绪！")
        print("命令: 'enroll <用户名>' | 'login <用户名>' | 'quit' 退出")
        print("直接说话即可执行语音指令")
        print("=" * 50)

        while self.running:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    cmd, result = self.listen_and_execute()
                    if cmd:
                        print(f"执行: {cmd} -> {result}")
                    else:
                        print(result)
                    continue
                if user_input.lower() == "quit":
                    break
                elif user_input.lower().startswith("enroll "):
                    uid = user_input[7:].strip()
                    self.enroll_user(uid)
                elif user_input.lower().startswith("login "):
                    uid = user_input[6:].strip()
                    self.authenticate_user(uid)
                else:
                    cmd, result = self.process_command(user_input)
                    print(f"执行: {cmd} -> {result}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")

        self.shutdown()

    def shutdown(self):
        self.running = False
        self.capture.close()
        print("语音助手已关闭")


def main():
    assistant = VoiceAssistant()
    assistant.run_interactive()


if __name__ == "__main__":
    from global_config import SAMPLE_RATE
    main()
