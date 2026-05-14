import threading


class SpeechFeedback:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 0.8)
            voices = self.engine.getProperty("voices")
            for v in voices:
                if "chinese" in v.name.lower() or "zh" in v.id.lower():
                    self.engine.setProperty("voice", v.id)
                    break
            print("语音反馈引擎初始化成功")
        except Exception as e:
            print(f"语音反馈引擎初始化失败: {e}")
            self.engine = None

    def speak(self, text):
        if not self.enabled or not self.engine or not text:
            return
        thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        thread.start()

    def _speak_sync(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_rate(self, rate):
        if self.engine:
            self.engine.setProperty("rate", rate)

    def set_volume(self, volume):
        if self.engine:
            self.engine.setProperty("volume", max(0.0, min(1.0, volume)))
