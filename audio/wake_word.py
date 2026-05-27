import numpy as np
from global_config import SAMPLE_RATE

class WakeWordDetector:
    def __init__(self, wake_words=None):
        self.wake_words = wake_words or ["你好助手", "嘿助手"]
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 3.0

    def feed(self, audio_chunk, recognizer):
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        max_samples = int(SAMPLE_RATE * self.buffer_duration)
        if len(self.buffer) > max_samples:
            self.buffer = self.buffer[-max_samples:]
        if len(self.buffer) < SAMPLE_RATE * 1.5:
            return False, None
        try:
            text = recognizer.transcribe(self.buffer.astype(np.float32))
            for kw in self.wake_words:
                if kw in text:
                    self.buffer = np.array([], dtype=np.float32)
                    return True, kw
        except Exception:
            pass
        return False, None

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)

