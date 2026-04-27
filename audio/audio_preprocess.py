import numpy as np
import noisereduce as nr
from global_config import SAMPLE_RATE

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE

    def process(self, audio_data):
        # 降噪
        audio = nr.reduce_noise(y=audio_data, sr=SAMPLE_RATE)
        # 简单阈值去静音（绕开webrtcvad报错）
        mask = np.abs(audio) > 0.01
        return audio[mask]