import pyaudio
import numpy as np
from global_config import SAMPLE_RATE, CHANNELS, FORMAT, CHUNK

class AudioCapture:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(FORMAT//8),
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

    def record_seconds(self, seconds=2):
        frames = []
        for _ in range(int(SAMPLE_RATE / CHUNK * seconds)):
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16).astype(np.float32)/32768.0)
        return np.concatenate(frames)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()