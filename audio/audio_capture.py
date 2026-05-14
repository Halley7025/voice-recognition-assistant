import pyaudio
import numpy as np
import time
from global_config import SAMPLE_RATE, CHANNELS, FORMAT, CHUNK


class AudioCapture:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = None
        self._init_stream()

    def _init_stream(self):
        try:
            self.stream = self.p.open(
                format=self.p.get_format_from_width(FORMAT // 8),
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            print("麦克风初始化成功")
        except Exception as e:
            print(f"麦克风初始化失败: {e}")

    def record_seconds(self, seconds=3):
        if self.stream is None:
            return np.array([], dtype=np.float32)
        frames = []
        num_chunks = int(self.sample_rate / self.chunk * seconds)
        for _ in range(num_chunks):
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )
            except Exception:
                continue
        if not frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames)

    def record_until_silence(self, max_seconds=10, silence_threshold=0.01,
                             silence_duration=1.5):
        if self.stream is None:
            return np.array([], dtype=np.float32)
        all_audio = []
        silence_start = None
        start_time = time.time()
        has_speech = False

        while time.time() - start_time < max_seconds:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                chunk_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                all_audio.append(chunk_audio)
                energy = np.sqrt(np.mean(chunk_audio ** 2))
                if energy > silence_threshold:
                    has_speech = True
                    silence_start = None
                elif has_speech:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        break
            except Exception:
                continue

        if not all_audio:
            return np.array([], dtype=np.float32)
        return np.concatenate(all_audio)

    def get_device_info(self):
        info = []
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev["maxInputChannels"] > 0:
                info.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["maxInputChannels"],
                    "sample_rate": dev["defaultSampleRate"],
                })
        return info

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
