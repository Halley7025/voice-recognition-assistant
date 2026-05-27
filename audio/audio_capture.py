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
        self._streaming = False
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

    def start_streaming(self):
        self._streaming = True

    def stop_streaming(self):
        self._streaming = False

    def stream_generator(self, chunk_duration=0.5):
        if self.stream is None:
            return
        chunk_size = int(self.sample_rate * chunk_duration)
        num_reads = max(1, chunk_size // self.chunk)
        self._streaming = True
        while self._streaming:
            frames = []
            for _ in range(num_reads):
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(
                        np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                except Exception:
                    break
            if frames:
                yield np.concatenate(frames)

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

    @staticmethod
    def load_audio_file(file_path, target_sr=16000):
        """Load audio file (WAV, MP3, FLAC, etc.) and return float32 numpy array at target_sr."""
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Try soundfile first (supports WAV, FLAC, OGG)
        try:
            import soundfile as sf
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                from scipy.signal import resample
                num_samples = int(len(data) * target_sr / sr)
                data = resample(data, num_samples).astype(np.float32)
            return data
        except Exception:
            pass

        # Try librosa (supports WAV, MP3, FLAC, OGG, etc.)
        try:
            import librosa
            data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return data.astype(np.float32)
        except Exception:
            pass

        # Fallback: wave module for WAV files
        if ext == '.wav':
            import wave
            with wave.open(file_path, 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            if sampwidth == 2:
                data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)

            if sr != target_sr:
                from scipy.signal import resample
                num_samples = int(len(data) * target_sr / sr)
                data = resample(data, num_samples).astype(np.float32)

            return data

        raise RuntimeError(f"Cannot load audio file: {file_path}. Install soundfile or librosa.")

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
