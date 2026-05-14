import time
import os
import numpy as np
from global_config import (
    WHISPER_MODEL_SIZE, WHISPER_LANGUAGE, WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURE, WHISPER_COMPUTE_TYPE, MODELS_DIR, SAMPLE_RATE
)


class SpeechRecognizer:
    def __init__(self, model_size=None, compute_type=None):
        self.model_size = model_size or WHISPER_MODEL_SIZE
        self.compute_type = compute_type or WHISPER_COMPUTE_TYPE
        self.language = WHISPER_LANGUAGE
        self.beam_size = WHISPER_BEAM_SIZE
        self.temperature = WHISPER_TEMPERATURE
        self.model = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        local_path = os.path.join(MODELS_DIR, "base")
        if os.path.exists(local_path) and self._try_load_ctranslate2(local_path):
            return
        if self._try_load_ctranslate2_hub():
            return
        self._try_load_openai_whisper(local_path)

    def _try_load_ctranslate2(self, model_path):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_path, compute_type=self.compute_type)
            self.model_type = "faster-whisper"
            print(f"[Whisper] CTranslate2 local model loaded: {model_path} ({self.compute_type})")
            return True
        except Exception as e:
            print(f"[Whisper] CTranslate2 local load failed: {e}")
            return False

    def _try_load_ctranslate2_hub(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, compute_type=self.compute_type)
            self.model_type = "faster-whisper"
            print(f"[Whisper] CTranslate2 hub model loaded: {self.model_size} ({self.compute_type})")
            return True
        except Exception as e:
            print(f"[Whisper] CTranslate2 hub load failed: {e}")
            return False

    def _try_load_openai_whisper(self, model_path):
        try:
            import whisper
            self.model = whisper.load_model(self.model_size, download_root=model_path)
            self.model_type = "openai-whisper"
            print(f"[Whisper] OpenAI Whisper loaded: {self.model_size}")
        except Exception as e:
            print(f"[Whisper] OpenAI Whisper load failed: {e}")
            try:
                import whisper
                self.model = whisper.load_model(self.model_size)
                self.model_type = "openai-whisper"
                print(f"[Whisper] OpenAI Whisper loaded from hub: {self.model_size}")
            except Exception as e2:
                print(f"[Whisper] All model loading attempts failed: {e2}")
                self.model = None
                self.model_type = None

    def transcribe(self, audio_data):
        if self.model is None:
            return "[模型未加载]"
        if len(audio_data) < SAMPLE_RATE * 0.3:
            return ""
        start_time = time.time()
        try:
            text = ""
            if self.model_type == "faster-whisper":
                text = self._transcribe_faster(audio_data)
            elif self.model_type == "openai-whisper":
                text = self._transcribe_openai(audio_data)
            elapsed = time.time() - start_time
            duration = len(audio_data) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else 0
            print(f"识别: '{text}' | 耗时: {elapsed:.3f}s | RTF: {rtf:.3f}")
            return text
        except Exception as e:
            print(f"识别错误: {e}")
            return ""

    def _transcribe_faster(self, audio_data):
        segments, info = self.model.transcribe(
            audio_data, language=self.language, beam_size=self.beam_size,
            temperature=self.temperature, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200)
        )
        return "".join(seg.text.strip() for seg in segments)

    def _transcribe_openai(self, audio_data):
        import torch
        audio_tensor = torch.from_numpy(audio_data).float()
        result = self.model.transcribe(
            audio_tensor, language=self.language, beam_size=self.beam_size,
            temperature=self.temperature
        )
        return result.get("text", "").strip()

    def transcribe_with_metrics(self, audio_data):
        if self.model is None:
            return "", {}
        start_time = time.time()
        text = self.transcribe(audio_data)
        elapsed = time.time() - start_time
        duration = len(audio_data) / SAMPLE_RATE
        metrics = {
            "text": text, "elapsed_sec": elapsed,
            "audio_duration_sec": duration,
            "rtf": elapsed / duration if duration > 0 else 0,
            "model_type": self.model_type,
            "compute_type": self.compute_type,
        }
        return text, metrics

    @staticmethod
    def compute_cer(reference, hypothesis):
        if not reference:
            return 0.0
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        n, m = len(ref_chars), len(hyp_chars)
        d = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        return d[n][m] / max(n, 1)
