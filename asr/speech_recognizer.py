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
        hf_cache = os.path.expanduser(
            rf"~\.cache\huggingface\hub\models--Systran--faster-whisper-{self.model_size}"
        )
        local_model_dir = os.path.join(MODELS_DIR, self.model_size)

        strategies = [
            ("CTranslate2 HF缓存", self._try_load_ctranslate2, (hf_cache,)),
            ("CTranslate2 本地", self._try_load_ctranslate2, (local_model_dir,)),
            ("CTranslate2 Hub", self._try_load_ctranslate2_hub, ()),
            ("OpenAI Whisper 本地", self._try_load_openai_whisper, (local_model_dir,)),
            ("OpenAI Whisper Hub", self._try_load_openai_whisper_hub, ()),
        ]

        for name, func, args in strategies:
            try:
                if func(*args):
                    print(f"[Whisper] {name} 加载成功 ({self.compute_type})")
                    return
            except Exception as e:
                print(f"[Whisper] {name} 失败: {e}")

        print("[Whisper] 所有加载策略均失败")
        self.model = None
        self.model_type = None

    def _try_load_ctranslate2(self, model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snaps = os.listdir(snapshots_dir)
            if snaps:
                model_path = os.path.join(snapshots_dir, snaps[0])

        config_file = os.path.join(model_path, "config.json")
        vocab_file = os.path.join(model_path, "vocabulary.txt")
        model_bin = os.path.join(model_path, "model.bin")

        if not (os.path.exists(config_file) and os.path.exists(model_bin)):
            return False
        if not os.path.exists(vocab_file):
            tokenizer_file = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_file):
                self._convert_tokenizer_to_vocab(tokenizer_file, vocab_file)
            else:
                return False

        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_path, compute_type=self.compute_type)
        self.model_type = "faster-whisper"
        return True

    def _convert_tokenizer_to_vocab(self, tokenizer_path, vocab_path):
        import json
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        vocab = tokenizer_data.get("added_tokens", [])
        model_vocab = tokenizer_data.get("model", {}).get("vocab", {})
        all_tokens = []
        for token, token_id in sorted(model_vocab.items(), key=lambda x: x[1]):
            all_tokens.append(token)
        while len(all_tokens) <= max((v.get("id", 0) for v in vocab), default=0):
            all_tokens.append("")
        for v in vocab:
            idx = v.get("id", 0)
            while len(all_tokens) <= idx:
                all_tokens.append("")
            all_tokens[idx] = v.get("content", "")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in all_tokens:
                f.write(token + "\n")

    def _try_load_ctranslate2_hub(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_size, compute_type=self.compute_type)
        self.model_type = "faster-whisper"
        return True

    def _try_load_openai_whisper(self, model_path):
        import whisper
        pt_file = os.path.join(model_path, "base.pt")
        if os.path.exists(pt_file):
            self.model = whisper.load_model(self.model_size, download_root=model_path)
            self.model_type = "openai-whisper"
            return True
        return False

    def _try_load_openai_whisper_hub(self):
        import whisper
        self.model = whisper.load_model(self.model_size)
        self.model_type = "openai-whisper"
        return True

    def transcribe(self, audio_data):
        if self.model is None:
            return "[模型未加载]"
        if len(audio_data) < SAMPLE_RATE * 0.3:
            return ""
        audio_float32 = np.asarray(audio_data, dtype=np.float32)
        start_time = time.time()
        try:
            text = ""
            if self.model_type == "faster-whisper":
                text = self._transcribe_faster(audio_float32)
            elif self.model_type == "openai-whisper":
                text = self._transcribe_openai(audio_float32)
            elapsed = time.time() - start_time
            duration = len(audio_float32) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else 0
            print(f"[Whisper] '{text}' | {elapsed:.3f}s | RTF={rtf:.3f}")
            return text
        except Exception as e:
            print(f"[Whisper] 识别错误: {e}")
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
