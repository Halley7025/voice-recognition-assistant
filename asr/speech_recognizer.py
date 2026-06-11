import time
import os
import re
import numpy as np

# HuggingFace mirror (safety net: set before any HF import)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from logger_config import setup_logger
_log = setup_logger(__name__)

HOTWORDS = "以下是普通话的句子。"

from global_config import (
    WHISPER_MODEL_SIZE, WHISPER_LANGUAGE, WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURE, WHISPER_COMPUTE_TYPE, MODELS_DIR, SAMPLE_RATE
)


class SpeechRecognizer:
    _INTENT_VERBS = (
        "打开", "启动", "运行", "开启",
        "拔开", "拨开", "罢开", "打开",  # Whisper misrecognition variants
        "关闭", "退出", "结束",
        "搜索", "查", "搜",
        "播放", "暂停", "停止",
        "下一首", "上一首",
        "音量", "声音", "静音",
        "截图", "截屏", "锁屏",
        "输入", "打字",
        "设置", "调", "增大", "减小",
        "最大化", "最小化",
        "新建", "删除",
        "清理", "关机", "重启",
        "休眠", "注销",
        "时间", "日期", "几点",
        "切换", "回收站",
        "命令行", "任务管理",
    )

    def __init__(self, model_size=None, compute_type=None):
        self.model_size = model_size or WHISPER_MODEL_SIZE
        self.compute_type = compute_type or WHISPER_COMPUTE_TYPE
        self.language = WHISPER_LANGUAGE
        self.beam_size = WHISPER_BEAM_SIZE
        self.temperature = WHISPER_TEMPERATURE
        self.model = None
        self.model_type = None
        # Hotwords string for biasing Whisper decoder toward known terms
        self._hotwords = HOTWORDS
        self._load_model()

    def update_hotwords(self, word_list):
        """Dynamically update hotwords for Whisper decoder biasing.

        Call this after SystemController scans app shortcuts so that
        app names like "网易云音乐" receive higher acoustic weight
        during beam search decoding.

        Args:
            word_list: list of strings (app names + core verbs).
        """
        base = HOTWORDS  # keep the base prompt
        extra = " ".join(w for w in word_list if w and len(w) >= 2)
        self._hotwords = f"{base} {extra}" if extra else base
        _log.info(f"Hotwords updated: {len(word_list)} terms, total len={len(self._hotwords)}")

    def _load_model(self):
        sizes = [self.model_size]
        if "," in self.model_size:
            sizes = [s.strip() for s in self.model_size.split(",")]
        for size in sizes:
            hf_cache = os.path.expanduser(
                rf"~\.cache\huggingface\hub\models--Systran--faster-whisper-{size}"
            )
            local_model_dir = os.path.join(MODELS_DIR, size)
            strategies = [
                (f"CTranslate2 HF缓存({size})", self._try_load_ctranslate2, (hf_cache,)),
                (f"CTranslate2 本地({size})", self._try_load_ctranslate2, (local_model_dir,)),
                (f"CTranslate2 Hub({size})", self._try_load_ctranslate2_hub, ()),
                (f"OpenAI Whisper 本地({size})", self._try_load_openai_whisper, (local_model_dir,)),
                (f"OpenAI Whisper Hub({size})", self._try_load_openai_whisper_hub, ()),
            ]
            for name, func, args in strategies:
                try:
                    if func(*args):
                        _log.info(f"{name} 加载成功 ({self.compute_type})")
                        return
                except Exception as e:
                    _log.warning(f"{name} 失败: {e}")
        _log.error("所有加载策略均失败")
        self.model = None
        self.model_type = None

    @staticmethod
    def _detect_whisper_device():
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        return "cuda"
                except Exception:
                    pass
                _log.info("CUDA not usable, using CPU")
        except Exception:
            pass
        return "cpu"

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
        device = self._detect_whisper_device()
        self.model = WhisperModel(model_path, device=device, compute_type=self.compute_type)
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
        device = self._detect_whisper_device()
        self.model = WhisperModel(self.model_size, device=device, compute_type=self.compute_type)
        self.model_type = "faster-whisper"
        return True

    def _try_load_openai_whisper(self, model_path):
        import whisper
        pt_file = os.path.join(model_path, f"{self.model_size}.pt")
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
        if len(audio_data) < SAMPLE_RATE * 0.2:
            return ""
        audio_float32 = np.asarray(audio_data, dtype=np.float32)
        if audio_float32.size == 0 or np.max(np.abs(audio_float32)) < 1e-7:
            return ""
        start_time = time.time()
        try:
            text = ""
            if self.model_type == "faster-whisper":
                text = self._transcribe_faster(audio_float32)
            elif self.model_type == "openai-whisper":
                text = self._transcribe_openai(audio_float32)
            text = self._to_simplified(text)
            text = self._remove_hallucination(text)
            elapsed = time.time() - start_time
            duration = len(audio_float32) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else 0
            _log.info(f"'{text}' | {elapsed:.3f}s | RTF={rtf:.3f}")
            return text
        except Exception as e:
            _log.error(f"识别错误: {e}")
            return ""

    def _transcribe_faster(self, audio_data):
        segments, info = self.model.transcribe(
            audio_data, language=self.language, beam_size=3,
            temperature=0.0, vad_filter=True,
            initial_prompt=self._hotwords,
            hotwords=self._hotwords,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=400, threshold=0.3),
            no_speech_threshold=0.35,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-0.8,
            suppress_blank=True, suppress_tokens=[-1],
            repetition_penalty=1.1,
        )
        return "".join(seg.text.strip() for seg in segments)

    def _transcribe_openai(self, audio_data):
        import torch
        audio_tensor = torch.from_numpy(audio_data).float()
        result = self.model.transcribe(
            audio_tensor, language=self.language, beam_size=self.beam_size,
            temperature=self.temperature, initial_prompt=self._hotwords,
            no_speech_threshold=0.35, condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
        )
        return result.get("text", "").strip()

    @staticmethod
    def _to_simplified(text):
        _t2s = {
            "\u958b": "\u5f00", "\u8855": "\u542f", "\u8853": "\u672f",
            "\u8a18": "\u8bb0", "\u528f": "\u6d4f", "\u89bd": "\u89c8",
            "\u8072": "\u58f0", "\u9396": "\u9501", "\u87a2": "\u8424",
            "\u5716": "\u56fe", "\u96fb": "\u7535", "\u8a08": "\u8ba1",
            "\u8a2d": "\u8bbe", "\u8a0a": "\u8baf", "\u865f": "\u53f7",
            "\u78bc": "\u7801", "\u6a94": "\u6863", "\u5939": "\u5939",
            "\u8996": "\u89c6", "\u9375": "\u952e", "\u76e4": "\u76d8",
            "\u6a19": "\u6807", "\u8a71": "\u8bdd", "\u8a9e": "\u8bed",
            "\u9ad4": "\u4f53", "\u57f7": "\u6267", "\u904b": "\u8fd0",
            "\u9023": "\u8fde", "\u7dda": "\u7ebf", "\u7db2": "\u7f51",
            "\u9801": "\u9875", "\u6a5f": "\u673a", "\u52d5": "\u52a8",
            "\u554f": "\u95ee", "\u984c": "\u9898", "\u5831": "\u62a5",
            "\u6578": "\u6570", "\u64da": "\u636e", "\u5eab": "\u5e93",
            "\u7c21": "\u7b80", "\u5c0d": "\u5bf9", "\u8acb": "\u8bf7",
            "\u5e6b": "\u5e2e", "\u5ee3": "\u5e7f", "\u7fa9": "\u4e49",
            "\u8ff4": "\u56de", "\u6230": "\u6218", "\u52d9": "\u52a1",
            "\u975c": "\u9759", "\u96b1": "\u9690", "\u986f": "\u663e",
            "\u8abf": "\u8c03", "\u7bc0": "\u8282", "\u6e1b": "\u51cf",
            "\u58d3": "\u538b", "\u7e2e": "\u7f29", "\u522a": "\u5220",
            "\u8907": "\u590d", "\u8cbc": "\u8d34", "\u88fd": "\u5236",
            "\u9304": "\u5f55", "\u95b1": "\u9605", "\u8b80": "\u8bfb",
            "\u5beb": "\u5199", "\u95dc": "\u5173", "\u9054": "\u8fbe",
            "\u8a66": "\u8bd5", "\u6e2c": "\u6d4b", "\u96f2": "\u4e91",
            "\u6a02": "\u4e50", "\u9ede": "\u70b9", "\u8edf": "\u8f6f",
            "\u9280": "\u94f6", "\u9322": "\u94b1", "\u93c8": "\u94fe",
            "\u983b": "\u9891", "\u98db": "\u98de", "\u8cb7": "\u4e70",
            "\u8ce3": "\u5356", "\u8eca": "\u8f66", "\u9580": "\u95e8",
            "\u96d9": "\u53cc", "\u8aaa": "\u8bf4", "\u8b70": "\u8bae",
            "\u8ad6": "\u8bba", "\u8a55": "\u8bc4", "\u767c": "\u53d1",
        }
        result = text
        for trad, simp in _t2s.items():
            result = result.replace(trad, simp)
        return result

    def _remove_hallucination(self, text):
        if not text:
            return text

        # 1. Excessive repetition
        segments = re.split(r"[\u3001\u3002\uff01\uff1f\s]+", text)
        segments = [s for s in segments if s]
        if len(segments) > 3:
            unique = set(segments)
            if len(unique) / len(segments) < 0.3:
                _log.warning(f"HALLUC(repeat): '{text[:30]}'")
                return ""

        # 2. Single-char repetition
        if re.match(r"^(.)\1{5,}$", text.strip()):
            _log.warning(f"HALLUC(char_repeat): '{text[:10]}'")
            return ""

        # 3. Check if any known intent verb is present (before length check)
        has_verb = any(verb in text for verb in self._INTENT_VERBS)

        # 4. Remove punctuation and check minimum length
        clean = re.sub(r"[\s\u3000-\u303f\uff00-\uffef.,!?;:\-\(\)\[\]\{}]", "", text)
        min_len = 2 if has_verb else 3
        if len(clean) < min_len:
            _log.warning(f"HALLUC(too_short): '{text}' -> '{clean}'")
            return ""

        # 5. No verb at all = likely hallucination
        if not has_verb:
            _log.warning(f"HALLUC(no_verb): '{text}'")
            return ""

        # 6. Too long
        if len(text) > 40:
            first = re.split(r"[\u3002\uff01\uff1f\n]", text)[0]
            return first if first else text[:30]

        # 7. Trailing repeated patterns
        deduped = re.sub(r"(\S{2,})(\s+\1){2,}", r"\1", text)
        return deduped.strip()

    def transcribe_with_metrics(self, audio_data):
        if self.model is None:
            return "", {}
        start_time = time.time()
        text = self.transcribe(audio_data)
        elapsed = time.time() - start_time
        duration = len(audio_data) / SAMPLE_RATE
        return text, {
            "text": text, "elapsed_sec": elapsed,
            "audio_duration_sec": duration,
            "rtf": elapsed / duration if duration > 0 else 0,
            "model_type": self.model_type, "compute_type": self.compute_type,
        }

    def transcribe_stream(self, audio_generator):
        if self.model is None:
            return
        buffer = np.array([], dtype=np.float32)
        min_samples = SAMPLE_RATE * 1
        for chunk in audio_generator:
            buffer = np.concatenate([buffer, chunk])
            if len(buffer) < min_samples:
                continue
            text = self.transcribe(buffer.astype(np.float32))
            if text:
                yield text, True
            buffer = np.array([], dtype=np.float32)

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
