"""Audio preprocessing with Silero VAD and linear peak normalization.

Pipeline: Silero VAD -> speech extraction -> peak normalization -> pre-emphasis.

Key design principles:
  - No non-linear transforms (no tanh, no spectral subtraction)
  - Silero VAD (ONNX) for precise speech endpoint detection
  - Linear peak normalization preserves transient fidelity
  - All operations in-memory, zero disk I/O
"""
import os
import numpy as np
from logger_config import setup_logger
_log = setup_logger(__name__)

from global_config import (
    SAMPLE_RATE, PRE_EMPHASIS_COEFF, FRAME_LENGTH_MS,
    FRAME_SHIFT_MS, FFT_SIZE, VAD_ENERGY_THRESHOLD
)

# Silero VAD model path (bundled with faster-whisper)
_SILERO_VAD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "silero_vad.onnx"
)
# Fallback: faster-whisper ships silero_vad_v6.onnx
_FW_SILERO_PATH = None
try:
    import faster_whisper
    _fw_dir = os.path.dirname(faster_whisper.__file__)
    _candidate = os.path.join(_fw_dir, "assets", "silero_vad_v6.onnx")
    if os.path.isfile(_candidate):
        _FW_SILERO_PATH = _candidate
except ImportError:
    pass


class AudioPreprocessor:
    """Audio preprocessor with Silero VAD and linear peak normalization.

    Replaces legacy AGC (tanh) + spectral subtraction with:
      1. Silero VAD (ONNX, CPU, ~ms latency) for precise endpoint detection
      2. Linear peak normalization (target peak = 0.7) for safe gain
      3. Pre-emphasis filter (optional, for ASR compatibility)

    Attributes:
        sample_rate: Audio sample rate (default 16000).
        frame_length: Frame length in samples (for legacy compatibility).
        frame_shift: Frame shift in samples (for legacy compatibility).
    """

    # Chunk size for Silero VAD v6 (fixed by model architecture)
    _SILERO_CHUNK_SIZE = 576

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.frame_length = int(FRAME_LENGTH_MS * sample_rate / 1000)
        self.frame_shift = int(FRAME_SHIFT_MS * sample_rate / 1000)
        self._mel_basis_cache = {}

        # Silero VAD state
        self._silero_session = None
        self._silero_threshold = 0.45  # Speech probability threshold (raised for precision)
        self._last_max_vad_prob = 0.0  # 最近一次 VAD 扫描的最大语音概率
        self._load_silero_vad()

    def _load_silero_vad(self):
        """Load Silero VAD ONNX model (try local, then faster-whisper bundled)."""
        try:
            import onnxruntime as ort
            # Priority: local model > faster-whisper bundled
            for candidate in [_SILERO_VAD_PATH, _FW_SILERO_PATH]:
                if candidate and os.path.isfile(candidate):
                    self._silero_session = ort.InferenceSession(
                        candidate, providers=["CPUExecutionProvider"]
                    )
                    _log.info(f"Silero VAD loaded: {os.path.basename(candidate)}")
                    return
            _log.warning("Silero VAD model not found, falling back to energy VAD")
        except Exception as e:
            _log.warning(f"Silero VAD load failed: {e}, falling back to energy VAD")

    # ------------------------------------------------------------------
    # Core: Silero VAD speech detection
    # ------------------------------------------------------------------
    def get_speech_timestamps(self, audio, threshold=None, min_speech_ms=250,
                              min_silence_ms=100):
        """Detect speech segments using Silero VAD (ONNX).

        Scans audio in 576-sample chunks, queries the neural VAD model
        for per-chunk speech probability, then merges contiguous speech
        regions.

        Args:
            audio: float32 numpy array, 16kHz mono.
            threshold: speech probability threshold (default 0.5).
            min_speech_ms: minimum speech segment duration to keep.
            min_silence_ms: minimum silence gap to split segments.

        Returns:
            List of dicts: [{"start": sample_idx, "end": sample_idx}, ...]
            Empty list if no speech detected.
        """
        if threshold is None:
            threshold = self._silero_threshold

        # Fallback to energy-based VAD if Silero not available
        if self._silero_session is None:
            return self._energy_vad_timestamps(audio, threshold)

        chunk_size = self._SILERO_CHUNK_SIZE
        # Pad audio to multiple of chunk_size
        n_chunks = len(audio) // chunk_size
        if n_chunks == 0:
            return []

        # Reshape into [n_chunks, chunk_size]
        chunks = audio[:n_chunks * chunk_size].reshape(n_chunks, chunk_size).copy()

        # Run inference with hidden state carry-over
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        probs = np.zeros(n_chunks, dtype=np.float32)

        # Process in batches to avoid memory spikes (batch_size=512 chunks = ~18s)
        batch_size = 512
        for i in range(0, n_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            out, h, c = self._silero_session.run(
                None, {"input": batch, "h": h, "c": c}
            )
            probs[i:i + len(batch)] = out[:len(batch)]

        # 记录最大 VAD 概率，供 strict_vad 等上层方法使用
        self._last_max_vad_prob = float(np.max(probs)) if n_chunks > 0 else 0.0

        # Build binary speech mask from probabilities
        speech_mask = probs >= threshold

        # Merge contiguous speech regions
        min_speech_chunks = max(1, int(min_speech_ms / 1000 * self.sample_rate / chunk_size))
        min_silence_chunks = max(1, int(min_silence_ms / 1000 * self.sample_rate / chunk_size))

        segments = []
        in_speech = False
        seg_start = 0
        silence_count = 0

        for i in range(n_chunks):
            if speech_mask[i]:
                if not in_speech:
                    seg_start = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_chunks:
                        seg_end = i - silence_count + 1
                        if seg_end - seg_start >= min_speech_chunks:
                            segments.append({
                                "start": seg_start * chunk_size,
                                "end": seg_end * chunk_size
                            })
                        in_speech = False
                        silence_count = 0

        # Handle trailing speech
        if in_speech:
            seg_end = n_chunks
            if seg_end - seg_start >= min_speech_chunks:
                segments.append({
                    "start": seg_start * chunk_size,
                    "end": seg_end * chunk_size
                })

        return segments

    def _energy_vad_timestamps(self, audio, threshold):
        """Fallback: energy-based VAD when Silero is unavailable."""
        frames = self.framing(audio)
        energy = np.sum(frames ** 2, axis=1)
        max_e = np.max(energy) if len(energy) > 0 else 1.0
        if max_e <= 0:
            return []
        norm_e = energy / max_e
        mask = norm_e > max(threshold * 0.1, 0.01)

        segments = []
        in_seg = False
        seg_start = 0
        for i, m in enumerate(mask):
            if m and not in_seg:
                seg_start = i
                in_seg = True
            elif not m and in_seg:
                segments.append({
                    "start": seg_start * self.frame_shift,
                    "end": min(i * self.frame_shift + self.frame_length, len(audio))
                })
                in_seg = False
        if in_seg:
            segments.append({
                "start": seg_start * self.frame_shift,
                "end": len(audio)
            })
        return segments

    # ------------------------------------------------------------------
    # Core: linear peak normalization
    # ------------------------------------------------------------------
    @staticmethod
    def peak_normalize(audio, target_peak=0.7):
        """Linear peak normalization: scale audio so max |sample| = target_peak.

        This is a pure linear gain — zero harmonic distortion, zero transient
        smearing. Preserves all spectral content including plosive consonants.

        Args:
            audio: float32 numpy array.
            target_peak: desired absolute peak value (default 0.7, approx -3dBFS).

        Returns:
            Normalized audio (float32). Returns original if silent.
        """
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio
        return (audio * (target_peak / peak)).astype(np.float32)

    # ------------------------------------------------------------------
    # Strict VAD (wraps Silero VAD for gui.py / gui_login.py)
    # ------------------------------------------------------------------
    def strict_vad(self, audio, energy_threshold=0.005, snr_threshold=6.0,
                     min_speech_sec=0.3):
          """Strict VAD using Silero VAD + energy/SNR checks.

          Returns (is_valid, speech_audio, meta).
          meta 始终包含 'max_vad_prob' 字段，表示有效语音段中的最大 VAD 概率值，
          供上层双重幻觉拦截网进行声学置信度交叉验证。
          """
          if len(audio) < self.sample_rate * 0.3:
              return False, audio, {"reason": "too_short", "max_vad_prob": 0.0}

          overall_rms = np.sqrt(np.mean(audio ** 2))

          # Use Silero VAD to find speech segments
          # get_speech_timestamps 会将本次扫描的最大 VAD 概率存入 self._last_max_vad_prob
          segments = self.get_speech_timestamps(audio, threshold=0.45)
          max_vad_prob = self._last_max_vad_prob

          if not segments:
              # No speech detected by Silero → check pure energy as last resort
              if overall_rms < energy_threshold:
                  return False, audio, {"reason": "silence", "rms": overall_rms, "max_vad_prob": max_vad_prob}
              return False, audio, {"reason": "no_speech_detected", "max_vad_prob": max_vad_prob}

          # Calculate total speech duration
          total_speech_samples = sum(s["end"] - s["start"] for s in segments)
          speech_duration = total_speech_samples / self.sample_rate

          if speech_duration < min_speech_sec:
              return False, audio, {
                  "reason": "speech_too_short",
                  "speech_duration": speech_duration,
                  "max_vad_prob": max_vad_prob
              }

          # Extract speech with padding
          speech_audio = self.extract_speech(audio, segments, pad_ms=400)

          _log.info(
              f"VAD通过: RMS={overall_rms:.4f} 语音={speech_duration:.2f}s 段数={len(segments)} 最大概率={max_vad_prob:.3f}"
          )
          return True, speech_audio, {
              "reason": "accepted",
              "speech_duration": speech_duration,
              "segments": len(segments),
              "max_vad_prob": max_vad_prob
          }

    def extract_speech(self, audio, segments=None, pad_ms=200):
        """Extract speech segments from audio with configurable padding.

        Args:
            audio: float32 numpy array.
            segments: list of {start, end} dicts (from get_speech_timestamps).
                      If None, runs get_speech_timestamps first.
            pad_ms: padding in milliseconds before/after each segment.

        Returns:
            Concatenated speech audio (float32). Returns original if no segments.
        """
        if segments is None:
            segments = self.get_speech_timestamps(audio)
        if not segments:
            return audio

        pad_samples = int(pad_ms * self.sample_rate / 1000)
        chunks = []
        for seg in segments:
            start = max(0, seg["start"] - pad_samples)
            end = min(len(audio), seg["end"] + pad_samples)
            chunks.append(audio[start:end])

        if not chunks:
            return audio
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # Soft spectral gate noise reduction
    # ------------------------------------------------------------------
    def denoise(self, audio, gate_floor=0.02, frame_ms=20):
        """Soft spectral gate noise reduction.

        Uses scipy.signal.stft/istft for robust STFT reconstruction,
        with a Wiener-style gain mask estimated from quiet frames.

        Args:
            audio: float32 mono, 16kHz.
            gate_floor: minimum gain (0.02 = -34dB max attenuation).
            frame_ms: frame size in ms.

        Returns:
            Denoised audio (float32, same length).
        """
        if len(audio) < self.frame_length * 2:
            return audio

        try:
            from scipy.signal import stft, istft
        except ImportError:
            return audio

        nperseg = 512
        noverlap = nperseg - int(frame_ms * self.sample_rate / 1000)  # ~320 step

        # STFT
        f, t_stft, Zxx = stft(audio, fs=self.sample_rate,
                               nperseg=nperseg, noverlap=noverlap,
                               window='hann', padded=True)

        magnitude = np.abs(Zxx).astype(np.float32)
        phase = np.angle(Zxx).astype(np.float32)
        power = magnitude ** 2

        # Estimate noise from quietest 20% of frames
        frame_energy = np.sum(power, axis=0)
        n_quiet = max(1, len(frame_energy) // 5)
        quiet_idx = np.argsort(frame_energy)[:n_quiet]
        noise_power = np.mean(power[:, quiet_idx], axis=1, keepdims=True).astype(np.float32)

        # Wiener gain: G = max(1 - noise/signal, floor)
        gain = np.maximum(1.0 - noise_power / np.maximum(power, 1e-10), gate_floor)

        # Temporal smoothing (suppress musical noise artifacts)
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(gain, size=3, axis=1).astype(np.float32)

        # Apply gain and reconstruct
        clean_Zxx = (magnitude * gain) * np.exp(1j * phase)
        _, clean_audio = istft(clean_Zxx, fs=self.sample_rate,
                                nperseg=nperseg, noverlap=noverlap,
                                window='hann')

        # Match length
        if len(clean_audio) > len(audio):
            clean_audio = clean_audio[:len(audio)]
        elif len(clean_audio) < len(audio):
            clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))

        return clean_audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Pre-emphasis (kept for ASR compatibility)
    # ------------------------------------------------------------------
    def pre_emphasis(self, audio: np.ndarray, coeff: float = PRE_EMPHASIS_COEFF) -> np.ndarray:
        """Pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    # ------------------------------------------------------------------
    # Legacy compatibility (framing, windowing, normalize, mel)
    # ------------------------------------------------------------------
    def framing(self, audio: np.ndarray) -> np.ndarray:
        num_frames = 1 + (len(audio) - self.frame_length) // self.frame_shift
        indices = (
            np.arange(self.frame_length)[None, :]
            + np.arange(num_frames)[:, None] * self.frame_shift
        )
        indices = np.clip(indices, 0, len(audio) - 1)
        return audio[indices]

    def windowing(self, frames):
        return frames * np.hamming(self.frame_length)

    def normalize(self, audio):
        """Legacy normalize (peak = 0.95). Prefer peak_normalize for new code."""
        return self.peak_normalize(audio, target_peak=0.95)

    # ------------------------------------------------------------------
    # Deprecated: kept as no-ops for backward compatibility
    # ------------------------------------------------------------------
    def auto_gain_control(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """DEPRECATED: replaced by peak_normalize. Returns audio unchanged."""
        _log.debug("auto_gain_control called but is deprecated; using peak_normalize")
        return self.peak_normalize(audio, target_peak=0.7)

    def spectral_subtraction(self, audio, alpha=1.0, beta=0.05):
        """DEPRECATED: removed. Returns audio unchanged."""
        _log.debug("spectral_subtraction called but is deprecated (removed)")
        return audio

    def update_noise_estimate(self, audio, n_fft=None):
        """DEPRECATED: removed. No-op."""
        pass

    def wiener_filter(self, audio, noise_power=None):
        """DEPRECATED: removed. Returns audio unchanged."""
        return audio

    def vad_energy(self, audio):
        """Legacy energy VAD. Returns boolean mask."""
        frames = self.framing(audio)
        energy = np.sum(frames ** 2, axis=1)
        max_energy = np.max(energy) if len(energy) > 0 else 1.0
        if max_energy > 0:
            energy_norm = energy / max_energy
        else:
            energy_norm = energy
        median_energy = np.median(energy_norm) if len(energy_norm) > 0 else 0
        threshold = max(VAD_ENERGY_THRESHOLD, median_energy * 1.5)
        speech_mask = energy_norm > threshold
        expanded_mask = np.zeros(len(audio), dtype=bool)
        for i, is_speech in enumerate(speech_mask):
            start = i * self.frame_shift
            end = min(start + self.frame_length, len(audio))
            if is_speech:
                expanded_mask[start:end] = True
        return expanded_mask

    def remove_silence(self, audio, pad_ms=200):
        """Remove silence using Silero VAD + padding."""
        segments = self.get_speech_timestamps(audio)
        return self.extract_speech(audio, segments, pad_ms=pad_ms)

    # ------------------------------------------------------------------
    # Main processing pipelines
    # ------------------------------------------------------------------
    def process_for_speaker(self, audio_data: np.ndarray) -> np.ndarray:
        """Lightweight preprocessing for speaker verification.

        Only Silero VAD + peak normalization. Preserves voice characteristics.
        """
        audio = audio_data.astype(np.float32)
        # VAD to extract speech
        segments = self.get_speech_timestamps(audio, threshold=0.45)
        if segments:
            audio = self.extract_speech(audio, segments, pad_ms=200)
        # Linear peak normalization (safe for speaker embeddings)
        audio = self.peak_normalize(audio, target_peak=0.7)
        return audio

    def process(self, audio_data: np.ndarray, use_spectral_subtraction: bool = True) -> np.ndarray:
        """Fast preprocessing pipeline for ASR.

        Pipeline: peak normalize -> pre-emphasis.
        VAD and speech extraction already done by strict_vad() upstream.
        Skips denoise (STFT too slow for short commands).
        """
        audio = audio_data.astype(np.float32)

        # Peak normalization
        audio = self.peak_normalize(audio, target_peak=0.7)

        # Pre-emphasis (boost high frequencies for Whisper)
        if len(audio) > 1:
            audio = self.pre_emphasis(audio)

        return audio

    # ------------------------------------------------------------------
    # Mel spectrogram (unchanged)
    # ------------------------------------------------------------------
    def extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
        stft = np.abs(np.fft.rfft(
            self.framing(audio) * np.hamming(self.frame_length),
            n=FFT_SIZE, axis=1
        ))
        power = stft ** 2
        cache_key = (n_mels, FFT_SIZE)
        if cache_key not in self._mel_basis_cache:
            self._mel_basis_cache[cache_key] = self._mel_filterbank(n_mels, FFT_SIZE)
        mel_basis = self._mel_basis_cache[cache_key]
        mel = np.dot(mel_basis, power.T)
        log_mel = 10 * np.log10(np.maximum(mel, 1e-10))
        log_mel = log_mel - np.max(log_mel)
        return log_mel

    def _mel_filterbank(self, n_mels, n_fft):
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)
        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(n_mels):
            for k in range(bin[m], bin[m + 1]):
                fbank[m, k] = (k - bin[m]) / (bin[m + 1] - bin[m] + 1e-10)
            for k in range(bin[m + 1], bin[m + 2]):
                fbank[m, k] = (bin[m + 2] - k) / (bin[m + 2] - bin[m + 1] + 1e-10)
        return fbank
