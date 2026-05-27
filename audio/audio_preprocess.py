import numpy as np
import scipy.signal as signal
from logger_config import setup_logger
_log = setup_logger(__name__)
from global_config import (
    SAMPLE_RATE, PRE_EMPHASIS_COEFF, FRAME_LENGTH_MS,
    FRAME_SHIFT_MS, FFT_SIZE, VAD_ENERGY_THRESHOLD
)


class AudioPreprocessor:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.frame_length = int(FRAME_LENGTH_MS * sample_rate / 1000)
        self.frame_shift = int(FRAME_SHIFT_MS * sample_rate / 1000)
        self.noise_estimate = None
        self.noise_frames = 0
        self._mel_basis_cache = {}

    def pre_emphasis(self, audio: np.ndarray, coeff: float = PRE_EMPHASIS_COEFF) -> np.ndarray:
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def framing(self, audio: np.ndarray) -> np.ndarray:
        num_frames = 1 + (len(audio) - self.frame_length) // self.frame_shift
        indices = (
            np.arange(self.frame_length)[None, :]
            + np.arange(num_frames)[:, None] * self.frame_shift
        )
        indices = np.clip(indices, 0, len(audio) - 1)
        return audio[indices]

    def windowing(self, frames):
        hamming = np.hamming(self.frame_length)
        return frames * hamming

    def auto_gain_control(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """RMS-based automatic gain control with soft clipping (tanh compression).

        Boosts quiet signals to target_rms while preventing hard clipping.
        Uses tanh for smooth saturation instead of hard cutoff.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio

        # Step 1: Linear gain to reach target RMS
        gain = target_rms / rms
        # Cap gain to prevent extreme amplification of pure noise
        gain = min(gain, 50.0)
        audio = audio * gain

        # Step 2: Soft clipping via tanh compression
        # Prevents harsh clipping distortion that destroys consonants
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            # tanh compression: maps (-inf, +inf) -> (-1, +1) smoothly
            # Scale so that peaks near 1.0 map to ~0.95
            audio = np.tanh(audio * 1.2) * 0.95

        return audio

    def spectral_subtraction(self, audio, alpha=1.0, beta=0.05):
        stft = np.fft.rfft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        if self.noise_estimate is None or self.noise_frames < 3:
            noise_mag = magnitude * 0.05
        else:
            if len(self.noise_estimate) != len(magnitude):
                from scipy.signal import resample
                self.noise_estimate = resample(self.noise_estimate, len(magnitude)).clip(min=0)
            noise_mag = self.noise_estimate
        clean_mag = magnitude - alpha * noise_mag
        clean_mag = np.maximum(clean_mag, beta * magnitude)
        clean_stft = clean_mag * np.exp(1j * phase)
        return np.fft.irfft(clean_stft, n=len(audio))

    def update_noise_estimate(self, audio, n_fft=None):
        if n_fft is not None:
            stft = np.fft.rfft(audio, n=n_fft)
        else:
            stft = np.fft.rfft(audio)
        magnitude = np.abs(stft)
        if self.noise_estimate is None:
            self.noise_estimate = magnitude.copy()
        else:
            if len(self.noise_estimate) == len(magnitude):
                self.noise_estimate = 0.9 * self.noise_estimate + 0.1 * magnitude
            else:
                self.noise_estimate = magnitude.copy()
        self.noise_frames += 1

    def wiener_filter(self, audio, noise_power=None):
        stft = np.fft.rfft(audio)
        magnitude = np.abs(stft)
        power = magnitude ** 2
        if noise_power is None:
            noise_power = np.mean(power) * 0.1
        gain = np.maximum(power - noise_power, 0) / np.maximum(power, 1e-10)
        clean_stft = gain * stft
        return np.fft.irfft(clean_stft, n=len(audio))

    def vad_energy(self, audio):
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

    def strict_vad(self, audio, energy_threshold=0.005, snr_threshold=6.0,
                   min_speech_sec=0.5):
        """Strict VAD: reject audio that is mostly noise or silence.

        Returns:
            (is_valid, speech_audio, meta) tuple.
            is_valid: True if audio contains enough speech.
            speech_audio: trimmed speech portion (or original if valid).
            meta: dict with diagnostic info.
        """
        if len(audio) < self.sample_rate * 0.3:
            return False, audio, {"reason": "too_short", "duration": len(audio)/self.sample_rate}

        # Frame-level analysis
        frames = self.framing(audio)
        frame_energy = np.sum(frames ** 2, axis=1)

        # Estimate noise floor from quietest 15% of frames
        n_noise = max(1, len(frame_energy) // 7)
        noise_idx = np.argsort(frame_energy)[:n_noise]
        noise_floor = np.mean(frame_energy[noise_idx])

        # Signal level from loudest 20% of frames
        n_signal = max(1, len(frame_energy) // 5)
        signal_idx = np.argsort(frame_energy)[-n_signal:]
        signal_level = np.mean(frame_energy[signal_idx])

        # Overall RMS
        overall_rms = np.sqrt(np.mean(audio ** 2))

        # SNR estimate
        if noise_floor > 1e-10:
            snr = 10 * np.log10(signal_level / noise_floor + 1e-10)
        else:
            snr = 40.0  # Very clean signal

        # Speech frame ratio
        frame_rms = np.sqrt(frame_energy + 1e-10)
        speech_frames = np.sum(frame_rms > energy_threshold)
        speech_ratio = speech_frames / len(frame_rms) if len(frame_rms) > 0 else 0

        meta = {
            "overall_rms": float(overall_rms),
            "snr_db": float(snr),
            "speech_ratio": float(speech_ratio),
            "n_frames": len(frame_rms),
        }

        # Decision: reject if too quiet
        if overall_rms < energy_threshold * 0.5:
            meta["reason"] = "too_quiet"
            _log.info(f"VAD拒绝: 音量过低 (RMS={overall_rms:.6f})")
            return False, audio, meta

        # Decision: reject if SNR too low (noise-dominated)
        if snr < snr_threshold:
            meta["reason"] = "low_snr"
            _log.info(f"VAD拒绝: 信噪比过低 (SNR={snr:.1f}dB)")
            return False, audio, meta

        # Decision: reject if speech ratio too low
        if speech_ratio < 0.2:
            meta["reason"] = "low_speech_ratio"
            _log.info(f"VAD拒绝: 语音比例过低 ({speech_ratio:.1%})")
            return False, audio, meta

        # Extract speech portion with padding
        speech_mask = frame_rms > energy_threshold
        pad_frames = 3  # ~75ms padding
        expanded = np.zeros(len(audio), dtype=bool)
        for i, is_speech in enumerate(speech_mask):
            if is_speech:
                for j in range(max(0, i - pad_frames), min(len(speech_mask), i + pad_frames + 1)):
                    start = j * self.frame_shift
                    end = min(start + self.frame_length, len(audio))
                    expanded[start:end] = True

        speech_audio = audio[expanded]
        speech_duration = len(speech_audio) / self.sample_rate

        if speech_duration < min_speech_sec:
            meta["reason"] = "speech_too_short"
            meta["speech_duration"] = speech_duration
            _log.info(f"VAD拒绝: 语音时长过短 ({speech_duration:.2f}s < {min_speech_sec}s)")
            return False, audio, meta

        meta["reason"] = "accepted"
        meta["speech_duration"] = speech_duration
        _log.info(f"VAD通过: RMS={overall_rms:.4f} SNR={snr:.1f}dB 语音={speech_ratio:.1%} 时长={speech_duration:.2f}s")
        return True, speech_audio, meta

    def remove_silence(self, audio, pad_ms=200):
        mask = self.vad_energy(audio)
        if np.sum(mask) == 0:
            return audio
        pad_samples = int(pad_ms * self.sample_rate / 1000)
        padded_mask = np.zeros_like(mask)
        speech_indices = np.where(mask)[0]
        if len(speech_indices) > 0:
            start = max(0, speech_indices[0] - pad_samples)
            end = min(len(mask), speech_indices[-1] + pad_samples + 1)
            padded_mask[start:end] = True
        return audio[padded_mask]

    def normalize(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio

    def process_for_speaker(self, audio_data: np.ndarray) -> np.ndarray:
        """Lightweight preprocessing for speaker verification.
        Only AGC + normalize, preserves voice characteristics."""
        audio = audio_data.astype(np.float32)
        audio = self.auto_gain_control(audio, target_rms=0.1)
        audio = self.normalize(audio)
        return audio

    def process(self, audio_data: np.ndarray, use_spectral_subtraction: bool = True) -> np.ndarray:
        audio = audio_data.astype(np.float32)
        audio = self.auto_gain_control(audio, target_rms=0.1)
        audio = self.normalize(audio)
        audio = self.pre_emphasis(audio)
        if use_spectral_subtraction:
            full_stft = np.fft.rfft(audio)
            signal_len = len(audio)
            if self.noise_estimate is not None and len(audio) >= self.frame_length * 3:
                frames = self.framing(audio)
                energy = np.sum(frames ** 2, axis=1)
                n_quiet = max(1, len(energy) // 7)
                quiet_idx = np.argsort(energy)[:n_quiet]
                for idx in quiet_idx:
                    start = idx * self.frame_shift
                    end = min(start + self.frame_length, len(audio))
                    frame_audio = audio[start:end]
                    if len(frame_audio) >= self.frame_length:
                        self.update_noise_estimate(frame_audio, n_fft=signal_len)
            elif self.noise_estimate is None:
                frames = self.framing(audio) if len(audio) >= self.frame_length else audio.reshape(1, -1)
                energy = np.sum(frames ** 2, axis=1)
                if len(energy) > 1:
                    quiet_idx = np.argmin(energy)
                    start = quiet_idx * self.frame_shift
                    end = min(start + self.frame_length, len(audio))
                    self.update_noise_estimate(audio[start:end], n_fft=signal_len)
                else:
                    self.update_noise_estimate(audio[:self.frame_length], n_fft=signal_len)
            audio = self.spectral_subtraction(audio, alpha=1.0, beta=0.05)
        audio = self.remove_silence(audio, pad_ms=200)
        audio = self.normalize(audio)
        return audio

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
