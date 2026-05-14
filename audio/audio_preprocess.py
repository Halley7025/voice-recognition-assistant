import numpy as np
import scipy.signal as signal
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

    def pre_emphasis(self, audio, coeff=PRE_EMPHASIS_COEFF):
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def framing(self, audio):
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

    def spectral_subtraction(self, audio, alpha=2.0, beta=0.01):
        stft = np.fft.rfft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        if self.noise_estimate is None or self.noise_frames < 10:
            noise_mag = magnitude * 0.1
        else:
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
            self.noise_estimate = magnitude
        else:
            self.noise_estimate = (
                0.9 * self.noise_estimate + 0.1 * magnitude
            )
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
        speech_mask = energy_norm > VAD_ENERGY_THRESHOLD
        expanded_mask = np.zeros(len(audio), dtype=bool)
        for i, is_speech in enumerate(speech_mask):
            start = i * self.frame_shift
            end = min(start + self.frame_length, len(audio))
            if is_speech:
                expanded_mask[start:end] = True
        return expanded_mask

    def remove_silence(self, audio):
        mask = self.vad_energy(audio)
        if np.sum(mask) == 0:
            return audio
        return audio[mask]

    def normalize(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio

    def process(self, audio_data, use_spectral_subtraction=True):
        audio = audio_data.astype(np.float32)
        audio = self.normalize(audio)
        audio = self.pre_emphasis(audio)
        if use_spectral_subtraction:
            full_stft = np.fft.rfft(audio)
            self.update_noise_estimate(audio[:self.frame_length * 3], n_fft=len(full_stft))
            audio = self.spectral_subtraction(audio)
        audio = self.remove_silence(audio)
        audio = self.normalize(audio)
        return audio

    def extract_mel_spectrogram(self, audio, n_mels=80):
        stft = np.abs(np.fft.rfft(
            self.framing(audio) * np.hamming(self.frame_length),
            n=FFT_SIZE, axis=1
        ))  # shape: (n_frames, n_fft//2+1)
        power = stft ** 2
        mel_basis = self._mel_filterbank(n_mels, FFT_SIZE)
        mel = np.dot(mel_basis, power.T)  # shape: (n_mels, n_frames)
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
