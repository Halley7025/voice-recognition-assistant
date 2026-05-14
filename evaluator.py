import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from asr.speech_recognizer import SpeechRecognizer
from audio.audio_preprocess import AudioPreprocessor
from global_config import SAMPLE_RATE


class Evaluator:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.results = {}

    def evaluate_cer(self, recognizer, test_pairs):
        cer_scores = []
        details = []
        for audio_path, reference in test_pairs:
            try:
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                if sr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                processed = self.preprocessor.process(audio.astype(np.float32))
                hypothesis = recognizer.transcribe(processed)
                cer = SpeechRecognizer.compute_cer(reference, hypothesis)
                cer_scores.append(cer)
                details.append({
                    "file": audio_path,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "cer": cer,
                })
                print(f"CER: {cer:.4f} | REF: '{reference}' | HYP: '{hypothesis}'")
            except Exception as e:
                print(f"评估错误 [{audio_path}]: {e}")
        avg_cer = np.mean(cer_scores) if cer_scores else 1.0
        self.results["cer"] = {"avg": avg_cer, "details": details}
        print(f"\n平均CER: {avg_cer:.4f} (共 {len(cer_scores)} 条)")
        return avg_cer, details

    def evaluate_latency(self, recognizer, audio_paths):
        latencies = []
        rtf_scores = []
        for audio_path in audio_paths:
            try:
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                if sr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                processed = self.preprocessor.process(audio.astype(np.float32))
                start = time.time()
                text, metrics = recognizer.transcribe_with_metrics(processed)
                elapsed = time.time() - start
                duration = len(processed) / SAMPLE_RATE
                rtf = elapsed / duration if duration > 0 else float("inf")
                latencies.append(elapsed)
                rtf_scores.append(rtf)
                print(f"延迟: {elapsed:.3f}s | RTF: {rtf:.3f} | 时长: {duration:.1f}s")
            except Exception as e:
                print(f"延迟评估错误: {e}")
        result = {
            "avg_latency": np.mean(latencies) if latencies else 0,
            "avg_rtf": np.mean(rtf_scores) if rtf_scores else 0,
            "max_latency": np.max(latencies) if latencies else 0,
            "min_latency": np.min(latencies) if latencies else 0,
        }
        self.results["latency"] = result
        print(f"\n平均延迟: {result['avg_latency']:.3f}s | 平均RTF: {result['avg_rtf']:.3f}")
        return result

    def evaluate_noise_robustness(self, recognizer, clean_audio, snr_levels=None):
        if snr_levels is None:
            snr_levels = [20, 15, 10, 5, 0, -5]
        results = []
        clean_processed = self.preprocessor.process(clean_audio.astype(np.float32))
        clean_text = recognizer.transcribe(clean_processed)
        print(f"干净语音识别: '{clean_text}'")
        for snr_db in snr_levels:
            noise = np.random.randn(len(clean_audio))
            signal_power = np.mean(clean_audio ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noisy_audio = clean_audio + np.sqrt(noise_power) * noise
            processed = self.preprocessor.process(noisy_audio.astype(np.float32))
            noisy_text = recognizer.transcribe(processed)
            cer = SpeechRecognizer.compute_cer(clean_text, noisy_text)
            results.append({
                "snr_db": snr_db,
                "text": noisy_text,
                "cer_vs_clean": cer,
            })
            print(f"SNR={snr_db}dB: CER={cer:.4f} | '{noisy_text}'")
        self.results["noise_robustness"] = results
        return results

    def evaluate_speaker_eer(self, verifier, genuine_pairs, impostor_pairs):
        genuine_scores = []
        for audio1, audio2 in genuine_pairs:
            emb1 = verifier.extract_embedding(audio1)
            emb2 = verifier.extract_embedding(audio2)
            if emb1 is not None and emb2 is not None:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
                genuine_scores.append(sim)
        impostor_scores = []
        for audio1, audio2 in impostor_pairs:
            emb1 = verifier.extract_embedding(audio1)
            emb2 = verifier.extract_embedding(audio2)
            if emb1 is not None and emb2 is not None:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
                impostor_scores.append(sim)
        eer, threshold = verifier.compute_eer(
            np.array(genuine_scores), np.array(impostor_scores)
        )
        result = {
            "eer": eer,
            "threshold": threshold,
            "genuine_scores": genuine_scores,
            "impostor_scores": impostor_scores,
        }
        self.results["speaker_eer"] = result
        print(f"EER: {eer:.4f} | 最优阈值: {threshold:.4f}")
        return result

    def save_report(self, path="evaluation_report.json"):
        save_data = {}
        for key, val in self.results.items():
            if isinstance(val, dict):
                save_data[key] = {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in val.items()
                }
            else:
                save_data[key] = val
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"评估报告已保存: {path}")


def generate_test_audio():
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freq = 440
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    envelope = np.ones_like(t)
    attack = int(0.05 * SAMPLE_RATE)
    release = int(0.05 * SAMPLE_RATE)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    audio *= envelope
    return audio.astype(np.float32)


if __name__ == "__main__":
    evaluator = Evaluator()
    recognizer = SpeechRecognizer()
    print("\n=== 评估系统就绪 ===")
    print("使用方式:")
    print("  evaluator.evaluate_cer(recognizer, [(audio_path, ref_text), ...])")
    print("  evaluator.evaluate_latency(recognizer, [audio_path1, ...])")
    print("  evaluator.evaluate_noise_robustness(recognizer, clean_audio)")
