import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from global_config import SAMPLE_RATE, MODELS_DIR


def benchmark_whisper_quantization(audio_path=None):
    from asr.speech_recognizer import SpeechRecognizer
    from audio.audio_preprocess import AudioPreprocessor
    import soundfile as sf

    preprocessor = AudioPreprocessor()

    if audio_path and os.path.exists(audio_path):
        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = preprocessor.process(audio.astype(np.float32))
    else:
        duration = 3.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio = preprocessor.process(audio)

    compute_types = ["float32", "float16", "int8"]
    results = []

    print("\n=== Whisper 量化对比实验 ===")
    print(f"音频时长: {len(audio)/SAMPLE_RATE:.1f}s")
    print("-" * 60)

    for ct in compute_types:
        try:
            print(f"\n测试 compute_type={ct} ...")
            rec = SpeechRecognizer(compute_type=ct)
            if rec.model is None:
                print(f"  模型加载失败，跳过")
                continue
            times = []
            for _ in range(3):
                start = time.time()
                text = rec.transcribe(audio)
                elapsed = time.time() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            rtf = avg_time / (len(audio) / SAMPLE_RATE)
            results.append({
                "compute_type": ct,
                "avg_time": avg_time,
                "rtf": rtf,
                "text": text,
            })
            print(f"  平均耗时: {avg_time:.3f}s | RTF: {rtf:.3f}")
        except Exception as e:
            print(f"  错误: {e}")

    print("\n" + "=" * 60)
    print("量化对比结果:")
    print(f"{'类型':<12} {'平均耗时(s)':<15} {'RTF':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['compute_type']:<12} {r['avg_time']:<15.3f} {r['rtf']:<10.3f}")
    return results


def benchmark_noise_robustness(audio_path=None):
    from asr.speech_recognizer import SpeechRecognizer
    from audio.audio_preprocess import AudioPreprocessor
    from evaluator import Evaluator
    import soundfile as sf

    preprocessor = AudioPreprocessor()
    evaluator = Evaluator()
    recognizer = SpeechRecognizer()

    if audio_path and os.path.exists(audio_path):
        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = audio.astype(np.float32)
    else:
        duration = 3.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("\n=== 噪声鲁棒性实验 ===")
    snr_levels = [20, 15, 10, 5, 0]
    results = evaluator.evaluate_noise_robustness(recognizer, audio, snr_levels)
    return results


def benchmark_spectral_subtraction():
    from audio.audio_preprocess import AudioPreprocessor

    print("\n=== 谱减法降噪效果对比 ===")
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = np.random.randn(len(clean)).astype(np.float32) * 0.3
    noisy = clean + noise

    preprocessor = AudioPreprocessor()
    pure_noise = np.random.randn(preprocessor.frame_length * 5).astype(np.float32) * 0.3
    preprocessor.update_noise_estimate(pure_noise)

    snr_before = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    denoised = preprocessor.spectral_subtraction(noisy)
    residual_noise = denoised[:len(clean)] - clean
    snr_after = 10 * np.log10(np.mean(clean**2) / np.mean(residual_noise**2))

    print(f"降噪前SNR: {snr_before:.1f} dB")
    print(f"降噪后SNR: {snr_after:.1f} dB")
    print(f"SNR提升: {snr_after - snr_before:.1f} dB")
    return {"snr_before": snr_before, "snr_after": snr_after, "improvement": snr_after - snr_before}


if __name__ == "__main__":
    print("=" * 60)
    print("语音识别系统 - 性能基准测试")
    print("=" * 60)
    benchmark_spectral_subtraction()
    if "--full" in sys.argv:
        benchmark_whisper_quantization()
        benchmark_noise_robustness()
    else:
        print("\n提示: 使用 --full 参数运行完整基准测试 (需要较长时间)")
