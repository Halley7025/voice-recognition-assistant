import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from global_config import SAMPLE_RATE, MODELS_DIR
from audio.audio_preprocess import AudioPreprocessor


def compare_whisper_models(test_audio_path=None):
    from asr.speech_recognizer import SpeechRecognizer
    import soundfile as sf

    preprocessor = AudioPreprocessor()
    model_sizes = ["base", "small"]
    results = []

    if test_audio_path and os.path.exists(test_audio_path):
        audio, sr = sf.read(test_audio_path)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = preprocessor.process(audio.astype(np.float32))
    else:
        duration = 3.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio = preprocessor.process(audio)

    print("=" * 70)
    print("双模型对比实验: Whisper-base vs Whisper-small")
    print("=" * 70)
    print(f"音频时长: {len(audio)/SAMPLE_RATE:.1f}s")
    print("-" * 70)

    for size in model_sizes:
        print(f"\n测试 Whisper-{size} ...")
        try:
            rec = SpeechRecognizer(model_size=size, compute_type="int8")
            if rec.model is None:
                print(f"  模型加载失败，跳过")
                continue
            times = []
            texts = []
            for trial in range(3):
                start = time.time()
                text = rec.transcribe(audio)
                elapsed = time.time() - start
                times.append(elapsed)
                texts.append(text)
            avg_time = np.mean(times)
            rtf = avg_time / (len(audio) / SAMPLE_RATE)
            model_size_mb = _estimate_model_size(size)
            results.append({
                "model": f"whisper-{size}",
                "avg_time": avg_time,
                "rtf": rtf,
                "model_size_mb": model_size_mb,
                "text": texts[0],
                "compute_type": "int8",
            })
            print(f"  平均耗时: {avg_time:.3f}s | RTF: {rtf:.3f} | 模型大小: {model_size_mb}MB")
            print(f"  识别结果: '{texts[0]}'")
        except Exception as e:
            print(f"  错误: {e}")

    if len(results) >= 2:
        cer = SpeechRecognizer.compute_cer(results[0]["text"], results[1]["text"])
        print(f"\n两模型CER差异: {cer:.4f}")

    print("\n" + "=" * 70)
    print("对比结果汇总:")
    print(f"{'模型':<20} {'平均耗时(s)':<15} {'RTF':<10} {'模型大小(MB)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {r['avg_time']:<15.3f} {r['rtf']:<10.3f} {r['model_size_mb']:<15}")

    report_path = os.path.join(os.path.dirname(__file__), "model_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")
    return results


def _estimate_model_size(model_size):
    sizes = {"tiny": 75, "base": 142, "small": 466, "medium": 1550, "large": 3100}
    return sizes.get(model_size, 0)


def ablation_study(test_audio_path=None):
    from asr.speech_recognizer import SpeechRecognizer
    import soundfile as sf

    preprocessor = AudioPreprocessor()

    if test_audio_path and os.path.exists(test_audio_path):
        audio, sr = sf.read(test_audio_path)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = audio.astype(np.float32)
    else:
        duration = 3.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("=" * 70)
    print("消融实验: 各模块贡献分析")
    print("=" * 70)

    rec = SpeechRecognizer(compute_type="int8")
    if rec.model is None:
        print("模型加载失败")
        return

    experiments = [
        ("无预处理", lambda x: x),
        ("仅预加重", lambda x: preprocessor.pre_emphasis(x)),
        ("仅谱减法", lambda x: preprocessor.spectral_subtraction(x)),
        ("仅VAD", lambda x: preprocessor.remove_silence(x)),
        ("完整流水线", lambda x: preprocessor.process(x)),
    ]

    results = []
    for name, proc_fn in experiments:
        try:
            processed = proc_fn(audio.copy())
            if len(processed) < SAMPLE_RATE * 0.1:
                processed = audio
            start = time.time()
            text = rec.transcribe(processed)
            elapsed = time.time() - start
            rtf = elapsed / (len(processed) / SAMPLE_RATE) if len(processed) > 0 else 0
            results.append({"name": name, "text": text, "time": elapsed, "rtf": rtf})
            print(f"  {name:<12} | RTF: {rtf:.3f} | '{text}'")
        except Exception as e:
            print(f"  {name:<12} | 错误: {e}")

    print("\n消融实验完成")
    return results


if __name__ == "__main__":
    print("语音识别系统 - 对比实验\n")
    compare_whisper_models()
    print("\n" + "=" * 70 + "\n")
    ablation_study()
