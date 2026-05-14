import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from global_config import SAMPLE_RATE


class TestAudioPreprocess(unittest.TestCase):
    def setUp(self):
        from audio.audio_preprocess import AudioPreprocessor
        self.pre = AudioPreprocessor()
        self.test_audio = np.random.randn(32000).astype(np.float32) * 0.1
        self.test_audio[10000:15000] += 0.5 * np.sin(np.linspace(0, 100 * np.pi, 5000))

    def test_pre_emphasis(self):
        result = self.pre.pre_emphasis(self.test_audio)
        self.assertEqual(len(result), len(self.test_audio))

    def test_framing(self):
        frames = self.pre.framing(self.test_audio)
        self.assertGreater(frames.shape[0], 0)
        self.assertEqual(frames.shape[1], self.pre.frame_length)

    def test_windowing(self):
        frames = self.pre.framing(self.test_audio)
        windowed = self.pre.windowing(frames)
        self.assertEqual(windowed.shape, frames.shape)

    def test_normalize(self):
        result = self.pre.normalize(self.test_audio)
        self.assertLessEqual(np.max(np.abs(result)), 1.0)

    def test_spectral_subtraction(self):
        self.pre.update_noise_estimate(
            np.random.randn(self.pre.frame_length * 5).astype(np.float32) * 0.1
        )
        result = self.pre.spectral_subtraction(self.test_audio)
        self.assertEqual(len(result), len(self.test_audio))

    def test_vad_energy(self):
        mask = self.pre.vad_energy(self.test_audio)
        self.assertEqual(len(mask), len(self.test_audio))
        self.assertTrue(mask.any())

    def test_process(self):
        result = self.pre.process(self.test_audio)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(np.max(np.abs(result)), 1.0)

    def test_mel_spectrogram(self):
        mel = self.pre.extract_mel_spectrogram(self.test_audio[:16000])
        self.assertEqual(mel.shape[0], 80)


class TestSpeechRecognizer(unittest.TestCase):
    def test_compute_cer_identical(self):
        from asr.speech_recognizer import SpeechRecognizer
        cer = SpeechRecognizer.compute_cer("hello", "hello")
        self.assertAlmostEqual(cer, 0.0)

    def test_compute_cer_different(self):
        from asr.speech_recognizer import SpeechRecognizer
        cer = SpeechRecognizer.compute_cer("abc", "xyz")
        self.assertAlmostEqual(cer, 1.0)

    def test_compute_cer_partial(self):
        from asr.speech_recognizer import SpeechRecognizer
        cer = SpeechRecognizer.compute_cer("hello", "helo")
        self.assertGreater(cer, 0)
        self.assertLess(cer, 1)

    def test_compute_cer_empty_hyp(self):
        from asr.speech_recognizer import SpeechRecognizer
        cer = SpeechRecognizer.compute_cer("hello", "")
        self.assertAlmostEqual(cer, 1.0)

    def test_compute_cer_empty_ref(self):
        from asr.speech_recognizer import SpeechRecognizer
        cer = SpeechRecognizer.compute_cer("", "hello")
        self.assertEqual(cer, 0.0)


class TestCommandParser(unittest.TestCase):
    def setUp(self):
        from controller.command_parser import CommandParser
        self.parser = CommandParser(use_nlu=False)

    def test_exact_match(self):
        self.assertEqual(self.parser.parse("打开浏览器"), "open_browser")

    def test_partial_match(self):
        self.assertEqual(self.parser.parse("帮我打开浏览器"), "open_browser")

    def test_volume_up(self):
        self.assertEqual(self.parser.parse("音量调大"), "volume_up")

    def test_volume_down(self):
        self.assertEqual(self.parser.parse("声音小一点"), "volume_down")

    def test_semantic_composition(self):
        result = self.parser.parse("请把声音调大一点")
        self.assertEqual(result, "volume_up")

    def test_screenshot(self):
        self.assertEqual(self.parser.parse("截个图"), "screenshot")

    def test_lock_screen(self):
        self.assertEqual(self.parser.parse("锁屏"), "lock_screen")

    def test_close_window(self):
        self.assertEqual(self.parser.parse("关闭窗口"), "close_window")

    def test_unknown_command(self):
        self.assertIsNone(self.parser.parse("今天天气怎么样"))

    def test_empty_input(self):
        self.assertIsNone(self.parser.parse(""))
        self.assertIsNone(self.parser.parse(None))


class TestSystemController(unittest.TestCase):
    def setUp(self):
        from controller.system_controller import SystemController
        self.ctrl = SystemController()

    def test_volume_up(self):
        success, result = self.ctrl.run("volume_up")
        self.assertTrue(success)
        self.assertIn("音量", result)

    def test_volume_down(self):
        success, result = self.ctrl.run("volume_down")
        self.assertTrue(success)
        self.assertIn("音量", result)

    def test_unknown_command(self):
        success, result = self.ctrl.run("nonexistent_cmd")
        self.assertFalse(success)

    def test_open_notepad(self):
        success, result = self.ctrl.run("open_notepad")
        self.assertTrue(success)


class TestSpeakerVerifier(unittest.TestCase):
    def test_embedding_extraction_no_model(self):
        from speaker.speaker_verifier import SpeakerVerifier
        sv = SpeakerVerifier()
        sv.model = None
        result = sv.extract_embedding(np.random.randn(16000).astype(np.float32))
        self.assertIsNone(result)

    def test_verify_unknown_user(self):
        from speaker.speaker_verifier import SpeakerVerifier
        sv = SpeakerVerifier()
        is_match, sim = sv.verify("nonexistent_user", np.random.randn(16000).astype(np.float32))
        self.assertFalse(is_match)
        self.assertEqual(sim, 0.0)

    def test_list_users_empty(self):
        from speaker.speaker_verifier import SpeakerVerifier
        sv = SpeakerVerifier()
        sv.embeddings_db = {}
        self.assertEqual(sv.list_users(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
