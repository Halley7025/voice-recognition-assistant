import os
import json
import time
import numpy as np
from logger_config import setup_logger
_log = setup_logger(__name__)
from global_config import SAMPLE_RATE, SPEAKER_SIMILARITY_THRESHOLD, SPEAKER_ENROLL_SAMPLES, SPEAKER_DB_DIR


class SpeakerVerifier:
    def __init__(self):
        self.model = None
        self.embeddings_db = {}
        self.threshold = SPEAKER_SIMILARITY_THRESHOLD
        self.enroll_samples_required = SPEAKER_ENROLL_SAMPLES
        self._load_model()
        self._load_db()

    def _load_model(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["SPEECHBRAIN_STRATEGY"] = "copy"
        cache_dir = os.path.expanduser(
            r"~\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb"
        )
        local_dir = os.path.join(os.path.dirname(__file__), "..", "models", "ecapa_tdnn")
        strategies = [
            ("本地目录", self._try_load_from_local, (local_dir,)),
            ("HF缓存", self._try_load_from_cache, (cache_dir,)),
        ]
        for name, func, args in strategies:
            try:
                if func(*args):
                    _log.info(f"ECAPA-TDNN 加载成功 [{name}]")
                    return
            except Exception as e:
                _log.warning(f"[{name}] 加载失败: {e}")
        _log.error("所有加载策略均失败，声纹功能不可用")
        self.model = None

    def _try_load_from_cache(self, cache_dir):
        if not os.path.exists(cache_dir):
            return False
        snapshots = os.path.join(cache_dir, "snapshots")
        if not os.path.exists(snapshots):
            return False
        snap_dirs = os.listdir(snapshots)
        if not snap_dirs:
            return False
        model_path = os.path.join(snapshots, snap_dirs[0])
        if not os.path.exists(os.path.join(model_path, "hyperparams.yaml")):
            return False
        lt = os.path.join(model_path, "label_encoder.txt")
        lc = os.path.join(model_path, "label_encoder.ckpt")
        if os.path.exists(lt) and not os.path.exists(lc):
            import shutil
            shutil.copy2(lt, lc)
        return self._init_speechbrain(model_path)

    def _try_load_from_local(self, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        hyperparams = os.path.join(local_dir, "hyperparams.yaml")
        if not os.path.exists(hyperparams) or os.path.getsize(hyperparams) == 0:
            return False
        lt = os.path.join(local_dir, "label_encoder.txt")
        lc = os.path.join(local_dir, "label_encoder.ckpt")
        if os.path.exists(lt) and not os.path.exists(lc):
            import shutil
            shutil.copy2(lt, lc)
        return self._init_speechbrain(local_dir)

    def _init_speechbrain(self, savedir):
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
        return True

    def extract_embedding(self, audio_data: np.ndarray):
        if self.model is None:
            _log.warning("声纹模型未加载")
            return None
        try:
            import torch
            audio_float32 = np.asarray(audio_data, dtype=np.float32)
            min_samples = int(SAMPLE_RATE * 0.5)
            if len(audio_float32) < min_samples:
                _log.warning(f"音频过短 ({len(audio_float32)/SAMPLE_RATE:.2f}s < 0.5s)")
                return None
            max_val = np.max(np.abs(audio_float32))
            if max_val > 0:
                audio_float32 = audio_float32 / max_val * 0.95
            audio_tensor = torch.from_numpy(audio_float32).float().unsqueeze(0)
            embedding = self.model.encode_batch(audio_tensor)
            emb_np = embedding.squeeze().cpu().numpy().astype(np.float32)
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-10)
            return emb_np
        except Exception as e:
            _log.error(f"特征提取失败: {e}")
            return None

    def register_speaker(self, user_id, audio_samples):
        if self.model is None:
            _log.error("声纹模型未加载，无法注册用户")
            return False
        embeddings = []
        for i, audio in enumerate(audio_samples):
            _log.info(f"提取特征 {i+1}/{len(audio_samples)} (长度: {len(audio)/SAMPLE_RATE:.2f}s)")
            emb = self.extract_embedding(audio)
            if emb is not None:
                embeddings.append(emb)
                _log.info(f"  特征 {i+1} 提取成功 (dim={len(emb)})")
            else:
                _log.warning(f"  特征 {i+1} 提取失败")
        if len(embeddings) < 1:
            _log.warning(f"注册失败: 无有效样本")
            return False
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)
        self.embeddings_db[user_id] = {
            "embedding": avg_embedding.tolist(),
            "num_samples": len(embeddings),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_db()
        _log.info(f"用户 '{user_id}' 注册成功 (样本数: {len(embeddings)})")
        return True

    def verify(self, user_id, audio_data):
        if user_id not in self.embeddings_db:
            _log.warning(f"用户 '{user_id}' 未注册")
            return False, 0.0
        emb = self.extract_embedding(audio_data)
        if emb is None:
            _log.warning("验证音频特征提取失败")
            return False, 0.0
        stored_emb = np.array(self.embeddings_db[user_id]["embedding"], dtype=np.float32)
        similarity = float(np.dot(emb, stored_emb) / (
            np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-10
        ))
        is_match = similarity >= self.threshold
        _log.info(f"声纹验证(user={user_id}): sim={similarity:.4f} thr={self.threshold} {'PASS' if is_match else 'FAIL'}")
        return is_match, similarity

    def verify_any_user(self, audio_data):
        """Verify speaker against ALL registered users.
        Returns: (is_verified, matched_user_id, similarity)
        """
        if not self.embeddings_db:
            _log.warning("无已注册用户，无法验证")
            return False, None, 0.0
        emb = self.extract_embedding(audio_data)
        if emb is None:
            _log.warning("验证音频特征提取失败")
            return False, None, 0.0
        best_user = None
        best_sim = -1.0
        for user_id, record in self.embeddings_db.items():
            stored_emb = np.array(record["embedding"], dtype=np.float32)
            sim = float(np.dot(emb, stored_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-10
            ))
            if sim > best_sim:
                best_sim = sim
                best_user = user_id
        is_verified = best_sim >= self.threshold
        if is_verified:
            _log.info(f"声纹验证通过: user={best_user} sim={best_sim:.4f}")
        else:
            _log.warning(f"声纹验证失败: best={best_user} sim={best_sim:.4f} thr={self.threshold}")
        return is_verified, best_user, best_sim

    def compute_eer(self, genuine_scores, impostor_scores):
        thresholds = np.linspace(0, 1, 1000)
        min_eer = 1.0
        best_threshold = 0.5
        for t in thresholds:
            far = np.mean(impostor_scores >= t)
            frr = np.mean(genuine_scores < t)
            eer = (far + frr) / 2
            if eer < min_eer:
                min_eer = eer
                best_threshold = t
        return min_eer, best_threshold

    def list_users(self):
        return list(self.embeddings_db.keys())

    def delete_user(self, user_id):
        if user_id in self.embeddings_db:
            del self.embeddings_db[user_id]
            self._save_db()
            return True
        return False

    def _save_db(self):
        db_path = os.path.join(SPEAKER_DB_DIR, "speakers.json")
        os.makedirs(SPEAKER_DB_DIR, exist_ok=True)
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(self.embeddings_db, f, ensure_ascii=False, indent=2)

    def _load_db(self):
        db_path = os.path.join(SPEAKER_DB_DIR, "speakers.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    self.embeddings_db = json.load(f)
                _log.info(f"已加载 {len(self.embeddings_db)} 个已注册用户")
            except Exception as e:
                _log.warning(f"声纹数据库加载失败: {e}")
                self.embeddings_db = {}
