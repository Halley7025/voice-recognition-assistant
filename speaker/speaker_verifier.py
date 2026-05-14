import os
import json
import time
import numpy as np
from global_config import SAMPLE_RATE, SPEAKER_SIMILARITY_THRESHOLD, SPEAKER_ENROLL_SAMPLES, SPEAKER_DB_DIR


class SpeakerVerifier:
    def __init__(self):
        self.model = None
        self.embeddings_db = {}
        self.threshold = SPEAKER_SIMILARITY_THRESHOLD
        self.enroll_samples_required = SPEAKER_ENROLL_SAMPLES
        self._load_model()

    def _load_model(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["SPEECHBRAIN_STRATEGY"] = "copy"
        os.environ["HF_HUB_OFFLINE"] = "1"

        cache_dir = os.path.expanduser(
            r"~\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb"
        )
        local_dir = os.path.join(os.path.dirname(__file__), "..", "models", "ecapa_tdnn")

        strategies = [
            ("HF缓存", self._try_load_from_cache, (cache_dir,)),
            ("本地目录", self._try_load_from_local, (local_dir,)),
            ("默认下载", self._try_load_default, ()),
        ]

        for name, func, args in strategies:
            try:
                if func(*args):
                    return
            except Exception as e:
                print(f"  [{name}] 失败: {e}")

        print("[Speaker] 所有加载策略均失败，声纹功能不可用")
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
        return self._init_speechbrain(model_path)

    def _try_load_from_local(self, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        hyperparams = os.path.join(local_dir, "hyperparams.yaml")
        if not os.path.exists(hyperparams):
            return False
        lt=os.path.join(local_dir,"label_encoder.txt")
        lc=os.path.join(local_dir,"label_encoder.ckpt")
        if os.path.exists(lt) and not os.path.exists(lc):
            import shutil; shutil.copy2(lt,lc)
        return self._init_speechbrain(local_dir)

    def _try_load_default(self):
        return self._init_speechbrain(None)

    def _init_speechbrain(self, savedir):
        from speechbrain.inference.speaker import EncoderClassifier
        kwargs = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "run_opts": {"device": "cpu"},
        }
        if savedir:
            kwargs["savedir"] = savedir
        self.model = EncoderClassifier.from_hparams(**kwargs)
        print(f"[Speaker] ECAPA-TDNN 加载成功")
        return True

    def extract_embedding(self, audio_data):
        if self.model is None:
            return None
        try:
            import torch
            audio_float32 = np.asarray(audio_data, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_float32).float().unsqueeze(0)
            embedding = self.model.encode_batch(audio_tensor)
            emb_np = embedding.squeeze().cpu().numpy().astype(np.float32)
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-10)
            return emb_np
        except Exception as e:
            print(f"[Speaker] 特征提取失败: {e}")
            return None

    def register_speaker(self, user_id, audio_samples):
        embeddings = []
        for audio in audio_samples:
            emb = self.extract_embedding(audio)
            if emb is not None:
                embeddings.append(emb)
        if len(embeddings) < 2:
            print(f"[Speaker] 注册失败: 有效样本不足 ({len(embeddings)}/{len(audio_samples)})")
            return False
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)
        self.embeddings_db[user_id] = {
            "embedding": avg_embedding.tolist(),
            "num_samples": len(embeddings),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_db()
        print(f"[Speaker] 用户 '{user_id}' 注册成功 (样本数: {len(embeddings)})")
        return True

    def verify(self, user_id, audio_data):
        if user_id not in self.embeddings_db:
            print(f"[Speaker] 用户 '{user_id}' 未注册")
            return False, 0.0
        emb = self.extract_embedding(audio_data)
        if emb is None:
            return False, 0.0
        stored_emb = np.array(self.embeddings_db[user_id]["embedding"], dtype=np.float32)
        similarity = float(np.dot(emb, stored_emb) / (
            np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-10
        ))
        is_match = similarity >= self.threshold
        print(f"[Speaker] 相似度={similarity:.4f} 阈值={self.threshold} 结果={'通过' if is_match else '拒绝'}")
        return is_match, similarity

    def compute_eer(self, genuine_scores, impostor_scores):
        thresholds = np.linspace(0, 1, 1000)
        min_eer = 1.0
        best_threshold = 0.5
        for t in thresholds:
            far = np.mean(impostor_scores >= t)
            frr = np.mean(genuine_scores < t)
            if abs(far - frr) < abs(min_eer - 0):
                min_eer = (far + frr) / 2
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
            with open(db_path, "r", encoding="utf-8") as f:
                self.embeddings_db = json.load(f)
            print(f"[Speaker] 已加载 {len(self.embeddings_db)} 个已注册用户")
