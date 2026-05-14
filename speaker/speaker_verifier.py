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
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.dirname(__file__), "..", "models", "ecapa_tdnn"),
                run_opts={"device": "cpu"}
            )
            print("ECAPA-TDNN声纹模型加载成功")
        except Exception as e:
            print(f"ECAPA-TDNN模型加载失败: {e}")
            self.model = None

    def extract_embedding(self, audio_data):
        if self.model is None:
            return None
        try:
            import torch
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            embedding = self.model.encode_batch(audio_tensor)
            emb_np = embedding.squeeze().cpu().numpy()
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-10)
            return emb_np
        except Exception as e:
            print(f"声纹特征提取失败: {e}")
            return None

    def register_speaker(self, user_id, audio_samples):
        embeddings = []
        for audio in audio_samples:
            emb = self.extract_embedding(audio)
            if emb is not None:
                embeddings.append(emb)
        if len(embeddings) < 2:
            print(f"注册失败: 有效样本不足 ({len(embeddings)}/{len(audio_samples)})")
            return False
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)
        self.embeddings_db[user_id] = {
            "embedding": avg_embedding.tolist(),
            "num_samples": len(embeddings),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_db()
        print(f"用户 '{user_id}' 注册成功 (样本数: {len(embeddings)})")
        return True

    def verify(self, user_id, audio_data):
        if user_id not in self.embeddings_db:
            print(f"用户 '{user_id}' 未注册")
            return False, 0.0
        emb = self.extract_embedding(audio_data)
        if emb is None:
            return False, 0.0
        stored_emb = np.array(self.embeddings_db[user_id]["embedding"])
        similarity = np.dot(emb, stored_emb) / (
            np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-10
        )
        is_match = similarity >= self.threshold
        print(f"声纹验证: 相似度={similarity:.4f} 阈值={self.threshold} 结果={'通过' if is_match else '拒绝'}")
        return is_match, float(similarity)

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
        save_data = {}
        for uid, data in self.embeddings_db.items():
            save_data[uid] = {
                "embedding": data["embedding"],
                "num_samples": data["num_samples"],
                "registered_at": data["registered_at"],
            }
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def _load_db(self):
        db_path = os.path.join(SPEAKER_DB_DIR, "speakers.json")
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                self.embeddings_db = json.load(f)
            print(f"已加载 {len(self.embeddings_db)} 个已注册用户")
