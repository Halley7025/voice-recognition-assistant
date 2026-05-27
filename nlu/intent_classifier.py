import os
import time
import numpy as np
from logger_config import setup_logger
_log = setup_logger(__name__)
from global_config import (
    INTENT_MODEL_NAME, INTENT_MAX_LENGTH, INTENT_LABELS,
    INTENT_TRAIN_DATA, INTENT_CONFIDENCE_THRESHOLD
)


class IntentClassifier:
    def __init__(self, lazy=True):
        self.model = None
        self.tokenizer = None
        self.bert = None
        self.intent_head = None
        self.label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
        self.num_labels = len(INTENT_LABELS)
        self.ready = False
        if not lazy:
            self._build_model()

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn
            from transformers import BertTokenizer, BertModel

            self.tokenizer = BertTokenizer.from_pretrained(INTENT_MODEL_NAME)
            self.bert = BertModel.from_pretrained(INTENT_MODEL_NAME)

            class IntentHead(nn.Module):
                def __init__(self, hidden_size, num_labels):
                    super().__init__()
                    self.dropout = nn.Dropout(0.1)
                    self.classifier = nn.Linear(hidden_size, num_labels)

                def forward(self, x):
                    x = self.dropout(x)
                    return self.classifier(x)

            self.intent_head = IntentHead(768, self.num_labels)
            self.ready = True
            _log.info("意图分类模型加载成功 (BERT-base-chinese)")
        except Exception as e:
            _log.error(f"意图分类模型加载失败: {e}")
            self.ready = False

    def encode(self, text: str):
        if self.tokenizer is None:
            return None
        inputs = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=INTENT_MAX_LENGTH, return_tensors="pt"
        )
        return inputs

    def predict(self, text: str):
        if not self.ready:
            return self._fallback_predict(text)
        try:
            import torch
            inputs = self.encode(text)
            with torch.no_grad():
                outputs = self.bert(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                logits = self.intent_head(cls_embedding)
                probs = torch.softmax(logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_id].item()
            label = self.id2label.get(pred_id, "unknown")
            if confidence < INTENT_CONFIDENCE_THRESHOLD:
                return "unknown", confidence
            return label, confidence
        except Exception as e:
            _log.error(f"意图预测错误: {e}")
            return self._fallback_predict(text)

    def _fallback_predict(self, text):
        from global_config import COMMAND_MAP
        best_match = None
        best_len = 0
        for keyword, intent in COMMAND_MAP.items():
            if keyword in text and len(keyword) > best_len:
                best_match = intent
                best_len = len(keyword)
        if best_match:
            return best_match, 0.8
        return "unknown", 0.0

    def train(self, train_data=None, epochs=5, lr=2e-5):
        if not self.ready:
            _log.warning("模型未就绪，无法训练")
            return
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        data = train_data or INTENT_TRAIN_DATA
        texts, labels = zip(*data)
        label_ids = [self.label2id.get(l, self.label2id["unknown"]) for l in labels]

        encodings = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=INTENT_MAX_LENGTH, return_tensors="pt"
        )
        label_tensor = torch.tensor(label_ids)
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], label_tensor
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(
            list(self.bert.parameters()) + list(self.intent_head.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        self.bert.train()
        self.intent_head.train()
        for epoch in range(epochs):
            total_loss = 0
            for input_ids, attention_mask, labels_batch in loader:
                optimizer.zero_grad()
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = self.intent_head(cls_emb)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
        self.bert.eval()
        self.intent_head.eval()
        _log.info("意图分类模型训练完成")
