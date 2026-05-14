import os
import json
import time
from global_config import DATA_DIR


class CommandHistory:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.history = []
        self.file_path = os.path.join(DATA_DIR, "command_history.json")
        self._load()

    def add(self, text, command, result, source="voice"):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "command": command,
            "result": result,
            "source": source,
        }
        self.history.append(entry)
        if len(self.history) > self.max_size:
            self.history = self.history[-self.max_size:]
        self._save()

    def get_recent(self, n=20):
        return self.history[-n:]

    def get_stats(self):
        if not self.history:
            return {"total": 0, "commands": {}}
        cmd_counts = {}
        for entry in self.history:
            cmd = entry.get("command", "unknown")
            cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
        return {
            "total": len(self.history),
            "commands": cmd_counts,
            "most_used": max(cmd_counts, key=cmd_counts.get) if cmd_counts else None,
        }

    def clear(self):
        self.history = []
        self._save()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
        except Exception:
            self.history = []
