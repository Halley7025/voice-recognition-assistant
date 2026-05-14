from global_config import COMMAND_MAP, KEYWORD_INTENT_MAP
from nlu.intent_classifier import IntentClassifier


class CommandParser:
    def __init__(self, use_nlu=False):
        self.keyword_map = COMMAND_MAP
        self.keyword_intent = KEYWORD_INTENT_MAP
        self.nlu = None
        self.use_nlu = use_nlu
        if use_nlu:
            try:
                self.nlu = IntentClassifier(lazy=False)
                print("NLU意图分类器加载成功")
            except Exception as e:
                print(f"NLU加载失败，使用关键词匹配: {e}")
                self.use_nlu = False
        print("指令解析器初始化完成")

    def parse(self, text):
        if not text or not text.strip():
            return None
        text = text.strip()

        if self.use_nlu and self.nlu is not None and self.nlu.ready:
            intent, confidence = self.nlu.predict(text)
            if intent != "unknown":
                print(f"NLU: '{text}' -> {intent} ({confidence:.3f})")
                return intent

        best_match = None
        best_len = 0
        for keyword, cmd in self.keyword_map.items():
            if keyword in text and len(keyword) > best_len:
                best_match = cmd
                best_len = len(keyword)
        if best_match:
            print(f"关键词匹配: '{text}' -> {best_match}")
            return best_match

        has_volume_keyword = False
        has_direction_keyword = False
        direction_intent = None
        for kw, intent in self.keyword_intent.items():
            if kw in text:
                if intent is None:
                    has_volume_keyword = True
                elif kw in ("大", "高", "增"):
                    has_direction_keyword = True
                    direction_intent = intent
                elif kw in ("小", "低", "减"):
                    has_direction_keyword = True
                    direction_intent = intent
                else:
                    if intent:
                        print(f"语义匹配: '{text}' -> {intent}")
                        return intent

        if has_volume_keyword and has_direction_keyword and direction_intent:
            print(f"语义组合: '{text}' -> {direction_intent}")
            return direction_intent

        print(f"未识别指令: '{text}'")
        return None
