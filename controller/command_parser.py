import re
import difflib
from logger_config import setup_logger
_log = setup_logger(__name__)
from global_config import COMMAND_MAP, KEYWORD_INTENT_MAP

try:
    from pypinyin import lazy_pinyin, Style
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False
    _log.warning("pypinyin not installed, pinyin matching disabled")


# ============================================================
# Supported apps whitelist (for open_app validation)
# ============================================================
SUPPORTED_APPS = [
    "记事本", "浏览器", "计算器", "文件管理器", "资源管理器",
    "命令行", "终端", "画图", "任务管理器", "设置", "控制面板",
    "微信", "QQ", "网易云音乐", "酷狗音乐", "QQ音乐",
    "钉钉", "腾讯会议", "飞书",
    "Word", "Excel", "PPT",
    "VSCode", "PyCharm", "Photoshop",
    "Chrome", "Edge", "Firefox",
    "播放器", "录音机", "截图工具", "远程桌面",
    "记事本应用", "计算器应用", "浏览器应用",
]

# Pre-compute pinyin for all supported apps
_APPS_PINYIN = {}
if _HAS_PYPINYIN:
    for app in SUPPORTED_APPS:
        _APPS_PINYIN[app] = lazy_pinyin(app, style=Style.NORMAL)


# Global app path map (populated by SystemController on init)
APP_PATH_MAP = {}


def inject_apps(app_names, path_map=None):
    """Dynamically add app names to the whitelist and rebuild pinyin index.

    Args:
        app_names: list of app name strings to add.
        path_map: optional dict {name: shortcut_path} for direct execution.
    """
    global _APPS_PINYIN, APP_PATH_MAP
    if path_map:
        APP_PATH_MAP.update(path_map)
    added = 0
    for name in app_names:
        if name and name not in SUPPORTED_APPS:
            SUPPORTED_APPS.append(name)
            if _HAS_PYPINYIN:
                _APPS_PINYIN[name] = lazy_pinyin(name, style=Style.NORMAL)
            added += 1
    if added:
        _log.info(f"Whitelist expanded: +{added} apps (total {len(SUPPORTED_APPS)})")


# ============================================================
# Pattern rules (regex-based)
# ============================================================
PATTERN_RULES = [
    (r"(?:打开|启动|运行|开启)\s*(\S+)", "open_app"),
    (r"(?:关闭|退出|结束|杀掉|关掉)\s*(\S+)", "close_app"),
    (r"(?:帮我搜(?:一下)?|帮我查(?:一下)?|搜索(?:一下)?|搜一下|查一下|百度一下|谷歌一下|网上搜)\s*(.+)", "search_web"),
    (r"(?:输入|打字|帮我打|键入|写入)\s*(.+)", "type_text"),
    (r"音量[调设]\s*(?:到|为)?\s*(\d+)", "set_volume"),
    (r"(?:现在几点|几点了|什么时间|当前时间)", "get_time"),
    (r"(?:今天.*日期|几月几号|什么日期|今天.*号)", "get_date"),
    (r"(?:暂停|停止|暂停播放|停一下|暂停音乐)", "media_pause"),
    (r"(?:播放|继续播放|开始播放|放歌|播放音乐)", "media_play"),
    (r"(?:下一首|下一个|切歌|下一首歌|换一首)", "media_next"),
    (r"(?:上一首|上一个|前一首|上一首歌)", "media_prev"),
    (r"(?:新建|创建)\s*(?:一个)?\s*文件夹\s*(.*)", "new_folder_named"),
    (r"(?:删除|移除)\s*(\S+)", "delete_file"),
    (r"(?:关机|关闭电脑)", "shutdown"),
    (r"(?:重启|重新启动)", "restart"),
    (r"(?:注销|切换用户)", "logout"),
    (r"(?:休眠|睡眠)", "sleep"),
    (r"(?:清理|清空)\s*(?:一下)?\s*(?:垃圾|缓存|临时文件)", "clean_temp"),
]

# Pinyin map for known command phrases
_PINYIN_MAP = {
    "打开记事本": "open_notepad", "打开浏览器": "open_browser",
    "打开计算器": "open_calculator", "打开文件管理器": "open_explorer",
    "打开资源管理器": "open_explorer", "打开命令行": "open_cmd",
    "打开终端": "open_cmd", "打开画图": "open_paint",
    "打开任务管理器": "open_task_manager", "打开设置": "open_settings",
    "打开微信": "open_wechat", "打开QQ": "open_qq",
    "打开网易云音乐": "open_app:网易云音乐",
    "打开酷狗音乐": "open_app:酷狗音乐",
    "打开QQ音乐": "open_app:QQ音乐",
    "打开钉钉": "open_app:钉钉",
    "打开腾讯会议": "open_app:腾讯会议",
    "打开Word": "open_word", "打开Excel": "open_excel",
    "打开PPT": "open_ppt", "打开文档": "open_word",
    "打开表格": "open_excel", "打开演示": "open_ppt",
    "音量调大": "volume_up", "音量调小": "volume_down",
    "增大音量": "volume_up", "减小音量": "volume_down",
    "声音大一点": "volume_up", "声音小一点": "volume_down",
    "静音": "mute", "取消静音": "unmute",
    "截屏": "screenshot", "截图": "screenshot",
    "锁屏": "lock_screen", "关闭窗口": "close_window",
    "最大化窗口": "maximize_window", "最小化窗口": "minimize_window",
    "切换窗口": "open_taskbar", "新建文件夹": "new_folder",
    "清空回收站": "empty_recycle", "打开百度": "open_browser",
    "播放音乐": "media_play", "暂停播放": "media_pause",
    "下一首歌": "media_next", "上一首歌": "media_prev",
    "现在几点了": "get_time", "今天几月几号": "get_date",
}


def _to_pinyin_list(text):
    """Convert text to list of pinyin syllables."""
    if _HAS_PYPINYIN:
        return [p.lower() for p in lazy_pinyin(text, style=Style.NORMAL) if p.strip()]
    return [ch.lower() for ch in text if ch.isalnum()]


def _to_pinyin_str(text):
    """Convert text to space-separated pinyin string."""
    return " ".join(_to_pinyin_list(text))


def _sliding_window_pinyin_match(input_text, target_phrase, threshold=0.8):
    """Sliding window pinyin matching with strict validation.

    Finds a contiguous sub-sequence in the input's pinyin that closely
    matches the target phrase's pinyin. Handles extra garbage chars
    before/after the actual command.

    Anti-false-positive rules:
    - Short targets (<=3 syllables) need ratio >= 0.9
    - Matched window must have at least 2/3 of target syllables matching
    - Reject if input is just a common prefix like "打开" with unrelated suffix

    Returns: (matched, similarity) tuple.
    """
    input_py = _to_pinyin_list(input_text)
    target_py = _to_pinyin_list(target_phrase)

    if not input_py or not target_py:
        return False, 0.0

    n = len(input_py)
    m = len(target_py)

    # Adaptive threshold: stricter for short targets
    effective_threshold = threshold
    if m <= 3:
        effective_threshold = max(threshold, 0.90)
    elif m <= 4:
        effective_threshold = max(threshold, 0.85)

    # If input is shorter than target, compare directly
    if n <= m:
        ratio = difflib.SequenceMatcher(None, input_py, target_py).ratio()
        return ratio >= effective_threshold, ratio

    best_ratio = 0.0
    # Try window sizes from m-1 to m+2
    for win_size in range(max(1, m - 1), min(n, m + 3)):
        for start in range(0, n - win_size + 1):
            window = input_py[start:start + win_size]
            ratio = difflib.SequenceMatcher(None, window, target_py).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
            if best_ratio >= effective_threshold:
                return True, best_ratio

    return False, best_ratio


def _validate_app_name(extracted_name):
    """Validate extracted app name against SUPPORTED_APPS whitelist.

    Uses pinyin similarity with adaptive threshold based on name length:
      - 2-char names (e.g. "豆包"): require >= 0.85 (strict, avoid false positives)
      - 3-char names: require >= 0.75
      - 4+ char names (e.g. "网易云音乐"): allow >= 0.60 (lenient, Whisper often errs on long names)

    Returns: (standard_name, similarity) or (None, 0.0) if no match.
    """
    if not extracted_name:
        return None, 0.0

    # Fast path: substring match (handles Whisper truncation like "网易" -> "网易云音乐")
    for app_name in SUPPORTED_APPS:
        if extracted_name in app_name and len(extracted_name) >= 2:
            _log.info(f"Substring match: '{extracted_name}' -> '{app_name}'")
            return app_name, 1.0
        if app_name in extracted_name and len(app_name) >= 2:
            _log.info(f"Reverse substring match: '{extracted_name}' contains '{app_name}'")
            return app_name, 1.0

    if not _HAS_PYPINYIN:
        return None, 0.0

    input_py = _to_pinyin_str(extracted_name)
    best_app = None
    best_ratio = 0.0

    for app_name, app_py_list in _APPS_PINYIN.items():
        app_py = " ".join(app_py_list)
        ratio = difflib.SequenceMatcher(None, input_py, app_py).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_app = app_name

    if not best_app:
        return None, 0.0

    # Adaptive threshold based on target app name length (in characters)
    name_len = len(best_app)
    if name_len <= 2:
        threshold = 0.85
    elif name_len <= 3:
        threshold = 0.75
    else:
        threshold = 0.60

    if best_ratio >= threshold:
        _log.info(f"App validated: '{extracted_name}' -> '{best_app}' (ratio={best_ratio:.2f}, thr={threshold})")
        return best_app, best_ratio

    # Fuzzy fallback: try difflib.get_close_matches on raw text (handles Whisper typos like "鸡死本" -> "记事本")
    close = difflib.get_close_matches(extracted_name, SUPPORTED_APPS, n=1, cutoff=0.6)
    if close:
        _log.info(f"Fuzzy fallback: '{extracted_name}' -> '{close[0]}'")
        return close[0], best_ratio

    _log.info(f"App rejected: '{extracted_name}' best='{best_app}' (ratio={best_ratio:.2f} < {threshold})")
    return None, best_ratio


class CommandParser:
    def __init__(self, use_nlu=False):
        self.keyword_map = COMMAND_MAP
        self.keyword_intent = KEYWORD_INTENT_MAP
        self.nlu = None
        self.use_nlu = use_nlu
        if use_nlu:
            try:
                from nlu.intent_classifier import IntentClassifier
                self.nlu = IntentClassifier(lazy=False)
                _log.info("NLU loaded")
            except Exception as e:
                _log.warning(f"NLU failed: {e}")
                self.use_nlu = False
        _log.info(f"Cmd parser ready (pypinyin={_HAS_PYPINYIN}, apps={len(SUPPORTED_APPS)})")

    def parse(self, text):
        if not text or not text.strip():
            return None
        text = text.strip()

        # 1. Exact keyword match
        result = self._exact_match(text)
        if result:
            return result

        # 2. Pattern match (with app whitelist validation)
        result = self._pattern_match(text)
        if result:
            return result

        # 3. Sliding window pinyin match
        result = self._sliding_pinyin_match(text)
        if result:
            return result

        # 4. NLU model
        if self.use_nlu and self.nlu is not None and self.nlu.ready:
            intent, confidence = self.nlu.predict(text)
            if intent != "unknown":
                _log.info(f"NLU: '{text}' -> {intent} ({confidence:.3f})")
                return intent

        # 5. Fuzzy keyword match
        result = self._fuzzy_match(text)
        if result:
            return result

        # 6. Semantic match
        result = self._semantic_match(text)
        if result:
            return result

        _log.warning(f"No match: '{text}'")
        return None

    def _exact_match(self, text):
        best_match = None
        best_len = 0
        for keyword, cmd in self.keyword_map.items():
            if keyword in text and len(keyword) > best_len:
                best_match = cmd
                best_len = len(keyword)
        if best_match:
            _log.info(f"Exact: '{text}' -> {best_match}")
            return best_match
        return None

    def _pattern_match(self, text):
        for rule in PATTERN_RULES:
            if isinstance(rule, tuple):
                pattern, intent = rule
            else:
                continue
            m = re.search(pattern, text)
            if m:
                param = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else None

                # For open_app: validate against whitelist
                if intent == "open_app" and param:
                    standard_name, sim = _validate_app_name(param)
                    if standard_name:
                        # Check if it maps to a known command
                        known_cmd = self.keyword_map.get(f"打开{standard_name}")
                        if known_cmd:
                            _log.info(f"Pattern+Whitelist: '{text}' -> {known_cmd}")
                            return known_cmd
                        result = f"open_app:{standard_name}"
                        _log.info(f"Pattern+Whitelist: '{text}' -> {result}")
                        return result
                    else:
                        # App not in whitelist - reject
                        _log.info(f"Pattern rejected (app not in whitelist): '{param}' (best sim={sim:.2f})")
                        continue

                if param:
                    result = f"{intent}:{param}"
                else:
                    result = intent
                _log.info(f"Pattern: '{text}' -> {result}")
                return result
        return None

    def _sliding_pinyin_match(self, text):
        """Sliding window pinyin matching against all known command phrases."""
        if not _HAS_PYPINYIN:
            return None

        best_intent = None
        best_ratio = 0.0
        best_phrase = None

        for phrase, intent in _PINYIN_MAP.items():
            matched, ratio = _sliding_window_pinyin_match(text, phrase, threshold=0.8)
            if matched and ratio > best_ratio:
                best_ratio = ratio
                best_intent = intent
                best_phrase = phrase

        if best_intent:
            # For open_app intents, try to resolve through whitelist
            if best_intent.startswith("open_app:"):
                app_name = best_intent.split(":", 1)[1]
                result = best_intent
            else:
                result = best_intent
            _log.info(f"SlidingPinyin: '{text}' -> '{best_phrase}' ({result}) ratio={best_ratio:.2f}")
            return result

        return None

    def _fuzzy_match(self, text):
        all_keywords = {}
        all_keywords.update(self.keyword_map)
        best_match = None
        best_ratio = 0.0
        for keyword, cmd in all_keywords.items():
            ratio = difflib.SequenceMatcher(None, keyword, text).ratio()
            if ratio > best_ratio and ratio > 0.6:
                best_match = cmd
                best_ratio = ratio
        if best_match:
            _log.info(f"Fuzzy: '{text}' -> {best_match} ({best_ratio:.2f})")
            return best_match
        return None

    def _semantic_match(self, text):
        has_volume_keyword = False
        has_direction_keyword = False
        direction_intent = None
        for kw, intent in self.keyword_intent.items():
            if kw in text:
                if intent is None:
                    has_volume_keyword = True
                elif kw in ("\u5927", "\u9ad8", "\u589e"):
                    has_direction_keyword = True
                    direction_intent = intent
                elif kw in ("\u5c0f", "\u4f4e", "\u51cf"):
                    has_direction_keyword = True
                    direction_intent = intent
                else:
                    if intent:
                        _log.info(f"Semantic: '{text}' -> {intent}")
                        return intent
        if has_volume_keyword and has_direction_keyword and direction_intent:
            _log.info(f"Semantic combo: '{text}' -> {direction_intent}")
            return direction_intent
        return None
