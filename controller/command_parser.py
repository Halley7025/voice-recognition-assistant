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


# ============================================================
# PinyinTrie: forward-maximum-match pinyin error correction
# ============================================================
class PinyinTrie:
    """Trie tree for pinyin-based app name correction.

    Each node stores children keyed by pinyin syllable.
    Leaf / intermediate nodes with is_terminal=True store the
    canonical Chinese app name so that Whisper fragmented output
    (e.g. 'wang yi yun yin yue' -> '网易云音乐') can be recovered.
    """

    __slots__ = ("children", "is_terminal", "canonical")

    def __init__(self):
        self.children: dict = {}
        self.is_terminal: bool = False
        self.canonical: str = ""

    def insert(self, pinyin_list, canonical_name):
        node = self
        for py in pinyin_list:
            key = py.lower()
            if key not in node.children:
                node.children[key] = PinyinTrie()
            node = node.children[key]
        node.is_terminal = True
        node.canonical = canonical_name

    def search_forward_max(self, pinyin_list, start_idx):
        """Greedy forward maximum match starting at start_idx.

        Returns (end_idx, canonical_name) of the longest match found,
        or (None, None) if no match at all.
        """
        node = self
        best_end = None
        best_canonical = None
        for i in range(start_idx, len(pinyin_list)):
            key = pinyin_list[i].lower()
            if key not in node.children:
                break
            node = node.children[key]
            if node.is_terminal:
                best_end = i + 1
                best_canonical = node.canonical
        return best_end, best_canonical


# Global trie instance, rebuilt on inject_apps()
_global_trie = None


def _build_trie():
    """Build (or rebuild) the global PinyinTrie from SUPPORTED_APPS."""
    global _global_trie
    if not _HAS_PYPINYIN:
        _global_trie = None
        return
    trie = PinyinTrie()
    for app_name in SUPPORTED_APPS:
        py_list = lazy_pinyin(app_name, style=Style.NORMAL)
        if py_list:
            trie.insert(py_list, app_name)
    _global_trie = trie
    _log.info(f"PinyinTrie built: {len(SUPPORTED_APPS)} entries")


# Build once at module load
_build_trie()


# Global app path map (populated by SystemController on init)
APP_PATH_MAP = {}


def inject_apps(app_names, path_map=None):
    """Dynamically add app names to the whitelist, rebuild pinyin index and trie.

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
        _build_trie()  # rebuild trie after whitelist changes


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


# ============================================================
# Verb prefix regex constants for app name body extraction
# ============================================================

# Narrow mode: matches only verb prefixes, used for fast separation in Trie correction
_VERB_NARROW_RE = re.compile(
    r"^(\u6253\u5f00|\u542f\u52a8|\u8fd0\u884c|\u5f00\u542f|\u5173\u95ed|\u9000\u51fa|\u641c\u7d22|\u64ad\u653e|\u6682\u505c|\u505c\u6b62)\s*"
)

# Wide mode: matches verb phrases with polite prefixes, for fallback app name extraction
# Handles variants like "帮我打开", "请帮我打开", "我要打开", "给咱打开" etc.
_VERB_WIDE_RE = re.compile(
    r"^(?:\u8bf7\s*)?(?:\u5e2e\u6211|\u6211(?:\u8981|\u60f3)|\u54b1(?:\u4eec)?|\u7ed9\u54b1?)?\s*"
    r"(\u6253\u5f00|\u542f\u52a8|\u8fd0\u884c|\u5f00\u542f|\u5173\u95ed|\u9000\u51fa|\u641c\u7d22|\u64ad\u653e|\u6682\u505c|\u505c\u6b62)\s*"
)


def _extract_app_body(text):
    if not text:
        return text, ""
    m = _VERB_WIDE_RE.match(text)
    if m:
        verb = m.group(0).strip()
        body = text[m.end():].strip()
        if body:
            return body, verb
    return text, ""


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



def _syllable_edit_distance(py_a, py_b):
    """计算两个拼音音节列表之间的编辑距离（Levenshtein Distance）。

    注意：此函数在**音节级别**（而非字符串字符级别）操作。
    例如 ['wang','yi','yun'] 与 ['wang','yi','lin'] 的距离为 1（只需将 'lin' 改为 'yun'），
    而非按字母逐位计算。

    使用滚动数组优化空间复杂度为 O(min(m,n))。
    """
    m, n = len(py_a), len(py_b)
    # 让较长的序列作为行，减少内存
    if m < n:
        return _syllable_edit_distance(py_b, py_a)
    # dp[j] = py_a[:i] 与 py_b[:j] 的编辑距离
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if py_a[i - 1].lower() == py_b[j - 1].lower():
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


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

    # ------------------------------------------------------------------
    # Trie-based pinyin correction (runs BEFORE all other parse strategies)
    # ------------------------------------------------------------------
    def _trie_correct(self, text):
        """Use PinyinTrie forward-maximum-match to replace fragmented
        Whisper output with canonical app names.

        Example: "拔开网意运因乐" -> pinyin scan finds "wang yi yun yin yue"
        matches trie node for "网易云音乐" -> replace and return
        "打开网易云音乐".

        Returns corrected text, or original text if no improvement found.
        """
        if not _HAS_PYPINYIN or _global_trie is None:
            return text

        # Separate leading verb from trailing content
        verb_match = _VERB_NARROW_RE.match(text)
        verb = ""
        body = text
        if verb_match:
            verb = verb_match.group(0).strip()
            body = text[verb_match.end():].strip()

        if not body:
            return text

        # Build (char, pinyin) pairs for index-safe fallback
        pairs = []
        for ch in body:
            if ch.strip():
                py = lazy_pinyin(ch, style=Style.NORMAL)
                if py:
                    pairs.append((ch, py[0].lower()))

        if len(pairs) < 2:
            return text

        body_py = [p[1] for p in pairs]

        # Forward maximum match scan through trie
        corrected_parts = []
        i = 0
        matched_any = False
        while i < len(pairs):
            end_idx, canonical = _global_trie.search_forward_max(body_py, i)
            if canonical and (end_idx - i) >= 2:
                corrected_parts.append(canonical)
                matched_any = True
                i = end_idx
            else:
                # Keep original character at this position (index-safe)
                corrected_parts.append(pairs[i][0])
                i += 1

        if matched_any:
            corrected_body = "".join(corrected_parts)
            result = f"{verb}{corrected_body}" if verb else corrected_body
            if result != text:
                _log.info(f"Trie correct: '{text}' -> '{result}'")
            return result

        # Trie 匹配失败，进入拼音音节级编辑距离兜底
        return self._pinyin_fallback_correct(text, body, verb, pairs)

    def _pinyin_fallback_correct(self, original_text, body, verb, pairs):
        """Pinyin syllable-level edit distance fallback correction.

        When PinyinTrie forward-maximum-match fails to hit any whitelisted app,
        this method iterates all known app names, computing syllable-level
        Levenshtein distance between the input and each standard app name,
        then selects the best match within a dynamic threshold.

        Core algorithm design (syllable-level edit distance):
          This edit distance operates at the **syllable level** (not character level).
          For example, ['wang','yi','yun'] vs ['wang','yi','lin'] has distance 1
          (just change 'lin' to 'yun'), NOT computed letter-by-letter.
          This avoids slice alignment issues when pinyin syllable counts differ
          from the original character count -- we always compare at the atomic
          syllable unit, regardless of how many letters each Chinese character maps to.

        Dynamic threshold rules:
          App name length <= 3 chars: max allowed edit distance = 1
          App name length >= 4 chars: max allowed edit distance = 2

        Tie-breaking: when edit distances are equal, prefer the app name whose
        character length is closest to the input.

        Args:
            original_text: full original text (including verb prefix)
            body: app name body after verb prefix removal
            verb: extracted verb prefix (e.g. "打开")
            pairs: (char, pinyin) pair list for index-safe alignment
        """
        # ------------------------------------------------------------------
        # Step 1: Ensure correct app name body extraction
        # ------------------------------------------------------------------
        # If the narrow-mode regex failed to capture the verb prefix
        # (e.g. "帮我打开网易云音了" where "帮我打开" was not stripped),
        # use the wide-mode regex to re-extract, preventing verb/polite
        # prefixes from inflating the edit distance.
        if not verb:
            extracted_body, extracted_verb = _extract_app_body(original_text)
            if extracted_verb and extracted_body:
                body = extracted_body
                verb = extracted_verb
                # Rebuild (char, pinyin) pairs from the new body
                # Process character-by-character to maintain strict alignment
                # with the original text (not by splitting pinyin strings)
                pairs = []
                for ch in body:
                    if ch.strip():
                        py = lazy_pinyin(ch, style=Style.NORMAL)
                        if py:
                            pairs.append((ch, py[0].lower()))

        if not pairs or not _APPS_PINYIN:
            return original_text

        input_py = [p[1] for p in pairs]
        input_char_len = len(pairs)

        # Only enable pinyin edit distance fallback for Chinese input.
        # Pure English input (like "Word") should not match Chinese app names via pinyin.
        has_chinese_input = any("\u4e00" <= p[0] <= "\u9fff" for p in pairs)
        if not has_chinese_input:
            return original_text

        # ------------------------------------------------------------------
        # Step 2: Iterate whitelisted apps, compute syllable edit distance
        # ------------------------------------------------------------------
        best_app = None
        best_dist = float("inf")
        best_len_diff = float("inf")

        for app_name, app_py in _APPS_PINYIN.items():
            # Skip non-Chinese app names (like Word, Chrome, Excel)
            # whose short pinyin representations easily cause false matches
            has_chinese_target = any("\u4e00" <= ch <= "\u9fff" for ch in app_name)
            if not has_chinese_target:
                continue

            # Dynamic threshold: based on target app name character count
            app_char_len = len(app_name)
            max_allowed = 2 if app_char_len >= 4 else 1

            dist = _syllable_edit_distance(input_py, app_py)

            if dist > max_allowed:
                continue

            # Select minimum edit distance; on tie, pick closest character length
            len_diff = abs(input_char_len - app_char_len)
            if dist < best_dist or (dist == best_dist and len_diff < best_len_diff):
                best_dist = dist
                best_app = app_name
                best_len_diff = len_diff

        # ------------------------------------------------------------------
        # Step 3: Replace on match, otherwise return original text
        # ------------------------------------------------------------------
        if best_app is not None:
            result = f"{verb}{best_app}" if verb else best_app
            _log.info(
                f"PinyinEdit fallback: '{original_text}' -> '{result}' "
                f"(dist={best_dist}, body='{body}' -> '{best_app}')"
            )
            return result

        return original_text


    def parse(self, text):
        if not text or not text.strip():
            return None
        text = text.strip()

        # 0. Trie-based pinyin correction (pre-processing)
        text = self._trie_correct(text)

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
                elif kw in ("大", "高", "增"):
                    has_direction_keyword = True
                    direction_intent = intent
                elif kw in ("小", "低", "减"):
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

