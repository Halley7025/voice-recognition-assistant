# -*- coding: utf-8 -*-
import random
import logging

logger = logging.getLogger(__name__)


class DynamicAudioSampler:
    CHARS_PER_SECOND = 3.5
    BUFFER_SECONDS = 1.0
    MIN_DURATION = 3.0
    MAX_DURATION = 5.0

    PROMPT_CORPUS = {
        "short": [
            "立即锁定屏幕",
            "打开计算器",
            "截个屏",
            "音量调大",
            "关闭窗口",
        ],
        "medium": [
            "打开计算器并新建文本",
            "请帮我打开网易云音乐",
            "打开浏览器搜索天气",
            "请把音量调到最大",
            "帮我打开任务管理器",
        ],
        "long": [
            "进入护眼模式并将音量调低",
            "帮我打开记事本并查询明天天气",
            "请打开设置并切换到深色模式",
            "打开文件管理器并新建一个文件夹",
        ],
    }

    def __init__(self, samples_per_enroll=3):
        self.samples_per_enroll = samples_per_enroll
        self._used_prompts = []

    @classmethod
    def calc_duration(cls, text):
        cleaned = text.replace(" ", "").replace("。", "").replace("，", "")
        chars = len(cleaned)
        raw = chars / cls.CHARS_PER_SECOND + cls.BUFFER_SECONDS
        return max(cls.MIN_DURATION, min(cls.MAX_DURATION, raw))

    def get_prompts(self, count=None):
        if count is None:
            count = self.samples_per_enroll
        all_prompts = []
        for category, prompts in self.PROMPT_CORPUS.items():
            for p in prompts:
                all_prompts.append(p)
        available = [p for p in all_prompts if p not in self._used_prompts]
        if len(available) < count:
            available = all_prompts[:]
            self._used_prompts.clear()
        selected = random.sample(available, min(count, len(available)))
        self._used_prompts.extend(selected)
        result = []
        for text in selected:
            dur = self.calc_duration(text)
            result.append({"text": text, "duration": round(dur, 1)})
            logger.info(
                "[DynamicSampler] prompt=\"%s\" chars=%d duration=%.1fs",
                text, len(text), dur
            )
        return result

    def get_single_prompt(self):
        return self.get_prompts(count=1)[0]

    def format_countdown(self, duration):
        if duration == int(duration):
            return "%d秒" % int(duration)
        return "%.1f秒" % duration
