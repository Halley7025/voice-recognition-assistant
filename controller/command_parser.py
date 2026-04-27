COMMAND_MAP = {
    "打开记事本": "open_notepad",
    "打开浏览器": "open_browser",
    "音量调大": "volume_up",
    "音量调小": "volume_down"
}

class CommandParser:
    def parse(self, text):
        for k, v in COMMAND_MAP.items():
            if k in text:
                return v
        return None