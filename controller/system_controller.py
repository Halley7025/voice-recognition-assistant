import subprocess
import os
import ctypes
import threading
import webbrowser


class SystemController:
    def __init__(self):
        self.volume_interface = None
        self._last_cmd = None
        self._init_volume()

    def _init_volume(self):
        try:
            from pycaw.pycaw import AudioUtilities
            devices = AudioUtilities.GetSpeakers()
            self.volume_interface = devices.EndpointVolume
            print("系统音量控制初始化成功")
        except Exception as e:
            print(f"音量控制初始化失败: {e}")

    def run(self, cmd):
        self._last_cmd = cmd
        handlers = {
            "open_notepad": self._open_notepad,
            "open_browser": self._open_browser,
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "open_calculator": self._open_calculator,
            "open_explorer": self._open_explorer,
            "screenshot": self._screenshot,
            "lock_screen": self._lock_screen,
            "close_window": self._close_window,
            "open_task_manager": self._open_task_manager,
            "open_settings": self._open_settings,
            "open_cmd": self._open_cmd,
            "open_paint": self._open_paint,
            "open_word": self._open_word,
            "open_excel": self._open_excel,
            "open_ppt": self._open_ppt,
            "search_web": self._search_web,
            "open_wechat": self._open_wechat,
            "open_qq": self._open_qq,
            "mute": self._mute,
            "unmute": self._unmute,
            "maximize_window": self._maximize_window,
            "minimize_window": self._minimize_window,
            "new_folder": self._new_folder,
            "empty_recycle": self._empty_recycle,
            "open_taskbar": self._switch_window,
        }
        handler = handlers.get(cmd)
        if handler:
            try:
                result = handler()
                print(f"指令执行成功: {cmd}")
                return True, result or cmd
            except Exception as e:
                print(f"指令执行失败 [{cmd}]: {e}")
                return False, str(e)
        else:
            print(f"未知指令: {cmd}")
            return False, f"未知指令: {cmd}"

    def run_async(self, cmd):
        thread = threading.Thread(target=self.run, args=(cmd,), daemon=True)
        thread.start()
        return thread

    def get_all_commands(self):
        return {
            "open_notepad": "打开记事本",
            "open_browser": "打开浏览器",
            "volume_up": "音量增大",
            "volume_down": "音量减小",
            "open_calculator": "打开计算器",
            "open_explorer": "打开文件管理器",
            "screenshot": "截屏",
            "lock_screen": "锁屏",
            "close_window": "关闭窗口",
            "open_task_manager": "任务管理器",
            "open_settings": "系统设置",
            "open_cmd": "命令行",
            "open_paint": "画图工具",
            "open_word": "打开Word",
            "open_excel": "打开Excel",
            "open_ppt": "打开PPT",
            "search_web": "网页搜索",
            "open_wechat": "打开微信",
            "open_qq": "打开QQ",
            "mute": "静音",
            "unmute": "取消静音",
            "maximize_window": "最大化窗口",
            "minimize_window": "最小化窗口",
        }

    def _open_notepad(self):
        subprocess.Popen(["notepad.exe"])
        return "已打开记事本"

    def _open_browser(self):
        webbrowser.open("https://www.baidu.com")
        return "已打开浏览器"

    def _volume_up(self):
        if self.volume_interface:
            current = self.volume_interface.GetMasterVolumeLevelScalar()
            new_vol = min(current + 0.1, 1.0)
            self.volume_interface.SetMasterVolumeLevelScalar(new_vol, None)
            return f"音量已增大至 {int(new_vol * 100)}%"
        return "音量控制不可用"

    def _volume_down(self):
        if self.volume_interface:
            current = self.volume_interface.GetMasterVolumeLevelScalar()
            new_vol = max(current - 0.1, 0.0)
            self.volume_interface.SetMasterVolumeLevelScalar(new_vol, None)
            return f"音量已减小至 {int(new_vol * 100)}%"
        return "音量控制不可用"

    def _open_calculator(self):
        subprocess.Popen(["calc.exe"])
        return "已打开计算器"

    def _open_explorer(self):
        subprocess.Popen(["explorer.exe"])
        return "已打开文件管理器"

    def _screenshot(self):
        try:
            import pyautogui
            img = pyautogui.screenshot()
            save_path = os.path.join(os.path.dirname(__file__), "..", "data", "screenshot.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            return f"截图已保存"
        except Exception as e:
            return f"截图失败: {e}"

    def _lock_screen(self):
        ctypes.windll.user32.LockWorkStation()
        return "已锁屏"

    def _close_window(self):
        import pyautogui
        pyautogui.hotkey("alt", "f4")
        return "已关闭当前窗口"

    def _open_task_manager(self):
        subprocess.Popen(["taskmgr.exe"])
        return "已打开任务管理器"

    def _open_settings(self):
        subprocess.Popen(["ms-settings:"])
        return "已打开系统设置"

    def _open_cmd(self):
        subprocess.Popen(["cmd.exe"])
        return "已打开命令行"

    def _open_paint(self):
        subprocess.Popen(["mspaint.exe"])
        return "已打开画图工具"

    def _open_word(self):
        try:
            subprocess.Popen(["start", "winword"], shell=True)
            return "正在打开Word"
        except Exception:
            return "未找到Word"

    def _open_excel(self):
        try:
            subprocess.Popen(["start", "excel"], shell=True)
            return "正在打开Excel"
        except Exception:
            return "未找到Excel"

    def _open_ppt(self):
        try:
            subprocess.Popen(["start", "powerpnt"], shell=True)
            return "正在打开PowerPoint"
        except Exception:
            return "未找到PowerPoint"

    def _search_web(self):
        webbrowser.open("https://www.baidu.com")
        return "已打开网页搜索"

    def _open_wechat(self):
        paths = [
            os.path.expandvars(r"%ProgramFiles%\Tencent\WeChat\WeChat.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Tencent\WeChat\WeChat.exe"),
        ]
        for p in paths:
            if os.path.exists(p):
                subprocess.Popen([p])
                return "已打开微信"
        return "未找到微信"

    def _open_qq(self):
        paths = [
            os.path.expandvars(r"%ProgramFiles%\Tencent\QQ\Bin\QQScLauncher.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Tencent\QQ\Bin\QQScLauncher.exe"),
        ]
        for p in paths:
            if os.path.exists(p):
                subprocess.Popen([p])
                return "已打开QQ"
        return "未找到QQ"

    def _mute(self):
        if self.volume_interface:
            self.volume_interface.SetMute(1, None)
            return "已静音"
        return "音量控制不可用"

    def _unmute(self):
        if self.volume_interface:
            self.volume_interface.SetMute(0, None)
            return "已取消静音"
        return "音量控制不可用"

    def _maximize_window(self):
        import pyautogui
        pyautogui.hotkey("win", "up")
        return "已最大化窗口"

    def _minimize_window(self):
        import pyautogui
        pyautogui.hotkey("win", "down")
        return "已最小化窗口"

    def _new_folder(self):
        import pyautogui
        pyautogui.hotkey("ctrl", "shift", "n")
        return "已创建新文件夹"

    def _empty_recycle(self):
        try:
            ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 7)
            return "已清空回收站"
        except Exception:
            return "清空回收站失败"

    def _switch_window(self):
        import pyautogui
        pyautogui.hotkey("alt", "tab")
        return "已切换窗口"
