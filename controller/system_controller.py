import subprocess
import os
import ctypes
import threading
from global_config import SAMPLE_RATE


class SystemController:
    def __init__(self):
        self.volume_interface = None
        self._init_volume()
        self._last_cmd = None

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

    def _open_notepad(self):
        subprocess.Popen(["notepad.exe"])
        return "已打开记事本"

    def _open_browser(self):
        os.startfile("https://www.baidu.com")
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
            img.save(save_path)
            return f"截图已保存: {save_path}"
        except Exception as e:
            return f"截图失败: {e}"

    def _lock_screen(self):
        ctypes.windll.user32.LockWorkStation()
        return "已锁屏"

    def _close_window(self):
        try:
            import pyautogui
            pyautogui.hotkey("alt", "f4")
            return "已关闭当前窗口"
        except Exception as e:
            return f"关闭窗口失败: {e}"

    def _open_task_manager(self):
        subprocess.Popen(["taskmgr.exe"])
        return "已打开任务管理器"

    def _open_settings(self):
        subprocess.Popen(["ms-settings:"])
        return "已打开系统设置"

    def _open_cmd(self):
        subprocess.Popen(["cmd.exe"])
        return "已打开命令行"
