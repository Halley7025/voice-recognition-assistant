import re
import subprocess
import os
import ctypes
import threading
import time
import webbrowser
from logger_config import setup_logger
_log = setup_logger(__name__)


class SystemController:
    def __init__(self):
        self.volume_interface = None
        self._last_cmd = None
        self._init_volume()
        # Build app_path_map by scanning desktop / Start Menu shortcuts
        self.app_path_map = self._scan_installed_apps()
        # Inject app names into command parser whitelist
        try:
            from controller.command_parser import inject_apps
            inject_apps(list(self.app_path_map.keys()), self.app_path_map)
        except Exception as e:
            _log.warning(f"Shortcut scan inject failed: {e}")

    def _init_volume(self):
        try:
            from pycaw.pycaw import AudioUtilities
            devices = AudioUtilities.GetSpeakers()
            self.volume_interface = devices.EndpointVolume
            _log.info("系统音量控制初始化成功")
        except Exception as e:
            _log.warning(f"音量控制初始化失败: {e}")

    def run(self, cmd: str):
        self._last_cmd = cmd
        # Handle commands with parameters (format: "intent:param")
        if ":" in cmd:
            intent, param = cmd.split(":", 1)
            param_handlers = {
                "open_app": lambda p: self._open_app(p),
                "close_app": lambda p: self._close_app(p),
                "search_web": lambda p: self._search_web_query(p),
                "type_text": lambda p: self._type_text(p),
                "set_volume": lambda p: self._set_volume(int(p)),
                "new_folder_named": lambda p: self._new_folder_named(p),
                "delete_file": lambda p: self._delete_file(p),
                "open_file": lambda p: self._open_file(p),
            }
            handler = param_handlers.get(intent)
            if handler:
                try:
                    result = handler(param)
                    _log.info(f"指令执行成功: {cmd}")
                    return True, result
                except Exception as e:
                    _log.error(f"指令执行失败 [{cmd}]: {e}")
                    return False, str(e)

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
            # New commands
            "get_time": self._get_time,
            "get_date": self._get_date,
            "media_play": self._media_play,
            "media_pause": self._media_pause,
            "media_next": self._media_next,
            "media_prev": self._media_prev,
            "shutdown": self._shutdown,
            "restart": self._restart,
            "sleep": self._sleep,
            "logout": self._logout,
            "clean_temp": self._clean_temp,
        }
        handler = handlers.get(cmd)
        if handler:
            try:
                result = handler()
                _log.info(f"指令执行成功: {cmd}")
                return True, result or cmd
            except Exception as e:
                _log.error(f"指令执行失败 [{cmd}]: {e}")
                return False, str(e)
        else:
            _log.warning(f"未知指令: {cmd}")
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
            "get_time": "当前时间",
            "get_date": "今天日期",
            "media_play": "播放",
            "media_pause": "暂停",
            "media_next": "下一首",
            "media_prev": "上一首",
        }

    # === Original handlers ===

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
            return "截图已保存"
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

    # === New handlers ===

    def _find_and_open_shortcut(self, app_name):
        """Scan desktop and Start Menu shortcuts to find and open an app.

        Searches four standard Windows shortcut directories (and subdirectories)
        for .lnk files whose filename contains app_name (case-insensitive).
        If found, opens the shortcut directly with os.startfile().

        Returns:
            (True, matched_filename) if found and opened, (False, None) otherwise.
        """
        search_dirs = [
            os.path.expanduser("~\\Desktop"),
            os.path.join(os.environ.get("PUBLIC", ""), "Desktop"),
            os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs"),
            os.path.join(os.environ.get("PROGRAMDATA", ""), "Microsoft\\Windows\\Start Menu\\Programs"),
        ]
        app_name_lower = app_name.lower()
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.lower().endswith(".lnk"):
                        continue
                    if app_name_lower in fname.lower():
                        shortcut_path = os.path.join(root, fname)
                        try:
                            os.startfile(shortcut_path)
                            _log.info(f"Shortcut opened: {shortcut_path}")
                            return True, fname
                        except Exception as e:
                            _log.warning(f"startfile failed for {shortcut_path}: {e}")
        return False, None

    def _scan_installed_apps(self):
        """Scan desktop and Start Menu shortcuts, return {app_name: shortcut_path} dict.

        Walks four standard Windows directories, extracts .lnk filenames as app names,
        and stores the shortcut path so we can os.startfile() it directly.
        """
        search_dirs = [
            os.path.expanduser("~\\Desktop"),
            os.path.join(os.environ.get("PUBLIC", ""), "Desktop"),
            os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs"),
            os.path.join(os.environ.get("PROGRAMDATA", ""), "Microsoft\\Windows\\Start Menu\\Programs"),
        ]
        app_map = {}
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.lower().endswith(".lnk"):
                        continue
                    name = fname[:-4].strip()
                    name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
                    if name and len(name) >= 2 and name not in app_map:
                        app_map[name] = os.path.join(root, fname)
        _log.info(f"Scanned {len(app_map)} shortcuts from desktop/Start Menu")
        return app_map

    def _open_app(self, name):
        """Open any application by name.

        Strategy:
            1. Check built-in Windows tools (fast path)
            2. Look up self.app_path_map (scanned from shortcuts)
            3. Fuzzy match against app_path_map keys
            4. Try direct execution as last resort
        """
        import difflib

        # 1. Built-in Windows tools
        builtin_map = {
            "记事本": "notepad.exe", "notepad": "notepad.exe",
            "计算器": "calc.exe", "calculator": "calc.exe",
            "画图": "mspaint.exe", "paint": "mspaint.exe",
            "命令行": "cmd.exe", "cmd": "cmd.exe", "终端": "cmd.exe",
            "任务管理器": "taskmgr.exe", "taskmanager": "taskmgr.exe",
            "资源管理器": "explorer.exe", "文件管理器": "explorer.exe",
            "浏览器": None, "chrome": None, "edge": None,
            "控制面板": "control.exe",
            "注册表": "regedit.exe",
            "截图工具": "SnippingTool.exe",
            "远程桌面": "mstsc.exe",
            "播放器": "wmplayer.exe",
            "录音机": "SoundRecorder.exe",
        }
        for key, exe in builtin_map.items():
            if key in name:
                if exe:
                    os.startfile(exe)
                    return f"已打开{name}"
                else:
                    webbrowser.open("https://www.baidu.com")
                    return "已打开浏览器"

        # 2. Direct lookup in app_path_map
        if name in self.app_path_map:
            try:
                os.startfile(self.app_path_map[name])
                _log.info(f"App path map hit: '{name}' -> {self.app_path_map[name]}")
                return f"已启动 {name}"
            except Exception as e:
                _log.warning(f"startfile failed for '{name}': {e}")

        # 3. Fuzzy match against app_path_map keys
        if self.app_path_map:
            matches = difflib.get_close_matches(name, self.app_path_map.keys(), n=1, cutoff=0.6)
            if matches:
                matched_name = matches[0]
                try:
                    os.startfile(self.app_path_map[matched_name])
                    _log.info(f"Fuzzy app match: '{name}' -> '{matched_name}'")
                    return f"已启动 {matched_name}"
                except Exception as e:
                    _log.warning(f"startfile failed for '{matched_name}': {e}")

        # 4. Shell fallback
        try:
            subprocess.Popen(["cmd", "/c", "start", "", name], shell=True)
            return f"正在尝试启动 {name}"
        except Exception:
            pass
        return f"未找到应用: {name}"

    def _close_app(self, name):
        """Close an application by name."""
        try:
            # Map common names to process names
            process_map = {
                "记事本": "notepad.exe", "计算器": "calc.exe",
                "画图": "mspaint.exe", "浏览器": "chrome.exe",
                "命令行": "cmd.exe", "任务管理器": "taskmgr.exe",
                "微信": "WeChat.exe", "qq": "QQ.exe",
                "word": "WINWORD.EXE", "excel": "EXCEL.EXE",
                "ppt": "POWERPNT.EXE",
            }
            proc_name = None
            for key, val in process_map.items():
                if key in name:
                    proc_name = val
                    break
            if not proc_name:
                proc_name = f"{name}.exe"
            subprocess.run(["taskkill", "/IM", proc_name, "/F"],
                           capture_output=True, timeout=5)
            return f"已关闭 {name}"
        except Exception as e:
            return f"关闭 {name} 失败: {e}"

    def _search_web_query(self, query):
        """Search the web with a specific query."""
        import urllib.parse
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}"
        webbrowser.open(url)
        return f"正在搜索: {query}"

    def _type_text(self, text):
        """Type text using keyboard simulation."""
        try:
            import pyautogui
            pyautogui.typewrite(text, interval=0.05) if text.isascii() else pyautogui.write(text)
            return f"已输入: {text}"
        except Exception as e:
            return f"输入失败: {e}"

    def _set_volume(self, level):
        """Set volume to a specific percentage (0-100)."""
        if self.volume_interface:
            level = max(0, min(100, level))
            self.volume_interface.SetMasterVolumeLevelScalar(level / 100.0, None)
            return f"音量已设为 {level}%"
        return "音量控制不可用"

    def _get_time(self):
        """Get current time."""
        current_time = time.strftime("%H:%M:%S")
        return f"现在是 {current_time}"

    def _get_date(self):
        """Get current date."""
        current_date = time.strftime("%Y年%m月%d日")
        return f"今天是 {current_date}"

    def _media_play(self):
        """Media play."""
        import pyautogui
        pyautogui.press("playpause")
        return "播放"

    def _media_pause(self):
        """Media pause."""
        import pyautogui
        pyautogui.press("playpause")
        return "已暂停"

    def _media_next(self):
        """Next track."""
        import pyautogui
        pyautogui.press("nexttrack")
        return "下一首"

    def _media_prev(self):
        """Previous track."""
        import pyautogui
        pyautogui.press("prevtrack")
        return "上一首"

    def _new_folder_named(self, name):
        """Create a new folder with a specific name."""
        import pyautogui
        pyautogui.hotkey("ctrl", "shift", "n")
        time.sleep(0.5)
        if name:
            pyautogui.typewrite(name, interval=0.05) if name.isascii() else None
            pyautogui.press("enter")
        return f"已创建文件夹: {name}"

    def _delete_file(self, name):
        """Delete selected file."""
        import pyautogui
        pyautogui.press("delete")
        return f"已删除: {name}"

    def _open_file(self, path):
        """Open a file or folder."""
        if path:
            try:
                os.startfile(path)
                return f"已打开: {path}"
            except Exception:
                pass
        subprocess.Popen(["explorer.exe"])
        return "已打开文件管理器"

    def _shutdown(self):
        """Shutdown the computer."""
        subprocess.run(["shutdown", "/s", "/t", "60"], capture_output=True)
        return "系统将在60秒后关机"

    def _restart(self):
        """Restart the computer."""
        subprocess.run(["shutdown", "/r", "/t", "60"], capture_output=True)
        return "系统将在60秒后重启"

    def _sleep(self):
        """Put the computer to sleep."""
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                       capture_output=True)
        return "系统即将休眠"

    def _logout(self):
        """Log out the current user."""
        subprocess.run(["shutdown", "/l"], capture_output=True)
        return "正在注销"

    def _clean_temp(self):
        """Clean temporary files."""
        import glob
        temp_dir = os.environ.get("TEMP", "")
        count = 0
        for f in glob.glob(os.path.join(temp_dir, "*")):
            try:
                if os.path.isfile(f):
                    os.remove(f)
                    count += 1
            except Exception:
                pass
        return f"已清理 {count} 个临时文件"

