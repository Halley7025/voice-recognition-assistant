"""Windows application finder: locate installed apps by Chinese name.

Searches Start Menu shortcuts, desktop shortcuts, and Windows Registry
to find the actual executable path for a given app name.
"""
import os
import re
import glob
import logging
import subprocess
from difflib import SequenceMatcher

try:
    import winreg
except ImportError:
    winreg = None

_log = logging.getLogger(__name__)


def _similarity(a, b):
    """String similarity ratio (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _read_lnk_target(lnk_path):
    """Read target path from a .lnk shortcut using PowerShell."""
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk_path}').TargetPath"
            ],
            capture_output=True, text=True, timeout=5, creationflags=0x08000000
        )
        target = result.stdout.strip()
        return target if target and os.path.exists(target) else None
    except Exception:
        return None


def _search_shortcuts(query):
    """Search .lnk files in Start Menu and Desktop for matching app name."""
    dirs = []
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        dirs.append(os.path.join(appdata, r"Microsoft\Windows\Start Menu\Programs"))
    progdata = os.environ.get("PROGRAMDATA", "")
    if progdata:
        dirs.append(os.path.join(progdata, r"Microsoft\Windows\Start Menu\Programs"))
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        dirs.append(os.path.join(user_profile, "Desktop"))

    best_match = None
    best_score = 0.0

    for base_dir in dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if not fname.lower().endswith(".lnk"):
                    continue
                name_no_ext = fname[:-4]
                score = _similarity(query, name_no_ext)
                if score > best_score and score >= 0.4:
                    lnk_path = os.path.join(root, fname)
                    target = _read_lnk_target(lnk_path)
                    if target:
                        best_score = score
                        best_match = (target, name_no_ext, score)

    return best_match


def _search_registry(query):
    """Search Windows Uninstall registry for installed app display names."""
    if winreg is None:
        return None

    reg_paths = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    ]
    best_match = None
    best_score = 0.0

    for reg_path in reg_paths:
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
        except OSError:
            continue
        i = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(key, i)
                i += 1
                try:
                    subkey = winreg.OpenKey(key, subkey_name)
                    display_name = ""
                    install_loc = ""
                    try:
                        display_name, _ = winreg.QueryValueEx(subkey, "DisplayName")
                    except OSError:
                        pass
                    try:
                        install_loc, _ = winreg.QueryValueEx(subkey, "InstallLocation")
                    except OSError:
                        pass
                    winreg.CloseKey(subkey)

                    if not display_name:
                        continue
                    score = _similarity(query, display_name)
                    if score > best_score and score >= 0.4:
                        exe_path = None
                        if install_loc and os.path.isdir(install_loc):
                            for f in os.listdir(install_loc):
                                if f.lower().endswith(".exe"):
                                    candidate = os.path.join(install_loc, f)
                                    exe_score = _similarity(query, f.replace(".exe", ""))
                                    if exe_score > 0.3 or f.lower().replace(".exe", "") in display_name.lower():
                                        exe_path = candidate
                                        break
                        if exe_path:
                            best_score = score
                            best_match = (exe_path, display_name, score)
                except OSError:
                    pass
            except OSError:
                break
        winreg.CloseKey(key)

    return best_match


def find_app(query):
    """Find an installed application by Chinese/English name.

    Args:
        query: App name to search (e.g., '网易云音乐', 'WeChat')

    Returns:
        (exe_path, matched_name, score) or None if not found.
    """
    # 1. Search Start Menu / Desktop shortcuts
    result = _search_shortcuts(query)
    if result and result[2] >= 0.6:
        _log.info(f"[Shortcut] '{query}' -> '{result[1]}' ({result[2]:.2f}) -> {result[0]}")
        return result

    # 2. Search Registry
    result2 = _search_registry(query)
    if result2 and result2[2] >= 0.5:
        _log.info(f"[Registry] '{query}' -> '{result2[1]}' ({result2[2]:.2f}) -> {result2[0]}")
        return result2

    # 3. Return best of both if any
    candidates = [r for r in [result, result2] if r]
    if candidates:
        best = max(candidates, key=lambda x: x[2])
        if best[2] >= 0.4:
            _log.info(f"[Best] '{query}' -> '{best[1]}' ({best[2]:.2f}) -> {best[0]}")
            return best

    _log.warning(f"[NotFound] '{query}' - no matching app found")
    return None
