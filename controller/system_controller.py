import subprocess, os
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

class SystemController:
    def __init__(self):
        devices = AudioUtilities.GetSpeakers()
        self.volume = cast(devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))

    def run(self, cmd):
        try:
            if cmd == "open_notepad": subprocess.Popen("notepad")
            if cmd == "open_browser": os.startfile("https://www.baidu.com")
            if cmd == "volume_up":
                current = self.volume.GetMasterVolumeLevelScalar()
                self.volume.SetMasterVolumeLevelScalar(min(current+0.1,1), None)
            if cmd == "volume_down":
                current = self.volume.GetMasterVolumeLevelScalar()
                self.volume.SetMasterVolumeLevelScalar(max(current-0.1,0), None)
            print("✅ 指令执行成功")
        except:
            print("❌ 执行失败")