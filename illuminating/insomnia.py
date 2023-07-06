'''
Prevent OS sleep/hibernate

Based On:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
'''

import ctypes
import os

## ----------------------
## https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx
## ----------------------
WINDOWS_ES_CONTINUOUS = 0x80000000
WINDOWS_ES_SYSTEM_REQUIRED = 0x00000001
## ----------------------

def induce():
    if os.name != "nt":
        print("MF_OS_INSOMNIA: insomnia only supported for Windows")
        return

    _induce_for_windows()

def dissuade():
    if os.name != "nt":
        print("MF_OS_INSOMNIA: insomnia only supported for Windows")
        return

    _dissuade_for_windows()

def _induce_for_windows():
    print("OS_INSOMNIA: preventing system from sleeping")
    ctypes.windll.kernel32.SetThreadExecutionState(WINDOWS_ES_CONTINUOUS | WINDOWS_ES_SYSTEM_REQUIRED)

def _dissuade_for_windows():
    print("OS_INSOMNIA: allowing system to sleep")
    ctypes.windll.kernel32.SetThreadExecutionState(WINDOWS_ES_CONTINUOUS)
