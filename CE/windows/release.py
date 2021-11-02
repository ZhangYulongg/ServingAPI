import os
import psutil
import sys
pids = psutil.pids()
for pid in pids:
    p = psutil.Process(pid)
    if p.name() == 'python.exe':
        cmd = 'taskkill /F /IM python.exe'
        os.system(cmd)
sys.exit(0)