import os
import psutil
pids = psutil.pids()
for pid in pids:
    p = psutil.Process(pid)
    if p.name() == 'python.exe':
        cmd = 'taskkill /F /IM python.exe'
        os.system(cmd)