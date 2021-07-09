import os


def kill_process(port):
    command = "kill -9 $(netstat -nlp | grep :"+str(port)+" | awk '{print $7}' | awk -F'/' '{{ print $1 }}')"
    os.system(command)
