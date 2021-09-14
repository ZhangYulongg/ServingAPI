import numpy as np


# output = []
def parse_log(filename="profile", num="161"):
    file = open(f"{filename}_pure", "w")
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        if f"ernie.cc:{num}" in line:
            list_ = line.split(" ")
            data = list_[4]
            print(data)
            file.write(data)
    file.close()

parse_log("gpu_profile_fixed")
parse_log("fp32_profile_fixed")
parse_log("fp16_profile_fixed")
parse_log("gpu_profile_212")
parse_log("fp32_profile_212")
parse_log("fp16_profile_212")
parse_log("gpu_profile_dev")
parse_log("fp32_profile_dev")
parse_log("fp16_profile_dev")
parse_log("fp32_212", num="156")
parse_log("fp16_212", num="156")
