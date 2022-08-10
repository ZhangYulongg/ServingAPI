# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import sys
import json
import numpy as np
import time
import argparse
import requests
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, help="log file")
    parser.add_argument("--yaml_file", type=str, help="yaml file")
    parser.add_argument("--url", type=str, help="url")
    parser.add_argument("--build_type", type=str, help="build_type")
    parser.add_argument("--repo", type=str, help="repo name")
    parser.add_argument("--description_path", type=str, help="description path")
    parser.add_argument("--branch", type=str, help="branch")
    parser.add_argument("--task_type", type=str, help="task type")
    parser.add_argument("--task_name", type=str, help="task name")
    parser.add_argument("--build_id", type=str, help="build_id")
    parser.add_argument("--build_number", type=str, help="build_number")
    parser.add_argument("--job_id", type=str, default="None", help="xly_job_id")

    return parser.parse_args()


def find_all_logs(log_path):
    for root, ds, files in os.walk(log_path):
        for file_name in files:
            if re.match(r'.*\.log', file_name):
                full_path = os.path.join(root, file_name)
                yield file_name, full_path


def process_log(log_file, yaml_file, case_name):
    with open(yaml_file, "r") as f:
        yaml_dict = yaml.safe_load(f.read())

    result_dict = {
        f"{case_name}_train_loss": {
            f"epoch_{i:03}": {
                "kpi_base": 0,
                "kpi_status": "Passed",
                "kpi_value": "None",
                "ratio": 0,
                "threshold": 0.05,
            } for i in range(120)
        },
        f"{case_name}_top1": {
            f"epoch_{i:03}": {
                "kpi_base": 0,
                "kpi_status": "Passed",
                "kpi_value": "None",
                "ratio": 0,
                "threshold": 0.02,
            } for i in range(120)
        },
        f"{case_name}_top5": {
            f"epoch_{i:03}": {
                "kpi_base": 0,
                "kpi_status": "Passed",
                "kpi_value": "None",
                "ratio": 0,
                "threshold": 0.01,
            } for i in range(120)
        },
    }
    with open(log_file, "r") as file:
        lines = [line.strip().split(" ") for line in file.readlines() if "END epoch:" in line]

    for line in lines:
        line = [i for i in line if i]
        epoch = int(line[5].split(":")[1])
        train_loss = float(line[8])
        top1 = float(line[10])
        top5 = float(line[12])
        train_loss_base = yaml_dict[f"{case_name}_train_loss"][f"epoch_{epoch:03}"]["kpi_base"]
        top1_base = yaml_dict[f"{case_name}_top1"][f"epoch_{epoch:03}"]["kpi_base"]
        top5_base = yaml_dict[f"{case_name}_top5"][f"epoch_{epoch:03}"]["kpi_base"]
        result_dict[f"{case_name}_train_loss"][f"epoch_{epoch:03}"]["kpi_value"] = train_loss
        result_dict[f"{case_name}_top1"][f"epoch_{epoch:03}"]["kpi_value"] = top1
        result_dict[f"{case_name}_top5"][f"epoch_{epoch:03}"]["kpi_value"] = top5
        result_dict[f"{case_name}_train_loss"][f"epoch_{epoch:03}"]["kpi_base"] = train_loss_base
        result_dict[f"{case_name}_top1"][f"epoch_{epoch:03}"]["kpi_base"] = top1_base
        result_dict[f"{case_name}_top5"][f"epoch_{epoch:03}"]["kpi_base"] = top5_base
        result_dict[f"{case_name}_train_loss"][f"epoch_{epoch:03}"]["ratio"] = (train_loss - train_loss_base) / train_loss_base
        result_dict[f"{case_name}_top1"][f"epoch_{epoch:03}"]["ratio"] = (top1 - top1_base) / top1_base
        result_dict[f"{case_name}_top5"][f"epoch_{epoch:03}"]["ratio"] = (top5 - top5_base) / top5_base

    # with open("result.yaml", "w") as f:
    #     yaml.dump(result_dict, f)

    return result_dict


def read_log(log_path, yaml_file):
    for file_name, full_path in find_all_logs(log_path):
        case_name = file_name.split(".")[0]
        result_dict = process_log(full_path, yaml_file, case_name)

    with open("result.yaml", "w") as f:
        yaml.dump(result_dict, f)

    return result_dict


def check_case_status(case_dict):
    status = "Passed"
    epoch = int(case_dict["kpi_name"].split("_")[1])
    if epoch >= 9:
        if abs(case_dict["kpi_value"] - case_dict["kpi_base"]) >= case_dict["threshold"]:
            status = "Failed"

    return status


def result_to_json(yaml_dict):
    json_list = []
    failed_num = 0
    for model_name in yaml_dict.keys():
        for kpi_name in yaml_dict[model_name]:
            case_dict = yaml_dict[model_name][kpi_name]
            case_dict["model_name"] = model_name
            case_dict["kpi_name"] = kpi_name
            case_dict["kpi_status"] = check_case_status(case_dict)
            if case_dict["kpi_status"] == "Failed":
                failed_num += 1
            json_list.append(case_dict)
    for case in json_list:
        print(case)

    return json_list, failed_num


def read_description_file(description_path):
    with open(description_path) as f:
        lines = [line.strip() for line in f.readlines()]
    description_dict = {line.split(":")[0]: line.split(":")[1] for line in lines}
    return description_dict


def send(args, json_list, failed_num, des_dict):
    if failed_num > 0:
        status = "Failed"
        exit_code = 8
    else:
        status = "Passed"
        exit_code = 0
    params = {
        "build_type_id": args.build_type,
        "build_id": args.build_id,
        "job_id": args.job_id,
        "repo": args.repo,
        "branch": args.branch,
        "commit_id": des_dict["commit_id"],
        "commit_time": des_dict["commit_time"],
        "status": status,
        "exit_code": exit_code,
        "duration": None,
        "case_detail": json.dumps(json_list)
    }
    res = requests.post(args.url, data=params)
    print(res.content)
    print("exit_code:", exit_code)
    sys.exit(exit_code)


if __name__ == '__main__':
    args = parse_args()
    des_dict = read_description_file(args.description_path)
    yaml_dict = read_log(args.log_path, args.yaml_file)
    json_list, failed_num = result_to_json(yaml_dict)
    send(args, json_list, failed_num, des_dict)
