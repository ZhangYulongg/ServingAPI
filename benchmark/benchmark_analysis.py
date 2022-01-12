import os
import sys
import re
import argparse
import json
import logging
import requests

from copy import deepcopy

import pandas as pd

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    # for local excel analysis
    parser.add_argument(
        "--log_path", type=str, default="./log", help="benchmark log path")
    parser.add_argument(
        "--output_name",
        type=str,
        default="benchmark_excel.xlsx",
        help="output excel file name")
    parser.add_argument(
        "--output_html_name",
        type=str,
        default="benchmark_data.html",
        help="output html file name")
    return parser.parse_args()


class BenchmarkLogAnalyzer(object):
    def __init__(self, args):
        """
        """
        self.args = args

        # PaddleInferBenchmark Log Analyzer version, should be same as PaddleInferBenchmark Log Version
        self.analyzer_version = "1.0.3"

        # init dataframe and dict style
        self.origin_df = pd.DataFrame(columns=[
            "model_name", "server_mode", "client_mode", "client_num", "batch_size", "cpu_util", "gpu_mem",
            "gpu_util", "QPS", "data_num", "inference_time(ms)", "median(ms)",
            "80%_cost(ms)", "90%_cost(ms)", "99%_cost(ms)", "total_time_s", "each_time_s"
            "paddle_version", "paddle_commit",
            "runtime_device", "ir_optim", "enable_memory_optim",
            "enable_tensorrt", "enable_mkldnn", "precision",
        ])
        self.benchmark_key = self.origin_df.to_dict()

    def find_all_logs(self, path_walk: str):
        """
        find all .log files from target dir
        """
        for root, ds, files in os.walk(path_walk):
            for file_name in files:
                if re.match(r'.*.log', file_name):
                    full_path = os.path.join(root, file_name)
                    yield file_name, full_path

    def process_log(self, file_name: str) -> dict:
        """
        """
        output_dict = deepcopy(self.benchmark_key)
        with open(file_name, 'r') as f:
            for i, data in enumerate(f.readlines()):
                if i == 0:
                    continue
                line_lists = data.split(" ")

                for key_name, _ in output_dict.items():
                    key_name_in_log = "".join([key_name, ":"])
                    if key_name_in_log in line_lists:
                        pos_buf = line_lists.index(key_name_in_log)
                        output_dict[key_name] = line_lists[pos_buf + 1].strip(
                        ).split(',')[0]

        # 微调
        output_dict["client_num"] = int(output_dict["client_num"])

        empty_values = []
        for k, _ in output_dict.items():
            if not output_dict[k]:
                output_dict[k] = None
                empty_values.append(k)

        if not empty_values:
            logger.info("no empty value found")
        else:
            logger.warning(f"{empty_values} is empty, not found in logs")
        return output_dict

    def __call__(self, log_path, to_database=False):
        """
        """
        # analysis log to dict and dataframe
        for file_name, full_path in self.find_all_logs(log_path):
            dict_log = self.process_log(full_path)
            self.origin_df = self.origin_df.append(dict_log, ignore_index=True)

        raw_df = self.origin_df.sort_values(by='model_name')
        raw_df.sort_values(by=["model_name", "client_mode", "client_num"], inplace=True)
        raw_df.to_excel(self.args.output_name, index=False)     # render excel
        raw_df.to_html(self.args.output_html_name) # render html
        print(raw_df)


def main():
    """
    main
    """
    args = parse_args()
    analyzer = BenchmarkLogAnalyzer(args)
    analyzer(args.log_path, True)


if __name__ == "__main__":
    main()