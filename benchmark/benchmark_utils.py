import argparse
import os
import time
import logging

import paddle
import paddle.inference as paddle_infer

from pathlib import Path

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH_ROOT = f"{CUR_DIR}/benchmark_logs"
# LOG_PATH_ROOT = f"/mnt/serving/zyl/benchmark_logs"
# LOG_PATH_ROOT = f"."

class PaddleInferBenchmark(object):
    def __init__(self,
                 config,
                 model_info: dict={},
                 data_info: dict={},
                 perf_info: dict={},
                 resource_info: dict={},
                 identifier: str="",
                 **kwargs):
        """
        Construct PaddleInferBenchmark Class to format logs.
        args:
            config(paddle.inference.Config): paddle inference config
            model_info(dict): basic model info
                {'model_name': 'resnet50'
                 'precision': 'fp32'}
            data_info(dict): input data info
                {'batch_size': 1
                 'shape': '3,224,224'
                 'data_num': 1000}
            perf_info(dict): performance result
                {'preprocess_time_s': 1.0
                'inference_time_s': 2.0
                'postprocess_time_s': 1.0
                'total_time_s': 4.0}
            resource_info(dict):
                cpu and gpu resources
                {'cpu_rss': 100
                 'gpu_rss': 100
                 'gpu_util': 60}
        """
        # PaddleInferBenchmark Log Version
        self.log_version = "1.0.3"

        # Paddle Version
        self.paddle_version = paddle.__version__
        self.paddle_commit = paddle.__git_commit__
        paddle_infer_info = paddle_infer.get_version()
        self.paddle_branch = paddle_infer_info.strip().split(': ')[-1]

        # model info
        self.model_info = model_info

        # data info
        self.data_info = data_info

        # perf info
        self.perf_info = perf_info
        try:
            # required value
            self.model_name = model_info['model_name']
            self.precision = model_info['precision']

            self.batch_size = data_info['batch_size']
            self.shape = data_info['shape']
            self.data_num = data_info['data_num']

            self.inference_time_ms = round(perf_info['inference_time_ms'], 4)
        except:
            self.print_help()
            raise ValueError(
                "Set argument wrong, please check input argument and its type")

        self.client_mode = perf_info.get('client_mode', 'nan')
        self.server_mode = perf_info.get('server_mode', 'nan')
        self.preprocess_time_s = perf_info.get('preprocess_time_s', 0)
        self.postprocess_time_s = perf_info.get('postprocess_time_s', 0)
        self.total_time_s = perf_info.get('total_time_s', 0)
        self.each_time_s = perf_info.get('each_time_s', 0)
        self.median = perf_info.get('median(ms)', 0)

        self.inference_time_ms_80 = round(perf_info.get("inference_time_ms_80", 0), 4)
        self.inference_time_ms_90 = round(perf_info.get("inference_time_ms_90", 0), 4)
        self.inference_time_ms_99 = round(perf_info.get("inference_time_ms_99", 0), 4)
        self.succ_rate = perf_info.get("succ_rate", "")
        self.qps = perf_info.get("qps", "")

        # conf info
        self.config_status = self.parse_config(config)

        # identifier
        self.thread_num = identifier.split(" ")[3]
        # self.batch_size

        # mem info
        if isinstance(resource_info, dict):
            self.cpu_rss_mb = int("-1" if 'cpu_rss_mb' not in resource_info or resource_info.get('cpu_rss_mb').strip()=="" else resource_info.get('cpu_rss_mb', 0))
            self.cpu_vms_mb = int("-1" if 'cpu_vms_mb' not in resource_info or resource_info.get('cpu_vms_mb').strip()=="" else resource_info.get('cpu_vms_mb', 0))
            self.cpu_shared_mb = int("-1" if 'cpu_shared_mb' not in resource_info or resource_info.get('cpu_shared_mb').strip()=="" else resource_info.get('cpu_shared_mb', 0))
            self.cpu_dirty_mb = int("-1" if 'cpu_dirty_mb' not in resource_info or resource_info.get('cpu_dirty_mb').strip()=="" else resource_info.get('cpu_dirty_mb', 0))
            self.cpu_util = round(resource_info.get('cpu_util', 0), 2)

            self.gpu_rss_mb = int("-1" if 'gpu_rss_mb' not in resource_info or resource_info.get('gpu_rss_mb').strip()=="" else resource_info.get('gpu_rss_mb', 0))
            #self.gpu_util = round(resource_info.get('gpu_util', 0), 2)
            self.gpu_util = resource_info.get('gpu_util', 0)
            #self.gpu_mem_util = round(resource_info.get('gpu_mem_util', 0), 2)
            self.gpu_mem =  resource_info.get('gpu_mem', 0)
            self.gpu_mem_util =  resource_info.get('gpu_mem_util', 0)
        else:
            self.cpu_rss_mb = 0
            self.cpu_vms_mb = 0
            self.cpu_shared_mb = 0
            self.cpu_dirty_mb = 0
            self.cpu_util = 0

            self.gpu_rss_mb = 0
            self.gpu_util = 0
            self.gpu_mem_util = 0

        # init benchmark logger
        self.benchmark_logger()

    def benchmark_logger(self):
        """
        benchmark logger
        """
        # remove other logging handler
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Init logger
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_output = f"{LOG_PATH_ROOT}/{self.model_name}_tn{self.thread_num}_bs{self.batch_size}_{self.client_mode}.log"
        Path(f"{LOG_PATH_ROOT}").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=FORMAT,
            handlers=[
                logging.FileHandler(
                    filename=log_output, mode='w'),
                logging.StreamHandler(),
            ])
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Paddle Inference benchmark log will be saved to {log_output}")

    def parse_config(self, config) -> dict:
        """
        parse paddle predictor config
        args:
            config(paddle.inference.Config): paddle inference config
        return:
            config_status(dict): dict style config info
        """
        if isinstance(config, paddle_infer.Config):
            config_status = {}
            config_status['runtime_device'] = "gpu" if config.use_gpu(
            ) else "cpu"
            config_status['ir_optim'] = config.ir_optim()
            config_status['enable_tensorrt'] = config.tensorrt_engine_enabled()
            config_status['precision'] = self.precision
            config_status['enable_mkldnn'] = config.mkldnn_enabled()
            config_status[
                'cpu_math_library_num_threads'] = config.cpu_math_library_num_threads(
                )
        elif isinstance(config, dict):
            config_status = {}
            config_status['runtime_device'] = config.get('runtime_device', "")
            config_status['ir_optim'] = config.get('ir_optim', "")
            config_status['enable_tensorrt'] = config.get('enable_tensorrt', "")
            config_status['precision'] = config.get('precision', "")
            config_status['enable_mkldnn'] = config.get('enable_mkldnn', "")
            config_status['cpu_math_library_num_threads'] = config.get(
                'cpu_math_library_num_threads', "")
        else:
            self.print_help()
            raise ValueError(
                "Set argument config wrong, please check input argument and its type"
            )
        return config_status

    def report(self, identifier=None):
        """
        print log report
        args:
            identifier(string): identify log
        """
        if identifier:
            identifier = f"[{identifier}]"
        else:
            identifier = ""

        self.logger.info("\n")
        self.logger.info(
            "---------------------- Paddle info ----------------------")
        self.logger.info(f"{identifier} paddle_version: {self.paddle_version}")
        self.logger.info(f"{identifier} paddle_commit: {self.paddle_commit}")
        self.logger.info(f"{identifier} paddle_branch: {self.paddle_branch}")
        self.logger.info(f"{identifier} log_api_version: {self.log_version}")
        self.logger.info(
            "----------------------- Conf info -----------------------")
        self.logger.info(
            f"{identifier} runtime_device: {self.config_status['runtime_device']}"
        )
        self.logger.info(
            f"{identifier} ir_optim: {self.config_status['ir_optim']}")
        self.logger.info(f"{identifier} enable_memory_optim: {True}")
        self.logger.info(
            f"{identifier} enable_tensorrt: {self.config_status['enable_tensorrt']}"
        )
        self.logger.info(
            f"{identifier} enable_mkldnn: {self.config_status['enable_mkldnn']}")
        self.logger.info(
            f"{identifier} cpu_math_library_num_threads: {self.config_status['cpu_math_library_num_threads']}"
        )
        self.logger.info(
            "----------------------- Model info ----------------------")
        self.logger.info(f"{identifier} model_name: {self.model_name}")
        self.logger.info(f"{identifier} precision: {self.precision}")
        self.logger.info(
            "----------------------- Data info -----------------------")
        self.logger.info(f"{identifier} batch_size: {self.batch_size}")
        self.logger.info(f"{identifier} input_shape: {self.shape}")
        self.logger.info(f"{identifier} data_num: {self.data_num}")
        self.logger.info(
            "----------------------- Perf info -----------------------")
        self.logger.info(
            f"{identifier} cpu_rss(MB): {self.cpu_rss_mb}, cpu_vms: {self.cpu_vms_mb}, cpu_shared_mb: {self.cpu_shared_mb}, cpu_dirty_mb: {self.cpu_dirty_mb}, cpu_util: {self.cpu_util}%"
        )
        self.logger.info(
            f"{identifier} gpu_rss(MB): {self.gpu_rss_mb}, gpu_util: {self.gpu_util}%, gpu_mem: {self.gpu_mem}, gpu_mem_util: {self.gpu_mem_util}%"
        )
        self.logger.info(
            f"{identifier} total time spent(s): {self.total_time_s}")
        self.logger.info(
            f"{identifier} thread_num: {self.thread_num}, client_mode: {self.client_mode}, server_mode: {self.server_mode}")
        self.logger.info(
            f"{identifier} preprocess_time(ms): {self.preprocess_time_s}, inference_time(ms): {self.inference_time_ms}, postprocess_time(ms): {self.postprocess_time_s}"
        )
        if self.inference_time_ms_90:
            self.logger.info(
                f"{identifier} 80%_cost(ms): {self.inference_time_ms_80}, 90%_cost(ms): {self.inference_time_ms_90}, 99%_cost(ms): {self.inference_time_ms_99}, succ_rate: {self.succ_rate}"
            )
        if self.qps:
            self.logger.info(f"{identifier} QPS: {self.qps}")
        self.logger.info("----------------------- Serving info -----------------------")
        self.logger.info(
            f"{identifier} client_num: {self.thread_num}, median(ms): {self.median}")
        self.logger.info(
            f"{identifier} total_time_s: {self.total_time_s}, each_time_s: {self.each_time_s}")

    def print_help(self):
        """
        print function help
        """
        print("""Usage:
            ==== Print inference benchmark logs. ====
            config = paddle.inference.Config()
            model_info = {'model_name': 'resnet50'
                          'precision': 'fp32'}
            data_info = {'batch_size': 1
                         'shape': '3,224,224'
                         'data_num': 1000}
            perf_info = {'preprocess_time_s': 1.0
                         'inference_time_s': 2.0
                         'postprocess_time_s': 1.0
                         'total_time_s': 4.0}
            resource_info = {'cpu_rss_mb': 100
                             'gpu_rss_mb': 100
                             'gpu_util': 60}
            log = PaddleInferBenchmark(config, model_info, data_info, perf_info, resource_info)
            log('Test')
            """)

    def __call__(self, identifier=None):
        """
        __call__
        args:
            identifier(string): identify log
        """
        self.report(identifier)
