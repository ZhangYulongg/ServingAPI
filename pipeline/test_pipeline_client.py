"""
test pipeline.pipeline_client
"""
import subprocess
import os
import yaml
import sys
import socket

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

sys.path.append("../paddle_serving_server")
from util import *


class TestPipelineClient(object):
    """test pipeline_client module"""
    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_model"
        self.img_path = f"{self.dir}/../data/daisy.jpg"

    def setup_method(self):
        """setup func"""
        self.err = None
        self.out = None

    def teardown_method(self):
        """release"""
        if self.err and self.out:
            self.err.close()
            self.out.close()

    def start_pipeline_server(self, sleep=5):
        """start pipeline server(need config)"""
        self.err = open("stderr.log", "w")
        self.out = open("stdout.log", "w")
        p = subprocess.Popen("python3.6 resnet_pipeline.py", shell=True, stdout=self.out, stderr=self.err)
        os.system(f"sleep {sleep}")

        print("======================stderr.log======================")
        os.system("cat stderr.log")
        print("======================stdout.log======================")
        os.system("cat stdout.log")
        print("======================================================")

    def test_predict(self):
        yml_dict = {
            "build_dag_each_worker": False,
            "worker_num": 1,
            "http_port": 18080,
            "rpc_port": 9993,
            "dag": {
                "is_thread_op": False
            },
            "op": {
                "imagenet": {
                    "concurrency": 2,
                    "local_service_conf": {
                        "model_config": f"{self.model_dir}",
                        "device_type": 1,
                        "devices": "0,1",
                        "client_type": "local_predictor",
                        "fetch_list": ["score"],
                        "ir_optim": False
                    }
                }
            }
        }
        with open("config.yml", "w") as f:
            yaml.dump(yml_dict, f, default_flow_style=False)

        self.start_pipeline_server(10)

        assert count_process_num_on_port(9993) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        client = PipelineClient()
        client.connect(["127.0.0.1:9993"])

        with open(self.img_path, "rb") as file:
            image_data = file.read()
        image = cv2_to_base64(image_data)

        # sync predict batch_size_1
        result = client.predict(feed_dict={"image_0": image}, fetch=["label", "prob"], asyn=False)
        print("sync RPC bs1 result:\n", result)
        assert result.key == ["label", "prob"]
        assert result.value == ["['daisy']", "[0.9341405034065247]"]

        # sync predict batch_size_2
        result = client.predict(feed_dict={"image_0": image, "image_1": image}, fetch=["label", "prob"], asyn=False)
        print("sync RPC bs2 result:\n", result)
        assert result.key == ["label", "prob"]
        assert result.value == ["['daisy', 'daisy']", '[0.9341405034065247, 0.9341405034065247]']

        # async predict batch_size_1
        future = client.predict(feed_dict={"image_0": image}, fetch=["label", "prob"], asyn=True)
        result = future.result()
        print("async RPC bs1 result:\n", result)
        assert result.key == ["label", "prob"]
        assert result.value == ["['daisy']", "[0.9341405034065247]"]

        # async predict batch_size_2
        future = client.predict(feed_dict={"image_0": image, "image_1": image}, fetch=["label", "prob"], asyn=True)
        result = future.result()
        print("async RPC bs2 result:\n", result)
        assert result.key == ["label", "prob"]
        assert result.value == ["['daisy', 'daisy']", '[0.9341405034065247, 0.9341405034065247]']

        kill_process(9993)
        os.system("kill -9 $(netstat -nlp | grep 'LISTENING' | awk '{print $9}' | awk -F'/' '{{ print $1 }}')")
        kill_process(18080, 3)

    def test_pack_request_package(self):
        """test proto message"""
        client = PipelineClient()
        client_ip = socket.gethostbyname(socket.gethostname())

        with open(self.img_path, "rb") as file:
            image_data = file.read()
        image = cv2_to_base64(image_data)
        feed_dict = {"image": image}

        req = client._pack_request_package(feed_dict=feed_dict, profile=False)
        print(req)
        assert req.clientip == client_ip
        assert req.logid == 0
        assert req.key == ["image"]
        assert req.value == [image]


if __name__ == '__main__':
    tpc = TestPipelineClient()
    tpc.setup_class()
    tpc.test_predict()