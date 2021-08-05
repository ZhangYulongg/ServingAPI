import subprocess
import os
import yaml
import sys
import socket

from paddle_serving_server.pipeline import PipelineClient

sys.path.append("../paddle_serving_server")
from util import *


class TestPipelineClient(object):
    """test pipeline_client module"""
    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.img_path = f"{self.dir}/../paddle_serving_server/daisy.jpg"

    def teardown_method(self):
        """release"""
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
                        "model_config": "../paddle_serving_server/resnet_v2_50_imagenet_model/",
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

        client = PipelineClient()
        client.connect(["127.0.0.1:9993"])
        print(client._channel, type(client._channel))
        print(client._stub, type(client._stub))

        os.system("cat stderr.log")

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
        print(req.value)
        print(req.key)
        print(req.logid)
        print(req.clientip)
        assert req.clientip == client_ip
        assert req.logid == 0
        assert req.key == ["image"]
        assert req.value == [image]


if __name__ == '__main__':
    tpc = TestPipelineClient()
    tpc.setup_class()
    tpc.test_pack_request_package()