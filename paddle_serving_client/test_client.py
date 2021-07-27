import os
import base64
import sys

from paddle_serving_server.serve import MainService
from paddle_serving_client import Client

sys.path.append("../paddle_serving_server")
from util import default_args


class TestClient(object):
    def setup_class(self):
        with open("./key", "rb") as f:
            self.key = f.read()

    def setup_method(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_client"

    def test_load_client_config(self):
        client = Client()
        client.load_client_config(self.client_dir)
        # check feed_names_ feed_names_to_idx and so on (feed_vars)
        assert client.feed_names_ == ["image"]
        assert client.feed_names_to_idx_ == {"image": 0}
        assert client.feed_types_ == {"image": 1}
        assert client.feed_shapes_ == {"image": [3, 224, 224]}
        assert client.feed_tensor_len == {"image": 3 * 224 * 224}
        # check fetch_vars(多模型串联时由最后一个模型决定)
        assert client.fetch_names_ == ["score"]
        assert client.fetch_names_to_idx_ == {"score": 0}
        assert client.fetch_names_to_type_ == {"score": 1}
        # TODO client.client_handle_(PredictorClient对象)

    def test_add_variant(self):
        pass

    def test_use_key(self):
        client = Client()
        client.load_client_config(self.client_dir)
        client.use_key("./key")

        # check key
        assert client.key == self.key

    def test_get_serving_port(self):
        args = default_args()
        print(args)

        client = Client()
        client.load_client_config(self.client_dir)
        client.use_key("./key")
        # client.get_serving_port(["127.0.0.1:9300"])
        pass


if __name__ == '__main__':
    tc = TestClient()
    tc.setup_method()
    tc.setup_class()
    tc.test_get_serving_port()

