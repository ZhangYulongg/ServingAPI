import os

from paddle_serving_client.io import inference_model_to_serving


class TestClientIO(object):
    def setup_class(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.origin_model = f"{self.dir}/ResNet50"

    def test_inference_model_to_serving(self):
        feed_names, fetch_names = inference_model_to_serving(dirname=self.origin_model, model_filename="model", params_filename="params")
        print("feed_names:", list(feed_names))
        print("fetch_names:", list(fetch_names))
        assert list(feed_names) == ["image"]
        assert list(fetch_names) == ["save_infer_model/scale_0.tmp_0"]


if __name__ == '__main__':
    tci = TestClientIO()
    tci.setup_class()
    tci.test_inference_model_to_serving()
