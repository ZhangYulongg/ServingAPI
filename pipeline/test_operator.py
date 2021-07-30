import pytest

from paddle_serving_server.pipeline.operator import Op
from paddle_serving_server import pipeline

# from resnet_pipeline import ImagenetOp, ImageService
from pipeline.resnet_pipeline import ImagenetOp, ImageService


class TestOp(object):
    @pytest.mark.api_pipelineOperator_setInputOps_parameters
    def test_set_input_ops(self):
        read_op = pipeline.RequestOp()
        image_op = ImagenetOp(name="imagenet", input_ops=[read_op])
        print("image_op._input_ops:", image_op._input_ops)
        assert len(image_op._input_ops) == 1
        assert isinstance(image_op._input_ops[0], pipeline.operator.RequestOp)

    def test_init_from_dict(self):
        """test op init"""
        read_op = pipeline.RequestOp()
        image_op = ImagenetOp(name="imagenet", input_ops=[read_op])
        conf_op_dict = {
            "concurrency": 1,
            "local_service_conf": {
                "model_config": "../paddle_serving_server/resnet_v2_50_imagenet_model/",
                "device_type": 0,
                "devices": "",
                "client_type": "local_predictor",
                "fetch_list": ["score"],
                "workdir": "",
                "thread_num": 2,
                "mem_optim": True,
                "ir_optim": False,
                "precision": "fp32",
                "use_calib": False,
                "use_mkldnn": False,
                "mkldnn_cache_capacity": 0,
            },
            "timeout": -1,
            "retry": 1,
            "batch_size": 1,
            "auto_batching_timeout": -1,
        }

        image_op.init_from_dict()


if __name__ == "__main__":
    to = TestOp()
    to.test_set_input_ops()
    pass
