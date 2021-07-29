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


if __name__ == '__main__':
    to = TestOp()
    to.test_set_input_ops()
    pass
