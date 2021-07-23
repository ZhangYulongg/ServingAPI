import argparse
import pytest
import yaml

from paddle_serving_server import OpMaker, OpSeqMaker


class TestOpMaker(object):
    def test_create_with_existed_node(self):
        op_maker = OpMaker()
        read_op = op_maker.create("general_reader")
        infer_op = op_maker.create("general_infer")
        response_op = op_maker.create("general_response")
        read_op_dict = yaml.safe_load(read_op)
        assert read_op_dict["name"] == "general_reader_0"
        assert read_op_dict["type"] == "GeneralReaderOp"
        infer_op_dict = yaml.safe_load(infer_op)
        assert infer_op_dict["name"] == "general_infer_0"
        assert infer_op_dict["type"] == "GeneralInferOp"
        response_op_dict = yaml.safe_load(response_op)
        assert response_op_dict["name"] == "general_response_0"
        assert response_op_dict["type"] == "GeneralResponseOp"

    def test_create_with_undefined_node(self):
        op_maker = OpMaker()
        with pytest.raises(Exception) as e:
            read_op = op_maker.create("general_classification")
        assert str(e.value) == "Op type general_classification is not supported right now"


class TestOpSeqMaker(object):
    def general_op_seq(self):
        op_seq_maker = OpSeqMaker()

        op_maker = OpMaker()
        read_op = op_maker.create("general_reader")
        infer_op = op_maker.create("general_infer")
        response_op = op_maker.create("general_response")
        op_seq_maker.add_op(read_op)
        op_seq_maker.add_op(infer_op)
        op_seq_maker.add_op(response_op)

        self.op_seq_maker = op_seq_maker

    @staticmethod
    def check_standard_workflow(single_workflow):
        assert single_workflow.name == "workflow1"
        assert single_workflow.workflow_type == "Sequence"
        # node 0 : general_reader
        assert single_workflow.nodes[0].name == "general_reader_0"
        assert single_workflow.nodes[0].type == "GeneralReaderOp"
        # node 1 : general_infer
        assert single_workflow.nodes[1].name == "general_infer_0"
        assert single_workflow.nodes[1].type == "GeneralInferOp"
        assert single_workflow.nodes[1].dependencies[-1].name == "general_reader_0"
        assert single_workflow.nodes[1].dependencies[-1].mode == "RO"
        # node 2 : general_response
        assert single_workflow.nodes[2].name == "general_response_0"
        assert single_workflow.nodes[2].type == "GeneralResponseOp"
        assert single_workflow.nodes[2].dependencies[-1].name == "general_infer_0"
        assert single_workflow.nodes[2].dependencies[-1].mode == "RO"

    def test_add_op(self):
        self.general_op_seq()
        self.check_standard_workflow(self.op_seq_maker.workflow)

    def test_get_op_sequence(self):
        self.general_op_seq()

        workflow_conf = self.op_seq_maker.get_op_sequence()
        assert len(workflow_conf.workflows) == 1
        self.check_standard_workflow(workflow_conf.workflows[0])


if __name__ == '__main__':
    tosm = TestOpMaker()
    tosm.test_create_with_undefined_node()
    pass
