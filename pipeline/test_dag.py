import pytest
import os
import copy
import yaml

from paddle_serving_server import pipeline
from paddle_serving_server.pipeline import dag, PipelineServer, PipelineClient, channel

from resnet_pipeline import ImagenetOp, ImageService


class TestDAG(object):
    def setup_method(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        image_service = ImageService()
        pipeline_server = PipelineServer("imagenet")

        read_op = pipeline.RequestOp()
        last_op = image_service.get_pipeline_response(read_op=read_op)
        response_op = pipeline.ResponseOp(input_ops=[last_op])
        self.read_op = read_op
        self.last_op = last_op
        self.response_op = response_op
        self.pipeline_server = pipeline_server

        self.default_yml_dict = {
            "build_dag_each_worker": False,
            "worker_num": 1,
            "http_port": 18080,
            "rpc_port": 9993,
            "dag": {
                "is_thread_op": False
            },
            "op": {
                "imagenet": {
                    "concurrency": 1,
                    "local_service_conf": {
                        "model_config": f"{self.dir}/../data/resnet_v2_50_imagenet_model/",
                        "device_type": 0,
                        "devices": "",
                        "client_type": "local_predictor",
                        "fetch_list": ["score"],
                        "ir_optim": False
                    }
                }
            }
        }

    def test_get_use_ops(self):
        """test get ops set"""
        # contact ops
        read_op = pipeline.RequestOp()
        last_op = ImagenetOp(name="imagenet", input_ops=[read_op])
        response_op = pipeline.ResponseOp(input_ops=[last_op])
        used_ops = [read_op, last_op]

        result_used_ops, succ_ops_of_use_op = dag.DAG.get_use_ops(response_op)
        # succ_ops_of_use_op(dict): post op list of each op (excluding ResponseOp)
        print("result_used_ops:\n", result_used_ops)
        print("succ_ops_of_use_op:\n", succ_ops_of_use_op)
        result_used_ops = list(result_used_ops)
        result_used_ops.sort(key=used_ops.index)
        assert result_used_ops[0] is read_op
        assert result_used_ops[-1] is last_op
        assert succ_ops_of_use_op["imagenet"] == []
        assert succ_ops_of_use_op["@DAGExecutor"] == [last_op]

    def test_topo_sort(self):
        """
        test Topological sort of DAG
        DAG :[A -> B -> C -> E]
                    \-> D /
        dag_views: [[E], [C, D], [B], [A]]
        """
        used_ops, succ_ops_of_use_op = dag.DAG.get_use_ops(self.response_op)

        # set dag executor
        self.pipeline_server.set_response_op(self.response_op)
        self.pipeline_server.prepare_server(yml_file=f"{self.dir}/config.yml")
        dag_conf = self.pipeline_server._conf
        dag_executor = dag.DAGExecutor(response_op=self.response_op, server_conf=dag_conf, worker_idx=-1)

        test_dag = dag_executor._dag
        dag_views, last_op_ = test_dag._topo_sort(used_ops=used_ops, response_op=self.response_op, out_degree_ops=succ_ops_of_use_op)
        print("dag_views:\n", dag_views)
        assert dag_views[0][0] is self.last_op
        assert dag_views[1][0] is self.read_op

    def test_build_dag_with_thread_op(self):
        """
        test build dag func with is_thread_op=True
        1. get_use_ops
        2. topo_sort
        3. create channels and virtual ops
        """
        yml_dict = copy.deepcopy(self.default_yml_dict)
        yml_dict["dag"]["is_thread_op"] = True
        yml_dict["op"]["imagenet"]["concurrency"] = 4
        # set dag executor
        self.pipeline_server.set_response_op(self.response_op)
        self.pipeline_server.prepare_server(yml_dict=yml_dict)
        dag_conf = self.pipeline_server._conf
        dag_executor = dag.DAGExecutor(response_op=self.response_op, server_conf=dag_conf, worker_idx=-1)

        test_dag = dag_executor._dag
        (actual_ops, channels, input_channel, output_channel, pack_func,
         unpack_func) = test_dag._build_dag(response_op=self.response_op)
        print("actual_ops:\n", actual_ops)
        print("channels:\n", channels)
        print("input_channel:\n", input_channel)
        print("output_channel:\n", output_channel)
        print("pack_func:\n", pack_func)
        print("unpack_func:\n", unpack_func)
        assert actual_ops[0] is self.last_op
        assert len(channels) == 2
        assert isinstance(input_channel, channel.ThreadChannel)
        assert isinstance(output_channel, channel.ThreadChannel)

    def test_build_dag_with_process_op(self):
        """test build dag func with is_thread_op=False"""
        yml_dict = copy.deepcopy(self.default_yml_dict)
        yml_dict["dag"]["is_thread_op"] = False
        yml_dict["op"]["imagenet"]["concurrency"] = 4
        # set dag executor
        self.pipeline_server.set_response_op(self.response_op)
        self.pipeline_server.prepare_server(yml_dict=yml_dict)
        dag_conf = self.pipeline_server._conf
        dag_executor = dag.DAGExecutor(response_op=self.response_op, server_conf=dag_conf, worker_idx=-1)

        test_dag = dag_executor._dag
        (actual_ops, channels, input_channel, output_channel, pack_func,
         unpack_func) = test_dag._build_dag(response_op=self.response_op)
        print("actual_ops:\n", actual_ops)
        print("channels:\n", channels)
        print("input_channel:\n", input_channel)
        print("output_channel:\n", output_channel)
        print("pack_func:\n", pack_func)
        print("unpack_func:\n", unpack_func)
        assert actual_ops[0] is self.last_op
        assert len(channels) == 2
        assert isinstance(input_channel, channel.ProcessChannel)
        assert isinstance(output_channel, channel.ProcessChannel)


if __name__ == '__main__':
    td = TestDAG()
    td.setup_method()
    td.test_build_dag_with_process_op()
