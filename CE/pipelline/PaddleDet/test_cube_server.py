import os
import sys
from paddle_serving_server import OpMaker
from paddle_serving_server import OpSeqMaker
from paddle_serving_server import Server

op_maker = OpMaker()
read_op = op_maker.create('general_reader')
general_dist_kv_infer_op = op_maker.create('general_dist_kv_infer')
response_op = op_maker.create('general_response')

op_seq_maker = OpSeqMaker()
op_seq_maker.add_op(read_op)
op_seq_maker.add_op(general_dist_kv_infer_op)
op_seq_maker.add_op(response_op)

server = Server()
server.set_op_sequence(op_seq_maker.get_op_sequence())
server.set_num_threads(4)
server.load_model_config(sys.argv[1])
server.prepare_server(
    workdir="work_dir1",
    port=9494,
    device="cpu",
    cube_conf="./cube/conf/cube.conf")
server.run_server()
