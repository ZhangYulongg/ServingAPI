#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;

DEFINE_string(model_file, "", "Path of the inference model file.");
DEFINE_string(params_file, "", "Path of the inference params file.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(mode, 1, "1=GPU, 2=TRT-FP32, 3=TRT-FP16");

std::shared_ptr<Predictor> InitPredictor() {
  int min_subgraph_size = 0;
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  } else {
    config.SetModel(FLAGS_model_file, FLAGS_params_file);
  }
// config.SwitchIrDebug(true);
  config.EnableUseGpu(2000, 2);
  if (FLAGS_mode == 2){
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, min_subgraph_size, PrecisionType::kFloat32, false, false);
  }
  else if (FLAGS_mode == 3)
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, min_subgraph_size, PrecisionType::kHalf, false, false);

  int batch = 32;
  int min_seq_len = 1;
  int max_seq_len = 128;
  int opt_seq_len = 128;
/*
  std::string name0 = "input_ids";
  std::string name1 = "token_type_ids";
  std::string name2 = "position_ids";
*/
  std::string name0 = "feed_0";
  std::string name1 = "feed_1";
  std::string name2 = "feed_2";
  std::string name3 = "feed_3";

  std::vector<int> min_shape = {1, 1, 1};
  std::vector<int> max_shape = {FLAGS_batch_size, 128, 1};
  std::vector<int> opt_shape = {FLAGS_batch_size, 64, 1};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {name0, min_shape},
      {name1, min_shape},
      {name2, min_shape},
      {name3, min_shape}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {name0, max_shape},
      {name1, max_shape},
      {name2, max_shape},
      {name3, max_shape}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {name0, opt_shape},
      {name1, opt_shape},
      {name2, opt_shape},
      {name3, opt_shape}};
//  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape, opt_input_shape);
  return CreatePredictor(config);
}

void run(Predictor *predictor, const std::vector<int64_t> &input,
         const std::vector<int> &input_shape, std::vector<float> *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  int run_batch = 1;
  const int run_seq_len = 128;

  int64_t i0[run_seq_len] = {
      1,    3558, 4,   75,  491, 89, 340, 313, 93,   4,   255,   10, 75,    321,
      4095, 1232, 34,   534, 49,  75, 6781, 14,  44,   868, 543,   15, 12043, 2,
      75,   201,  340, 9,   1564,  7674, 486, 218, 1140, 279, 12043, 2};
  int64_t i1[run_seq_len] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t i2[run_seq_len] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  float i3[run_seq_len] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
//  std::vector<int64_t> i0(128);
//  for (int i=0; i < input_num; ++i) {
//    i0[i] = i % 10 * 0.1;
//  }
//  LOG(INFO) << i0[23] << std::endl;

  // first input
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({run_batch, run_seq_len, 1});
  input_t->CopyFromCpu(i0);

  // second input
  auto input_t2 = predictor->GetInputHandle(input_names[1]);
  input_t2->Reshape({run_batch, run_seq_len, 1});
  input_t2->CopyFromCpu(i1);

  // third input.
  auto input_t3 = predictor->GetInputHandle(input_names[2]);
  input_t3->Reshape({run_batch, run_seq_len, 1});
  input_t3->CopyFromCpu(i2);

  auto input_t4 = predictor->GetInputHandle(input_names[3]);
  input_t4->Reshape({run_batch, run_seq_len, 1});
  input_t4->CopyFromCpu(i3);

  int warmup = 0;
  int repeat = 1;

  for (int i = 0; i < warmup; i++)
    predictor->Run();

  for (int i = 0; i < repeat; i++) {
    predictor->Run();
    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }
  std::cout << "infer done!\n";
//  std::cout << "batch: " << FLAGS_batch_size << " predict cost: " << latency << "ms" << std::endl;

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
  for (int i = 0; i < (out_num < 100 ? out_num : 100); i++) {
    LOG(INFO) << (*out_data)[i] << std::endl;
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  int c = 128;
  int h = 58;
  //int w = 2048;

  std::vector<int> input_shape = {FLAGS_batch_size, h};
  // Init input as 1.0 here for example. You can also load preprocessed real
  // pictures to vectors as input.
  std::vector<int64_t> input_data;
  std::vector<float> out_data;

  for (int a = 0; a < 10; a++){
    run(predictor.get(), input_data, input_shape, &out_data);
  }
  return 0;
}