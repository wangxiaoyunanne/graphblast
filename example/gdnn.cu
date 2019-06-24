#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>
#include <set>
#include <map>

#include <cstdio>
#include <cstdlib>

// #include <cuda_profiler_api.h>
#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/dnn.hpp"
#include "graphblas/algorithm/common.hpp"
#include "test/test.hpp"

bool debug_;
bool memory_;

int main(int argc, char** argv) {
  // Parameters
  int ntrain_sample = 60000;
  std::set<int> nlayers = {120, 480, 1920};
  std::map<int, float> nneurons = {{1024, -0.3}, {4096, -0.35}, {16384, -0.4}, {65536, -0.45}};

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [data_dir]\n", argv[0]);
    exit(1);
  }

  // Parse arguments
  bool debug;
  bool mtxinfo;
  bool filter;
  bool transpose;
  int  directed;
  int  nneuron;
  int  nlayer;
  int  batch_size;
  char* dat_name;
  po::variables_map vm;

  // Parse args
  parseArgs(argc, argv, &vm);
  debug      = vm["debug"     ].as<bool>();
  mtxinfo    = vm["mtxinfo"   ].as<bool>();
  filter     = vm["filter"    ].as<bool>();
  transpose  = vm["transpose" ].as<bool>();
  directed   = vm["directed"  ].as<int>();
  nneuron    = vm["nneuron"   ].as<int>();
  nlayer     = vm["nlayer"    ].as<int>();
  batch_size = vm["batch_size"].as<int>();

  if (nneurons.count(nneuron) == 0 || nlayers.count(nlayer) == 0) {
    std::cout << "Error: Invalid neuron or layer input!\n";
    return 0;
  }

  // Parameters for this iteration
  // nlayer:  from commandline
  // nneuron: from commandline
  float bias = nneurons[nneuron];

  // File names
  std::string true_categories_file_path = std::string(argv[argc-1]) + "/DNN/neuron" + std::to_string(nneuron) + "-l" + std::to_string(nlayer) + "-categories.tsv";
  std::string input_features_file_path = std::string(argv[argc-1]) + "/MNIST/sparse-images-" + std::to_string(nneuron) + ".mtx";
  std::string layers_file_prefix = std::string(argv[argc-1]) + "/DNN/neuron" + std::to_string(nneuron) + "/n" + std::to_string(nneuron) + "-l";

  std::vector<graphblas::Index> row_indices, row_idx_mnist, bias_idx(nneuron);
  std::vector<graphblas::Index> col_indices, col_idx_mnist;
  std::vector<float> values, val_mnist, bias_v(nneuron, bias);
  std::vector<int> true_categories_idx;
  std::vector<bool> true_categories(ntrain_sample, 0);
  graphblas::Index nrows, ncols, nvals, nrow_mnist, ncol_mnist, nval_mnist;

  std::vector<graphblas::Matrix<float>> Weights(nlayer, graphblas::Matrix<float>(nneuron, nneuron));

  // Read true categories
  std::ifstream categories_file;
  categories_file.open(true_categories_file_path.c_str());
  int x;
  while (categories_file >> x) {
    true_categories_idx.push_back(x);
    true_categories[x-1] = 1;
  }

  // Read input features
  readMtx(input_features_file_path.c_str(), &row_idx_mnist, &col_idx_mnist, &val_mnist, &nrow_mnist, &ncol_mnist, &nval_mnist, directed, mtxinfo, NULL);
  std::cout << input_features_file_path << std::endl;
  std::cout << "Batch size:        " << batch_size << std::endl;
  std::cout << "Number of neurons: " << ncol_mnist << std::endl;
  std::cout << "Number of layers:  " << nlayer << std::endl;
  
  // Read weights
  for (int layer = 0; layer < nlayer; layer++) {
    // Read mtx file of layers
    std::string file_name = layers_file_prefix + std::to_string(layer+1) + ".mtx";
    if (transpose)
      readMtx(file_name.c_str(), &col_indices, &row_indices, &values,
          &nrows, &ncols, &nvals, directed, mtxinfo, NULL);
    else
      readMtx(file_name.c_str(), &row_indices, &col_indices, &values,
          &nrows, &ncols, &nvals, directed, mtxinfo, NULL);
    std::cout << file_name << std::endl;
    
    // Build matrix
    CHECK((Weights[layer]).build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name));
    if (debug)
      CHECK(Weights[layer].print());
  }

  // Bias Vector
  Vector<float> Biases(nneuron);
  CHECK(Biases.fill(bias));

  /*!
   * This is an imperfect solution, because this should happen in
   * desc.loadArgs(vm) instead of application code!
   * TODO(@ctcyang): fix this
   */
  graphblas::Descriptor desc;
  CHECK(desc.loadArgs(vm));

  std::vector<bool> categories_val;
  float total_infer_time = 0.f;
  float total_check_time = 0.f;

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  for (graphblas::Index i = 0; i < ntrain_sample; i += batch_size) {
    // Compute current batch size
    graphblas::Index curr_batch_size = std::min(batch_size, ntrain_sample - i);

    // Prepare tuple subset that fits into current batch
    // e.g. suppose we are on second batch and want rows 15000-29999
    //     row   col  val
    //    [14999  20 0.625] x
    //    [15000   4 0.625] keep, call this start
    //      ...             keep
    //    [29999   2 0.625] keep, call this start + length
    //    [30000  15 0.625] x

    int start  = -1;
    int length = 0;
    for (size_t j = 0; j < row_idx_mnist.size(); ++j) {
      int curr_row = row_idx_mnist[j];
      if (start == -1 && curr_row >= i)
        start = j;
      if (curr_row >= i && curr_row < i + curr_batch_size)
        ++length;
      if (curr_row >= i + curr_batch_size)
        break;
    }
    if (length == 0) {
      std::cout << "Error: Zero elements in current batch!\n";
      return 0;
    }

    std::vector<graphblas::Index> temp_row_idx;
    std::vector<graphblas::Index> temp_col_idx;
    std::vector<float> temp_val_idx;
    temp_row_idx.assign(row_idx_mnist.begin()+start,
        row_idx_mnist.begin()+start+length);
    temp_col_idx.assign(col_idx_mnist.begin()+start,
        col_idx_mnist.begin()+start+length);
    temp_val_idx.assign(val_mnist.begin()+start,
        val_mnist.begin()+start+length);

    // Renumber tuple subset to start from 0
    // e.g. [15000 4 0.625] keep, call this start
    //        ...             keep
    //      [29999 2 0.625] keep, call this start + length
    // now becomes
    //      [  0   4 0.625]
    //        ...
    //      [14999 2 0.625]
    for (auto& val : temp_row_idx)
      val -= i;

    Matrix<float> mnist;
    if (transpose) {
      mnist.nnew(ncol_mnist, curr_batch_size);
      CHECK(mnist.build(&temp_col_idx, &temp_row_idx, &temp_val_idx, length,
          GrB_NULL, NULL));
    } else {
      mnist.nnew(curr_batch_size, ncol_mnist);
      CHECK(mnist.build(&temp_row_idx, &temp_col_idx, &temp_val_idx, length,
          GrB_NULL, NULL));
    }
    CHECK(mnist.nrows(&nrows));
    CHECK(mnist.ncols(&ncols));
    CHECK(mnist.nvals(&nvals));
    if (debug)
      CHECK(mnist.print());

    Matrix<float> Y;
    if (transpose)
      Y.nnew(ncol_mnist, curr_batch_size);
    else
      Y.nnew(curr_batch_size, ncol_mnist);
    Y.dup(&mnist);

    float gpu_infer_time = graphblas::algorithm::dnn(nneuron, curr_batch_size,
        mnist, Y, Weights, Biases, filter, transpose, &desc);
    warmup.Stop();

    // Extract results
    Vector<float> C(curr_batch_size);
    Vector<bool> categories(curr_batch_size);
    CHECK(categories.fill(false));
    std::vector<bool> temp_categories_val;
    Index categories_ind_size;

    graphblas::backend::GpuTimer gpu_check;
    float gpu_check_time = 0.f;
    gpu_check.Start();

    if (transpose) {
      CHECK(desc.toggle(GrB_INP0));
      graphblas::reduce<float, float, float>(&C, GrB_NULL, GrB_NULL,
          graphblas::PlusMonoid<float>(), &Y, &desc);
      CHECK(desc.toggle(GrB_INP0));
    } else {
      graphblas::reduce<float, float, float>(&C, GrB_NULL, GrB_NULL,
          graphblas::PlusMonoid<float>(), &Y, &desc);
    }

    // Extract category pattern into dense vectors
    // Note: Non-zero = true, zero = false
    assign<bool, float>(&categories, &C, GrB_NULL, true, GrB_ALL,
        curr_batch_size, &desc);

    categories_ind_size = curr_batch_size;
    CHECK(categories.extractTuples(&temp_categories_val, &categories_ind_size));

    gpu_check.Stop();
    gpu_check_time += gpu_check.ElapsedMillis();
    std::cout << "Infer time: " << gpu_infer_time << std::endl;
    std::cout << "Check time: " << gpu_check_time << std::endl;

    total_infer_time += gpu_infer_time;
    total_check_time += gpu_check_time;

    categories_val.insert(categories_val.end(), temp_categories_val.begin(),
        temp_categories_val.end());
  }
  warmup.Stop();

  // Check correctness (not timed)
  BOOST_ASSERT_LIST(true_categories, categories_val, ntrain_sample);

  std::cout << "Total time (build, infer, check): " 
      << warmup.ElapsedMillis() << std::endl;
  std::cout << "Total infer time: " << total_infer_time << std::endl;
  std::cout << "Total check time: " << total_check_time << std::endl;

  // // Benchmark
  // CpuTimer dnn_gpu_timer;
  // // cudaProfilerStart();
  // dnn_gpu_timer.Start();
  // for (int i = 0; i < niter; i++) {
  //   graphblas::algorithm::dnn(...);
  // }
  // // cudaProfilerStop();
  // dnn_gpu_timer.Stop();

  // float flop = 0;
  // std::cout << "cpu, " << dnn_cpu.ElapsedMillis() << ", \n";
  // std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
  //   flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  // float elapsed_dnn_gpu = dnn_gpu_timer.ElapsedMillis();

  // // if (niter) {
  // //   std::vector<float> h_dnn_gpu2;
  // //   CHECK(y.extractTuples(&h_dnn_gpu2, &nrows));
  // //   BOOST_ASSERT_LIST_FLOAT(h_dnn_cpu, h_dnn_gpu2, nrows);
  // // }

  return 0;
}
