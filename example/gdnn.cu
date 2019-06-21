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
  std::set<int> nlayers = {120, 480, 1920};
  std::map<int, float> nneurons = {{1024, -0.3}, {4096, -0.35}, {16384, -0.4}, {65536, -0.45}};
  int nfeatures = 60000;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [data_dir]\n", argv[0]);
    exit(1);
  }

  // Parse arguments
  bool debug;
  bool mtxinfo;
  bool filter;
  int  directed;
  int  nneuron;
  int  nlayer;
  char* dat_name;
  po::variables_map vm;

  // Parse args
  parseArgs(argc, argv, &vm);
  debug     = vm["debug"    ].as<bool>();
  mtxinfo   = vm["mtxinfo"  ].as<bool>();
  filter    = vm["filter"   ].as<bool>();
  directed  = vm["directed" ].as<int>();
  nneuron   = vm["nneuron"  ].as<int>();
  nlayer    = vm["nlayer"   ].as<int>();

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
  std::vector<bool> true_categories(nfeatures, 0);
  graphblas::Index nrows, ncols, nvals, nrow_mnist, ncol_mnist, nval_mnist;

  std::vector<graphblas::Matrix<float>> Weights(nlayer, graphblas::Matrix<float>(nneuron, nneuron));
  // std::vector<graphblas::Matrix<float>> Biases(nlayer, graphblas::Matrix<float>(nneuron, nneuron));
  graphblas::Vector<bool> TrueCategories(nfeatures);

  // Read true categories
  std::ifstream categories_file;
  categories_file.open(true_categories_file_path.c_str());
  int x;
  while (categories_file >> x) {
    true_categories_idx.push_back(x);
    true_categories[x-1] = 1;
  }
  TrueCategories.build(&true_categories, nfeatures);
  if (debug)
    CHECK(TrueCategories.print());

  // Read input features
  readMtx(input_features_file_path.c_str(), &row_idx_mnist, &col_idx_mnist, &val_mnist, &nrow_mnist, &ncol_mnist, &nval_mnist, directed, mtxinfo, NULL);
  std::cout << input_features_file_path << std::endl;
  std::cout << "Batch size:        " << nrow_mnist << std::endl;
  std::cout << "Number of neurons: " << ncol_mnist << std::endl;
  std::cout << "Number of layers:  " << nlayer << std::endl;
  graphblas::Matrix<float> mnist(nrow_mnist, ncol_mnist);
  CHECK(mnist.build(&row_idx_mnist, &col_idx_mnist, &val_mnist, nval_mnist, GrB_NULL, NULL));
  CHECK(mnist.nrows(&nrows));
  CHECK(mnist.ncols(&ncols));
  CHECK(mnist.nvals(&nvals));
  if (debug)
    CHECK(mnist.print());
  
  // Read weights
  for (int layer = 0; layer < nlayer; layer++) {
    // Read mtx file of layers
    std::string file_name = layers_file_prefix + std::to_string(layer+1) + ".mtx";
    readMtx(file_name.c_str() , &row_indices, &col_indices, &values, &nrows, &ncols, &nvals, directed, mtxinfo, NULL);
    std::cout<< file_name << std::endl;
    
    // Build matrix
    CHECK((Weights[layer]).build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name));
    // CHECK(Weights[layer].nrows(&nrows));
    // CHECK(Weights[layer].ncols(&ncols));
    // CHECK(Weights[layer].nvals(&nvals));
    if (debug)
      CHECK(Weights[layer].print());
    // bias MATRIX
    // graphblas::Matrix<float> b(nrows, ncols);
    // CHECK(b.build(&row_idx_b, &col_idx_b, &diag_val_b, nneuron, GrB_NULL, dat_name));
    // Biases[layer] = b;
    // CHECK(b.print());
  }

  // bias VECTOR
  // graphblas ::  Vector<float> Biases(nrows);
  // for (int i =0; i < nneuron; i++) {bias_idx [i] = i ;}
  // CHECK(Biases.build(&bias_idx, &bias_v, nrows,GrB_NULL));
  Vector<float> Biases(nrows);
  CHECK(Biases.fill(bias));

  /*!
   * This is an imperfect solution, because this should happen in
   * desc.loadArgs(vm) instead of application code!
   * TODO(@ctcyang): fix this
   */
  graphblas::Descriptor desc;
  CHECK(desc.loadArgs(vm));

  // // Cpu BFS
  // CpuTimer dnn_cpu;
  // float* h_dnn_cpu = reinterpret_cast<float*>(malloc(nrows*sizeof(float)));
  // int depth = 10000;
  // dnn_cpu.Start();
  // int d = graphblas::algorithm::dnnCpu(...);
  // dnn_cpu.Stop();

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::dnn(nneuron, nfeatures, 
                            mnist, Weights, Biases, 
                            true, /*TrueCategories,*/ true_categories, // Alternative: dense vector
                            filter, &desc);
  warmup.Stop();

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
