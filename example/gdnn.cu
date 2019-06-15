#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

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
  int nlayers = 5;
  int numNeurons = 1024;
  int numFeatures = 60000;
  float bias = -0.3f;

  std::vector<graphblas::Index> row_indices, row_idx_mnist, bias_idx(numNeurons);
  std::vector<graphblas::Index> col_indices, col_idx_mnist;
  std::vector<float> values, val_mnist, bias_v(numNeurons, bias);
  std::vector<int> true_categories_idx;
  std::vector<bool> true_categories(numFeatures, 0);
  graphblas::Index nrows, ncols, nvals, nrow_mnist, ncol_mnist, nval_mnist;

  // // Vectors to build bias MATRIX
  // std::vector<graphblas::Index> row_idx_bias(numNeurons);
  // std::iota(std::begin(row_idx_bias), std::end(row_idx_bias), 0);
  // std::vector<graphblas::Index> col_idx_bias(numNeurons);
  // std::iota(std::begin(col_idx_bias), std::end(col_idx_bias), 0);
  // std::vector<float> diag_val_bias(numNeurons);
  // std::fill(std::begin(diag_val_bias), std::end(diag_val_bias), bias);

  // Parse arguments
  bool debug;
  bool transpose;
  bool mtxinfo;
  int  directed;
  char* dat_name;

  std::vector<graphblas::Matrix<float>> Weights(nlayers, graphblas::Matrix<float>(numNeurons, numNeurons));
  // std::vector<graphblas::Matrix<float>> Biases(nlayers, graphblas::Matrix<float>(numNeurons, numNeurons));
  graphblas::Vector<bool> TrueCategories(numFeatures);
  po::variables_map vm;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  }

  // Parse args
  parseArgs(argc, argv, &vm);
  debug     = vm["debug"    ].as<bool>();
  transpose = vm["transpose"].as<bool>();
  mtxinfo   = vm["mtxinfo"  ].as<bool>();
  directed  = vm["directed" ].as<int>();

  // Read true categories
  std::ifstream categories_file;
  categories_file.open(argv[argc-3]);
  int x;
  while (categories_file >> x) {
    true_categories_idx.push_back(x);
    true_categories[x-1] = 1;
  }
  TrueCategories.build(&true_categories, numFeatures);
  if (debug)
    CHECK(TrueCategories.print());

  // Read input features
  readMtx(argv[argc-2], &row_idx_mnist, &col_idx_mnist, &val_mnist, &nrow_mnist, &ncol_mnist,
      &nval_mnist, directed, mtxinfo, NULL);
  std::cout << nrow_mnist << ", " << ncol_mnist << std::endl;
  graphblas::Matrix<float> mnist(nrow_mnist, ncol_mnist);
  CHECK(mnist.build(&row_idx_mnist, &col_idx_mnist, &val_mnist, nval_mnist, GrB_NULL,
    NULL));
  CHECK(mnist.nrows(&nrows));
  CHECK(mnist.ncols(&ncols));
  CHECK(mnist.nvals(&nvals));
  if (debug)
    CHECK(mnist.print());
  
  // Read weights
  for (int layer = 0; layer < nlayers; layer++)
  {
      // Read mtx file of layers
      std::string file_name = std::string(argv[argc-1]) + "n1024-l" + std::to_string(layer+1) + ".mtx";
      readMtx(file_name.c_str() , &row_indices, &col_indices, &values, &nrows, &ncols, &nvals, directed, mtxinfo, NULL);
      std::cout<< file_name <<std::endl;
      
      // Build matrix
      CHECK((Weights[layer]).build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name));
      // CHECK(Weights[layer].nrows(&nrows));
      // CHECK(Weights[layer].ncols(&ncols));
      // CHECK(Weights[layer].nvals(&nvals));
      if (debug)
        CHECK(Weights[layer].print());

      // bias MATRIX
      // graphblas::Matrix<float> b(nrows, ncols);
      // CHECK(b.build(&row_idx_b, &col_idx_b, &diag_val_b, numNeurons, GrB_NULL, dat_name));
      // Biases[layer] = b;
      // CHECK(b.print());
  }

  // bias VECTOR
  // graphblas ::  Vector<float> Biases(nrows);
  // for (int i =0; i < numNeurons; i++) {bias_idx [i] = i ;}
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
  if (transpose)
    CHECK(desc.toggle(graphblas::GrB_INP1));

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
  graphblas::algorithm::dnn(numNeurons, numFeatures, 
                            mnist, Weights, Biases, 
                            true, /*TrueCategories,*/ true_categories, // Alternative: dense vector
                            &desc);
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
