#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>

#include <cstdio>
#include <cstdlib>

// #include <cuda_profiler_api.h>
#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
//#include "graphblas/algorithm/DNN.hpp"
#include "graphblas/algorithm/common.hpp"
//#include "test/test.hpp"

bool debug_;
bool memory_;
int main(int argc, char** argv) {

  std::vector<graphblas::Index> row_indices, row_idx_mnist;
  std::vector<graphblas::Index> col_indices, col_idx_mnist;
  std::vector<float> values, val_mnist;
  std::vector<int> true_categories_idx;
  std::vector<bool> true_categories(60000,0);
  graphblas::Index nrows, ncols, nvals, nrow_mnist, ncol_mnist, nval_mnist;
   
  // Parse arguments
  bool debug;
  bool transpose;
  bool mtxinfo;
  int  directed;
  char* dat_name;
  int nlayers = 120;
  double bias = -0.3; 
  std :: vector <graphblas :: Matrix <float> > Weights (nlayers,graphblas :: Matrix <float> (1024,1024) ) ; 
   

  po::variables_map vm;
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else {    
    parseArgs(argc, argv, &vm);
    debug     = vm["debug"    ].as<bool>();
    transpose = vm["transpose"].as<bool>();
    mtxinfo   = vm["mtxinfo"  ].as<bool>();
    directed  = vm["directed" ].as<int>();

    /*!
     * This is an imperfect solution, because this should happen in
     * desc.loadArgs(vm) instead of application code!
     * TODO(@ctcyang): fix this
     */
    // read layers
    //std :: string mnist_file = "/home/wangxy/GraphChallenge/code/data/MNIST/sparse-images-1024.mtx" ;
    //std :: string cate_file = "/home/wangxy/GraphChallenge/code/data/DNN/neuron1024-l120-categories.tsv";
    std :: ifstream categ_file;
    categ_file.open (argv[argc-3]);
    int x;
    while (categ_file >> x) {
        true_categories_idx.push_back(x);
        true_categories[x-1] = 1;
    }
    //std :: cout<<categories.size() <<std :: endl;
    

    readMtx(argv[argc-2], &row_idx_mnist, &col_idx_mnist, &val_mnist, &nrow_mnist, &ncol_mnist,
        &nval_mnist, directed, mtxinfo, NULL);
    
   
    for (int layer = 0; layer < nlayers; layer ++ )
    {
        std :: string file_name = std :: string(argv[argc-1]) + "n1024-l";
        file_name += std::to_string(layer+1);
        file_name += ".mtx";
        readMtx(file_name.c_str() , &row_indices, &col_indices, &values, &nrows, &ncols,
        &nvals, directed, mtxinfo, NULL);
        std::cout<< file_name << nvals<<std::endl;
        //graphblas::Descriptor desc;
        // CHECK(desc.loadArgs(vm));
        //if (transpose)
        //  CHECK(desc.toggle(graphblas::GrB_INP1)); 
        
        std::cout<< "after trans" <<std::endl;
        graphblas::Matrix<float> a(nrows, ncols);
        CHECK(a.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
        dat_name));
        CHECK(a.nrows(&nrows));
        CHECK(a.ncols(&ncols));
        CHECK(a.nvals(&nvals));
        Weights[layer] = a ;
        CHECK(a.print());
        std::cout << "#values = " << values.size()  << std::endl; 
    }
    graphblas::Descriptor desc;
    CHECK(desc.loadArgs(vm));
    if (transpose)
        CHECK(desc.toggle(graphblas::GrB_INP1)); 

  } //end of reading

  graphblas::Matrix<float> mnist(nrow_mnist, ncol_mnist);
  CHECK(mnist.build(&row_idx_mnist, &col_idx_mnist, &val_mnist, nval_mnist, GrB_NULL,
        NULL));
  CHECK(mnist.nrows(&nrows));
  CHECK(mnist.ncols(&ncols));
  CHECK(mnist.nvals(&nvals));
  if (debug) CHECK(mnist.print());
  std::cout << "#mnist values = " << nval_mnist << std::endl;

  // read categories to 

 
 

  graphblas::Matrix<float> y(nrow_mnist, ncol_mnist);
  y = mnist ;
  //for (int layer = 0; layer < nlayers; layer++ )
  //{
  //    y = graphblas ::algorithm :: DNN (&y, &Weights[layer]);
  //}

  
  


return 0;
}

