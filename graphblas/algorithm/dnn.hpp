#ifndef GRAPHBLAS_ALGORITHM_DNN_HPP_
#define GRAPHBLAS_ALGORITHM_DNN_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_dnn.hpp"
#include "graphblas/backend/cuda/util.hpp"
#include "test/test.hpp"

namespace graphblas {
namespace algorithm {

//--------------------------------------------------------------------------
// numNeurons = nrows = ncols
// numNeurons: 1024..., numLayers: 120..., numFeatures: 65536

// W: vector of Matrix<T>, size (nlayers, numNeurons, numNeurons)

// Bias: Vector<T>, size (numNeurons, 1)

// Y0: Matrix<T>, size (numFeatures, numNeurons)
//--------------------------------------------------------------------------

template <typename Enumeration>
auto as_integer(Enumeration const value)
    -> typename std::underlying_type<Enumeration>::type {
  return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}


template <typename T>
Info dnn ( 
  int numNeurons,               // # of neurons
  int numFeatures,              // # of features
  Matrix<T>& Y0,                // Input features: nfeatures-by-nneurons
  Matrix<T>& Y,                 // Activation: nfeatures-by-nneurons
  std::vector<Matrix<T>>& W,    // W, size (nlayers, numNeurons, numNeurons)
  Vector<T>& Bias,              // Bias, size (numNeurons, 1)
  bool filter,                  // Filter out 0's from matrix or not
  bool transpose,               // Whether we are doing Y_1 = Y_0 x W + b or
                                // Y_1^T = W^T x Y_0^T + b^T
  Descriptor* desc              // Descriptor
) {
  int nlayers = W.size();
  Index Y0_rows, Y0_cols, Y0_nrows, Y0_ncols;
  // Using alternative: dense vector
  CHECK(Bias.size(&Y0_rows));

  // Vector doesn't have .empty()
  if (Y0_rows == 0) {
    std::cout << "Error: Bias vector empty." << std::endl;
    return GrB_NULL_POINTER;
  }

  CHECK(Y0.nrows(&Y0_rows));
  CHECK(Y0.ncols(&Y0_cols));
  if (W.empty() || Y0_rows == 0 || Y0_cols == 0) {
    std::cout << "Error: Weights and/or input features empty." << std::endl;
    return GrB_NULL_POINTER;
  }

  Matrix<T> Y_swap(Y0_rows, Y0_cols);

  backend::GpuTimer gpu_infer;
  float gpu_infer_time = 0.f;
  gpu_infer.Start();
  for (int layer = 0; layer < nlayers; layer++) {
    if (desc->descriptor_.debug())
      std::cout << "=====Layer " << layer + 1 << "=====\n";

    if (transpose)
      mxm<T, T, T, T>(&Y_swap, GrB_NULL, GrB_NULL, PlusMultipliesSemiring<T>(), &(W[layer]), &Y, desc);
    else
      mxm<T, T, T, T>(&Y_swap, GrB_NULL, GrB_NULL, PlusMultipliesSemiring<T>(), &Y, &(W[layer]), desc);

    CHECK(Y.swap(&Y_swap));

    // Null mask and accum, and + semiring for C = A + B
    if (!transpose)
      CHECK(desc->toggle(graphblas::GrB_INP1));
    eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, GreaterPlusSemiring<T>(),
        &Y, &Bias, desc);
    if (!transpose)
      CHECK(desc->toggle(graphblas::GrB_INP1));

    // Null mask and accum, and >0 semiring for ReLU: C = max_elem(A, 0)
    eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, PlusMaximumSemiring<T>(),
        &Y, 0.f, desc);

    // Filter out 0's from sparse matrix
    if (filter)
      CHECK(Y.rebuild(0.f, desc));

    // Optional: clipping of values above 32 
    eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, PlusMinimumSemiring<T>(),
        &Y, 32.f, desc);
  }
  gpu_infer.Stop();
  gpu_infer_time += gpu_infer.ElapsedMillis();
  std::cout << "Inference time: " << gpu_infer_time << std::endl;

  return GrB_SUCCESS;
}

template <typename T>
Info dnnCpu (
    std::vector<Matrix<T>>& W,  // W [0..nlayers-1], each nneurons-by-nneurons
    Vector<T>& Bias,
    int numNeurons,             // # of neurons
    Matrix<T>& Y0,              // Input features: nfeatures-by-nneurons
    Matrix<T>& Y,               // Activations: nfeatures-by-nneurons
    Descriptor* desc            // Descriptor
)
{
  return SimpleReferenceDnn<T>();
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_SSSP_HPP_
