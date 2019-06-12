#ifndef GRAPHBLAS_ALGORITHM_DNN_HPP_
#define GRAPHBLAS_ALGORITHM_DNN_HPP_

#include <limits>
#include <vector>
#include <string>

#include "graphblas/algorithm/test_dnn.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {

// Use float for now for both v and A
template <typename T>
Info dnn
( 
    // Matrix<T> *Yhandle,       // Y, created on output
    std::vector<Matrix<T>>& W,             // W [0..nlayers-1], each nneurons-by-nneurons
    // std::vector<Matrix<T>>& Bias,          // Bias [0..nlayers-1], diagonal nneurons-by-nneurons
    Vector<T>& Bias,
    int numNeurons,              // # of neurons
    Matrix<T>& Y0,            // Input features: nfeatures-by-nneurons
    bool checkResult,         // Check results or not
    Vector<T>& TrueCategories, // Categories
    Descriptor* desc          // Descriptor
)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    int nlayers = W.size();
    graphblas::Index Y0_rows, Y0_cols;
    CHECK(TrueCategories.size(&Y0_rows));
    if (checkResult && Y0_rows == 0) // Vector doesn't have .empty()
    {
        std::cout << "ERROR: Check results but results not provided." << std::endl;
        return (GrB_NULL_POINTER) ;
    }

    CHECK(Bias.size(&Y0_rows));
    if (Y0_rows == 0) // Vector doesn't have .empty()
    {
        std::cout << "ERROR: Bias vector empty." << std::endl;
        return (GrB_NULL_POINTER) ;
    }

    CHECK(Y0.nrows(&Y0_rows));
    CHECK(Y0.ncols(&Y0_cols));
    if (W.empty() || Y0_rows == 0 || Y0_cols == 0) // Matrix doesn't have .empty()
    {
        std::cout << "ERROR: Weights and/or input features empty." << std::endl;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // propagate the features through the neuron layers
    //--------------------------------------------------------------------------

    Matrix<T> Y(Y0_rows, Y0_cols);
    // (*Yhandle) = NULL;

    backend::GpuTimer gpu_infer;
    float gpu_infer_time = 0.f;
    gpu_infer.Start();
    for (int layer = 0 ; layer < nlayers ; layer++)
    {
        // Null mask and accum, and *+ semiring for C = A * B
        mxm<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, PlusMultipliesSemiring<T>(), 
                    ((layer == 0) ? &Y0 : &Y), &(W[layer]), desc);

        // Null mask and accum, and + semiring for C = A + B
        // bias MATRIX
        // eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, GreaterPlusSemiring<T>(), &Y, &(Bias[layer]), desc);
        // bias VECTOR
        CHECK(desc->toggle(graphblas::GrB_INP1));
        eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, GreaterPlusSemiring<T>(), &Y, &Bias, desc);
        CHECK(desc->toggle(graphblas::GrB_INP1));

        // Null mask and accum, and >0 semiring for ReLU: C = max_elem(A, 0)
        eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, PlusMaximumSemiring<T>(), &Y, 0.0, desc);

        // Optional: ReLU clipping 
    }
    gpu_infer.Stop();
    gpu_infer_time += gpu_infer.ElapsedMillis();
    std::cout << "Inference time: %f" << std::endl;


    if (checkResult) {
      Vector<T> C(Y0_rows);
      Vector<bool> Categories(Y0_rows);
      std::vector<Index> Categories_ind;
      std::vector<bool> Categories_val;
      Index Categories_ind_size;

      backend::GpuTimer gpu_check;
      float gpu_check_time = 0.f;
      gpu_check.Start();

      // C = sum(Y)
      reduce<T, T, T>(&C, GrB_NULL, PlusMonoid<T>(), &Y, desc);

      // Extract category pattern into dense vectors
      assign<bool, T>(&Categories, &C, GrB_NULL, 1, GrB_ALL, numNeurons, desc); // Non-zero = true, zero = false
      CHECK(Categories.extractTuples(&Categories_ind, &Categories_val, &Categories_ind_size)); // Convert sparse to dense

      gpu_check.Stop();
      gpu_check_time += gpu_check.ElapsedMillis();
      std::cout << "Test passed" << std::endl;
      std::cout << "Check time: %f" << std::endl;

      // Check correctness (not timed)
      for (int i = 0; i < Categories_ind_size; i++) {
        Index idx = Categories_ind[i];
        if (Categories_val[idx] != TrueCategories[idx]) {
            // printArray("True: ", TrueCategories, 5);
            // printArray("Categores: ", Categories, 5);
            std::cout << "ERROR: Mismatch at " << idx << ": (" << Categories_val[idx] << " vs " << TrueCategories[idx] << std::endl;
            return GrB_PANIC;
        }
      }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // (*Yhandle) = Y;
    return GrB_SUCCESS;
}

template <typename T>
Info dnnCpu
(
    // Matrix<T> *Yhandle,      // Y, created on output
    std::vector<Matrix<T>>& W,  // W [0..nlayers-1], each nneurons-by-nneurons
    // std::vector<Matrix<T>>& Bias,          // Bias [0..nlayers-1], diagonal nneurons-by-nneurons
    Vector<T>& Bias,
    int numNeurons,             // # of neurons
    Matrix<T>& Y0,            // Input features: nfeatures-by-nneurons
    bool checkResult,         // Check results or not
    Vector<T>& TrueCategories, // Categories
    Descriptor* desc          // Descriptor
)
{
  return SimpleReferenceDnn();
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_SSSP_HPP_
