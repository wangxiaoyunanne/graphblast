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
// numNeurons: 1024..., numLayers: 120..., numFeatures: 60000

// W: vector of Matrix<T>, size (nlayers, numNeurons, numNeurons)

// Bias: Vector<T>, size (numNeurons, 1)

// Y0: Matrix<T>, size (numFeatures, numNeurons)

// TrueCategories: Vector<bool>, size (numFeatures, 1)
// Or:
// TrueCategories: std::vector<bool>, size(numFeatures, 1)
//--------------------------------------------------------------------------

template <typename Enumeration>
auto as_integer(Enumeration const value)
    -> typename std::underlying_type<Enumeration>::type
{
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}


template <typename T>
Info dnn
( 
    int numNeurons,               // # of neurons
    int numFeatures,              // # of features

    // std::vector<Matrix<T>>& Bias, // Bias [0..nlayers-1], diagonal nneurons-by-nneurons

    Matrix<T>& Y0,                // Input features: nfeatures-by-nneurons
    std::vector<Matrix<T>>& W,    // W, size (nlayers, numNeurons, numNeurons)
    Vector<T>& Bias,              // Bias, size (numNeurons, 1)

    bool checkResult,             // Check results or not
    // Vector<bool>& TrueCategories, // TrueCategories, size (numFeatures, 1)
    std::vector<bool>& TrueCategories, // Alternative: TrueCategories, size (numFeatures, 1)

    Descriptor* desc              // Descriptor
)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    int nlayers = W.size();
    Index Y0_rows, Y0_cols, Y0_nrows, Y0_ncols;
    // CHECK(TrueCategories.size(&Y0_rows));
    Y0_rows = TrueCategories.size(); // Alternative: dense vector
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
    Y.dup(&Y0);
    Matrix<T> Y_swap(Y0_rows, Y0_cols);

    backend::GpuTimer gpu_infer;
    float gpu_infer_time = 0.f;
    gpu_infer.Start();
    for (int layer = 0; layer < nlayers; layer++)
    {
        std::cout << "################## Layer: #" << (layer + 1) << " ##################" << std::endl;

        Storage s;
        // Null mask and accum, and *+ semiring for C = A * B
        std::cout << "********** MXM" << std::endl;
        // std::cout << "W: *****" << std::endl;
        // CHECK(W[layer].print());
        // std::cout << "Y: *****" << std::endl;
        // CHECK(Y.print());
        mxm<T, T, T, T>(&Y_swap, GrB_NULL, GrB_NULL, PlusMultipliesSemiring<T>(), &Y, &(W[layer]), desc);

        CHECK(Y.swap(&Y_swap));

        // CHECK(Y0.getStorage(&s));
        // std::cout << "Y0 storage: " << as_integer(s) << std::endl;
        // CHECK(Y.getStorage(&s));
        // std::cout << "Y storage after Y*W: " << as_integer(s) << std::endl;
        // CHECK(W[layer].getStorage(&s));
        // std::cout << "W storage: " << as_integer(s) << std::endl;

        // Null mask and accum, and + semiring for C = A + B
        // bias MATRIX
        // eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, GreaterPlusSemiring<T>(), &Y, &(Bias[layer]), desc);
        // bias VECTOR
        std::cout << "********** Plus bias" << std::endl;
        CHECK(desc->toggle(graphblas::GrB_INP1));
        eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, GreaterPlusSemiring<T>(), &Y, &Bias, desc);
        CHECK(desc->toggle(graphblas::GrB_INP1));
        // CHECK(Bias.getStorage(&s));
        // std::cout << "Bias storage: " << as_integer(s) << std::endl;

        // Null mask and accum, and >0 semiring for ReLU: C = max_elem(A, 0)
        std::cout << "********** ReLU" << std::endl;
        eWiseMult<T, T, T, T>(&Y, GrB_NULL, GrB_NULL, PlusMaximumSemiring<T>(), &Y, 0.0, desc);
        // CHECK(Y.getStorage(&s));
        // std::cout << "Y storage after ReLU: " << as_integer(s) << std::endl;

        // Optional: ReLU clipping 
    }
    gpu_infer.Stop();
    gpu_infer_time += gpu_infer.ElapsedMillis();
    std::cout << "Inference time: " << gpu_infer_time << std::endl;

    if (checkResult) {
      Vector<T> C(numFeatures);
      Vector<bool> Categories(numFeatures);
      CHECK(Categories.fill(false));
      std::vector<bool> Categories_val;
      std::vector<Index> Categories_ind; // If Categories is sparse
      Index Categories_ind_size;

      backend::GpuTimer gpu_check;
      float gpu_check_time = 0.f;
      gpu_check.Start();

      Storage s;
      // C = sum(Y)
      reduce<T, T, T>(&C, GrB_NULL, GrB_NULL, PlusMonoid<T>(), &Y, desc);
      // CHECK(Y.getStorage(&s));
      // std::cout << "Y0 storage before: " << as_integer(s) << std::endl;

      // Extract category pattern into dense vectors
      // CHECK(C.getStorage(&s));
      // CHECK(Categories.setStorage(s));
      // std::cout << "Categories storage before: " << as_integer(s) << std::endl;
      assign<bool, T>(&Categories, &C, GrB_NULL, true, GrB_ALL, numFeatures, desc); // Non-zero = true, zero = false
      // CHECK(Categories.getStorage(&s));
      // std::cout << "Categories storage after: " << as_integer(s) << std::endl;
      // std::cout << "....." << std::endl;

      // CHECK(Categories.print());
      Categories_ind_size = numFeatures;
      CHECK(Categories.extractTuples(&Categories_val, &Categories_ind_size));

      gpu_check.Stop();
      gpu_check_time += gpu_check.ElapsedMillis();
      std::cout << "Test passed" << std::endl;
      std::cout << "Check time: " << gpu_check_time << std::endl;

      // Check correctness (not timed)
      // for (int i = 0; i < Categories_ind_size; i++) { // Sparse version
      //   Index idx = Categories_ind[i];
      BOOST_ASSERT_LIST(TrueCategories, Categories_val, numFeatures);
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------
    return GrB_SUCCESS;
}

template <typename T>
Info dnnCpu
(
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
  return SimpleReferenceDnn<T>();
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_SSSP_HPP_
