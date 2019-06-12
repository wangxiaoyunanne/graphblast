#ifndef GRAPHBLAS_ALGORITHM_TEST_DNN_HPP_
#define GRAPHBLAS_ALGORITHM_TEST_DNN_HPP_

namespace graphblas {
namespace algorithm {

// A simple CPU-based reference SSSP ranking implementation
template <typename T>
Info SimpleReferenceDnn() {
  return GrB_SUCCESS;
}
}  // namespace algorithm
}  // namespace graphblas

#endif  // GRAPHBLAS_ALGORITHM_TEST_DNN_HPP_
