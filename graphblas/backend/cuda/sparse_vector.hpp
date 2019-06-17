#ifndef GRAPHBLAS_BACKEND_CUDA_SPARSE_VECTOR_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPARSE_VECTOR_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <unordered_set>
#include <algorithm>

#include "graphblas/types.hpp"
#include "graphblas/util.hpp"

namespace graphblas {
namespace backend {

template <typename T>
class DenseVector;

template <typename T>
class SparseVector {
 public:
  SparseVector()
      : nsize_(0), nvals_(0), h_ind_(NULL), h_val_(NULL),
        d_ind_(NULL), d_val_(NULL), need_update_(0) {}

  explicit SparseVector(Index nsize)
      : nsize_(nsize), nvals_(0), h_ind_(NULL), h_val_(NULL),
        d_ind_(NULL), d_val_(NULL), need_update_(0) {
    allocate();
  }

  // Need to write Default Destructor
  ~SparseVector();

  // C API Methods
  Info nnew(Index nsize);
  Info dup(const SparseVector* rhs);
  Info clear();
  inline Info size(Index* nsize_t) const;
  inline Info nvals(Index* nvals_t) const;
  template <typename BinaryOpT>
  Info build(const std::vector<Index>* indices,
             const std::vector<T>*     values,
             Index                     nvals,
             BinaryOpT                 dup);
  Info build(const std::vector<T>* values,
             Index                 nvals);
  Info build(Index* indices,
             T*     values,
             Index  nvals);
  Info setElement(T val,
                  Index index);
  Info extractElement(T*    val,
                      Index index);
  Info extractTuples(std::vector<Index>* indices,
                     std::vector<T>*     values,
                     Index*              n);

  // handy methods
  const T& operator[](Index ind);
  Info resize(Index nsize);
  Info fill(Index vals);
  Info print(bool force_update = false);
  Info countUnique(Index* count);
  Info allocateCpu();
  Info allocateGpu();
  Info allocate();
  Info cpuToGpu();
  Info gpuToCpu(bool force_update = false);
  Info swap(SparseVector* rhs);

 private:
  Index  nsize_;  // 5 ways to set: (1) Vector (2) nnew (3) dup (4) resize
                  //                (5) allocate
  Index  nvals_;  // 4 ways to set: (1) Vector (2) dup (3) build (4) resize
  Index* h_ind_;
  T*     h_val_;
  Index* d_ind_;
  T*     d_val_;

  bool  need_update_;  // set to true by changing SparseVector
                       // set to false by gpuToCpu()
};

template <typename T>
SparseVector<T>::~SparseVector() {
  if (h_ind_ != NULL) free(h_ind_);
  if (h_val_ != NULL) free(h_val_);
  if (d_ind_ != NULL) CUDA_CALL(cudaFree(d_ind_));
  if (d_ind_ != NULL) CUDA_CALL(cudaFree(d_val_));
}

template <typename T>
Info SparseVector<T>::nnew(Index nsize) {
  nsize_ = nsize;
  CHECK(allocate());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::dup(const SparseVector* rhs) {
  nvals_ = rhs->nvals_;
  nsize_ = rhs->nsize_;

  if (d_ind_ == NULL && h_ind_ == NULL && d_val_ == NULL && h_val_ == NULL)
    CHECK(allocate());

  CUDA_CALL(cudaMemcpy(d_ind_, rhs->d_ind_, nsize_*sizeof(Index),
      cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(d_val_, rhs->d_val_, nsize_*sizeof(T),
      cudaMemcpyDeviceToDevice));

  need_update_ = true;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::clear() {
  nvals_ = 0;
  return GrB_SUCCESS;
}

template <typename T>
inline Info SparseVector<T>::size(Index* nsize_t) const {
  *nsize_t = nsize_;
  return GrB_SUCCESS;
}

template <typename T>
inline Info SparseVector<T>::nvals(Index* nvals_t) const {
  *nvals_t = nvals_;
  return GrB_SUCCESS;
}

template <typename T>
template <typename BinaryOpT>
Info SparseVector<T>::build(const std::vector<Index>* indices,
                            const std::vector<T>*     values,
                            Index                     nvals,
                            BinaryOpT                 dup) {
  if (nvals > nsize_) {
    std::cout << "SpVec Build with indices greater than nsize_\n";
    std::cout << "Error: Feature not implemented yet!\n";
    return GrB_PANIC;
  }
  if (nvals_ > 0)
    return GrB_OUTPUT_NOT_EMPTY;
  if (h_ind_ == NULL || h_val_ == NULL || d_ind_ == NULL || d_val_ == NULL) {
    std::cout << "Error: SpVec Uninitialized object!\n";
    return GrB_UNINITIALIZED_OBJECT;
  }

  nvals_ = nvals;

  for (Index i = 0; i < nvals; i++) {
    h_ind_[i] = (*indices)[i];
    h_val_[i] = (*values) [i];
  }

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::build(const std::vector<T>* values,
                            Index                 nvals) {
  std::cout << "Sparse Build with dense input\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::build(Index* indices,
                            T*     values,
                            Index  nvals) {
  d_ind_ = indices;
  d_val_ = values;
  nvals_ = nvals;

  need_update_ = true;
  CHECK(allocateCpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::setElement(T val, Index index) {
  CHECK(gpuToCpu());
  h_ind_[nvals_] = index;
  h_val_[nvals_] = val;
  nvals_++;
  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::extractElement(T*    val, Index index) {
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::extractTuples(std::vector<Index>* indices,
                                    std::vector<T>*     values,
                                    Index*              n) {
  CHECK(gpuToCpu());
  indices->clear();
  values->clear();

  if (*n > nvals_) {
    std::cout << *n << " > " << nvals_ << std::endl;
    std::cout << "Error: *n > nvals!\n";
    return GrB_UNINITIALIZED_OBJECT;
  } else if (*n < nvals_) {
    std::cout << *n << " < " << nvals_ << std::endl;
    std::cout << "Error: *n < nvals!\n";
    return GrB_INSUFFICIENT_SPACE;
  }

  for (Index i = 0; i < *n; i++) {
    indices->push_back(h_ind_[i]);
    values->push_back(h_val_[i]);
  }

  return GrB_SUCCESS;
}

// If ind is found, then return the value at that ind
// Else if ind is not found, return 0 of type T
template <typename T>
const T& SparseVector<T>::operator[](Index ind) {
  gpuToCpu();
  if (ind >= nvals_) std::cout << "Error: Spvec Index out of bounds!\n";

  for (Index i = 0; i < nvals_; i++)
    if (h_ind_[i] == ind)
      return h_val_[i];
  return T(0);
}

// Clears and reallocates from nsize_ x 1 to nsize x 1
template <typename T>
Info SparseVector<T>::resize(Index nsize) {
  Index* h_temp_ind = h_ind_;
  T*     h_temp_val = h_val_;
  Index* d_temp_ind = d_ind_;
  T*     d_temp_val = d_val_;

  // Compute how much to copy
  Index to_copy = min(nsize, nvals_);

  nsize_ = nsize;
  h_ind_ = reinterpret_cast<Index*>(malloc( nsize_*sizeof(Index)));
  h_val_ = reinterpret_cast<T*>(malloc( (nsize_+1)*sizeof(T)));
  if (h_temp_ind != NULL)
    memcpy(h_ind_, h_temp_ind, to_copy*sizeof(Index));
  if (h_temp_val != NULL)
    memcpy(h_val_, h_temp_val, to_copy*sizeof(T));

  CUDA_CALL(cudaMalloc(&d_ind_, nsize_*sizeof(Index)));
  CUDA_CALL(cudaMalloc(&d_val_, (nsize_+1)*sizeof(T)));
  printMemory("SpVec");
  if (d_temp_ind != NULL)
    CUDA_CALL(cudaMemcpy(d_ind_, d_temp_ind, to_copy*sizeof(Index),
        cudaMemcpyDeviceToDevice));
  if (d_temp_val != NULL)
    CUDA_CALL(cudaMemcpy(d_val_, d_temp_val, to_copy*sizeof(T),
        cudaMemcpyDeviceToDevice));
  nvals_ = to_copy;

  free(h_temp_ind);
  free(h_temp_val);
  CUDA_CALL(cudaFree(d_temp_ind));
  CUDA_CALL(cudaFree(d_temp_val));

  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::fill(Index nvals) {
  for (Index i = 0; i < nvals; i++)
    h_val_[i] = i;

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::print(bool force_update) {
  std::cout << nsize_ << " x 1:" << nvals_ << " nnz\n";
  CUDA_CALL(cudaDeviceSynchronize());
  CHECK(gpuToCpu(force_update));
  printArray("ind", h_ind_, std::min(nvals_, 40));
  printArray("val", h_val_, std::min(nvals_, 40));
  if (nvals_ == 0)
    std::cout << "Error: SparseVector is empty!\n";
  return GrB_SUCCESS;
}

// Count number of unique numbers
template <typename T>
Info SparseVector<T>::countUnique(Index* count) {
  CHECK(gpuToCpu());
  std::unordered_set<Index> unique;
  for (Index block = 0; block < nvals_; block++) {
    if (unique.find(h_val_[block]) == unique.end()) {
      unique.insert(h_val_[block]);
    }
  }
  *count = unique.size();

  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::allocateCpu() {
  // Host malloc
  if (nsize_ != 0 && h_ind_ == NULL && h_val_ == NULL) {
    h_ind_ = reinterpret_cast<Index*>(malloc(nsize_*sizeof(Index)));
    h_val_ = reinterpret_cast<T*>(malloc((nsize_+1)*sizeof(T)));
  } else {
    // std::cout << "Error: SpVec Host allocation unsuccessful!\n";
  }

  if (nsize_ != 0 && (h_ind_ == NULL || h_val_ == NULL)) {
    std::cout << "Error: SpVec Out of memory!\n";
    // return GrB_OUT_OF_MEMORY;
  }

  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::allocateGpu() {
  // GPU malloc
  if (nsize_ != 0 && d_ind_ == NULL && d_val_ == NULL) {
    CUDA_CALL(cudaMalloc(&d_ind_, nsize_*sizeof(Index)));
    CUDA_CALL(cudaMalloc(&d_val_, (nsize_+1)*sizeof(T)));
    printMemory("d_ind, d_val");
  } else {
    // std::cout << "Error: SpVec Device allocation unsuccessful!\n";
  }

  if (nsize_ != 0 && (d_ind_ == NULL || d_val_ == NULL)) {
    std::cout << "Error: SpVec Out of memory!\n";
    // return GrB_OUT_OF_MEMORY;
  }

  return GrB_SUCCESS;
}

// Allocate just enough (different from CPU impl since kcap_ratio=1.)
template <typename T>
Info SparseVector<T>::allocate() {
  CHECK(allocateCpu());
  CHECK(allocateGpu());
  return GrB_SUCCESS;
}

// Copies graph to GPU
template <typename T>
Info SparseVector<T>::cpuToGpu() {
  CUDA_CALL(cudaMemcpy(d_ind_, h_ind_, nvals_*sizeof(Index),
      cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_val_, h_val_, nvals_*sizeof(T),
      cudaMemcpyHostToDevice));
  return GrB_SUCCESS;
}

// Copies graph to CPU
template <typename T>
Info SparseVector<T>::gpuToCpu(bool force_update) {
  if (need_update_ || force_update) {
    CUDA_CALL(cudaMemcpy(h_ind_, d_ind_, nvals_*sizeof(Index),
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_val_, d_val_, nvals_*sizeof(T),
        cudaMemcpyDeviceToHost));
  }
  need_update_ = false;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseVector<T>::swap(SparseVector* rhs) {  // NOLINT(build/include_what_you_use)
  // Swap scalars
  std::swap(nsize_, rhs->nsize_);
  std::swap(nvals_, rhs->nvals_);

  // Swap CPU pointers
  std::swap(h_ind_, rhs->h_ind_);
  std::swap(h_val_, rhs->h_val_);

  // Swap GPU pointers
  std::swap(d_ind_, rhs->d_ind_);
  std::swap(d_val_, rhs->d_val_);

  std::swap(need_update_, rhs->need_update_);
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPARSE_VECTOR_HPP_
