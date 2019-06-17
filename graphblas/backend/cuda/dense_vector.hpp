#ifndef GRAPHBLAS_BACKEND_CUDA_DENSE_VECTOR_HPP_
#define GRAPHBLAS_BACKEND_CUDA_DENSE_VECTOR_HPP_

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
class SparseVector;

template <typename T>
class DenseVector {
 public:
  DenseVector()
      : nvals_(0), nnz_(0), h_val_(NULL), d_val_(NULL), need_update_(0) {}

  explicit DenseVector(Index nsize)
      : nvals_(nsize), nnz_(0), h_val_(NULL), d_val_(NULL), need_update_(0) {
    allocate();
  }

  // Need to write Default Destructor
  ~DenseVector();

  // C API Methods
  Info nnew(Index nsize);
  Info dup(const DenseVector* rhs);
  Info clear();
  inline Info size(Index* nsize_) const;
  inline Info nvals(Index* nvals_) const;
  inline Info nnz(Index* nnz_) const;
  Info computeNnz(Index* nnz, T identity, Descriptor* desc);
  template <typename BinaryOpT>
  Info build(const std::vector<Index>* indices,
             const std::vector<T>*     values,
             Index                     nvals,
             BinaryOpT                 dup);
  Info build(const std::vector<T>* values,
             Index                 nvals);
  Info build(T*    values,
             Index nvals);
  Info setElement(T val, Index index);
  Info extractElement(T* val, Index index);
  Info extractTuples(std::vector<Index>* indices,
                     std::vector<T>*     values,
                     Index*              n);
  Info extractTuples(std::vector<T>* values,
                     Index*          n);

  // handy methods
  const T& operator[](Index ind);
  Info resize(Index nsize);
  Info fill(T val);
  Info fillAscending(Index vals);
  Info print(bool force_update = false);
  Info countUnique(Index* count);
  Info allocateCpu();
  Info allocateGpu();
  Info allocate();
  Info cpuToGpu();
  Info gpuToCpu(bool force_update = false);
  Info swap(DenseVector* rhs);

 private:
  // Note nsize_ is understood to be the same as nvals_, so it is omitted
  Index nvals_;  // 6 ways to set: (1) Vector (2) nnew (3) dup (4) build
                 //                (5) resize (6) allocate
  Index nnz_;
  T*    h_val_;
  T*    d_val_;

  bool  need_update_;  // set to true by changing DenseVector
                       // set to false by gpuToCpu()
};

template <typename T>
DenseVector<T>::~DenseVector() {
  if (h_val_ != NULL) free(h_val_);
  if (d_val_ != NULL) CUDA_CALL(cudaFree(d_val_));
}

template <typename T>
Info DenseVector<T>::nnew(Index nsize) {
  nvals_ = nsize;
  CHECK(allocate());
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::dup(const DenseVector* rhs) {
  nvals_ = rhs->nvals_;

  if (d_val_ == NULL || h_val_ == NULL)
    CHECK(allocate());

  CUDA_CALL(cudaMemcpy(d_val_, rhs->d_val_, nvals_*sizeof(T),
      cudaMemcpyDeviceToDevice));

  need_update_ = true;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::clear() {
  CHECK(fill((T)0));
  return GrB_SUCCESS;
}

template <typename T>
inline Info DenseVector<T>::size(Index* nsize_t) const {
  *nsize_t = nvals_;
  return GrB_SUCCESS;
}

template <typename T>
inline Info DenseVector<T>::nvals(Index* nvals_t) const {
  *nvals_t = nvals_;
  return GrB_SUCCESS;
}

template <typename T>
inline Info DenseVector<T>::nnz(Index* nnz_t) const {
  *nnz_t = nnz_;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::computeNnz(Index* nnz_t, T identity, Descriptor* desc) {
  // Nasty bug if you pass in the length of array rather than size of array in
  // bytes!
  CHECK(desc->resize((nvals_+1)*sizeof(T), "buffer"));

	// Get descriptor parameters for nthreads
	Desc_value nt_mode;
	CHECK(desc->get(GrB_NT, &nt_mode));
	const int nt = static_cast<int>(nt_mode);
	dim3 NT, NB;
	NT.x = nt;
	NT.y = 1;
	NT.z = 1;
	NB.x = (nvals_+nt-1)/nt;
	NB.y = 1;
	NB.z = 1;

  countZero<<<NB, NT>>>(reinterpret_cast<Index*>(desc->d_buffer_), identity,
      d_val_, nvals_);
  Index* d_nnz = reinterpret_cast<Index*>(desc->d_buffer_)+nvals_;

  size_t temp_storage_bytes = 0;
  plus<Index> op;

  if (nvals_ == 0)
    return GrB_INVALID_OBJECT;

  if (desc->debug()) {
    printDevice("zeros", reinterpret_cast<Index*>(desc->d_buffer_), nvals_);
    std::cout << "nvals_: " << nvals_ << std::endl;
  }

  CUDA_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes,
      reinterpret_cast<Index*>(desc->d_buffer_), d_nnz, nvals_, op, 0));

  CHECK(desc->resize(temp_storage_bytes, "temp"));
  if (desc->debug()) {
    std::cout << temp_storage_bytes << " <= " << desc->d_temp_size_ <<
        std::endl;
  }

  CUDA_CALL(cub::DeviceReduce::Reduce(desc->d_temp_, temp_storage_bytes,
      reinterpret_cast<Index*>(desc->d_buffer_), d_nnz, nvals_, op, 0));
  CUDA_CALL(cudaMemcpy(&nnz_, d_nnz, sizeof(Index), cudaMemcpyDeviceToHost));

  nnz_ = nvals_ - nnz_;
  *nnz_t = nnz_;
  return GrB_SUCCESS;
}

template <typename T>
template <typename BinaryOpT>
Info DenseVector<T>::build(const std::vector<Index>* indices,
                           const std::vector<T>*     values,
                           Index                     nvals,
                           BinaryOpT                 dup) {
  std::cout << "DeVec Build Using Sparse Indices\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::build(const std::vector<T>* values,
                           Index                 nvals) {
  if (nvals > nvals_)
    return GrB_INDEX_OUT_OF_BOUNDS;
  if (d_val_ == NULL || h_val_ == NULL)
    return GrB_UNINITIALIZED_OBJECT;

  for (Index i = 0; i < nvals; i++)
    h_val_[i] = (*values)[i];

  CHECK(cpuToGpu());

  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::build(T*    values,
                           Index nvals) {
  d_val_       = values;
  nvals_       = nvals;
  need_update_ = true;

  CHECK(allocateCpu());
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::setElement(T val, Index index) {
  CHECK(gpuToCpu());
  h_val_[index] = val;
  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::extractElement(T* val, Index index) {
  std::cout << "DeVec ExtractElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::extractTuples(std::vector<Index>* indices,
                                   std::vector<T>*     values,
                                   Index*              n) {
  std::cout << "DeVec ExtractTuples into Sparse Indices\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::extractTuples(std::vector<T>* values, Index* n) {
  CHECK(gpuToCpu());
  values->clear();

  if (*n > nvals_) {
    std::cout << *n << " > " << nvals_ << std::endl;
    std::cout << "Error: DeVec Too many tuples requested!\n";
    return GrB_UNINITIALIZED_OBJECT;
  }
  if (*n < nvals_) {
    std::cout << *n << " < " << nvals_ << std::endl;
    std::cout << "Error: DeVec Insufficient space!\n";
    return GrB_INSUFFICIENT_SPACE;
  }

  for (Index i = 0; i < *n; i++)
    values->push_back(h_val_[i]);

  return GrB_SUCCESS;
}

template <typename T>
const T& DenseVector<T>::operator[](Index ind) {
  CHECKVOID(gpuToCpu());
  if (ind >= nvals_) std::cout << "Error: Index out of bounds!\n";
  return h_val_[ind];
}

// Copies the val to arrays kresize_ratio x bigger than capacity
template <typename T>
Info DenseVector<T>::resize(Index nsize) {
  T* h_tempVal = h_val_;
  T* d_tempVal = d_val_;

  // Compute how much to copy
  Index to_copy = std::min(nsize, nvals_);

  nvals_ = nsize;
  h_val_ = reinterpret_cast<T*>(malloc(nvals_*sizeof(T)));
  if (h_tempVal != NULL)
    memcpy(h_val_, h_tempVal, to_copy*sizeof(T));

  CUDA_CALL(cudaMalloc(&d_val_, nvals_*sizeof(T)));
  if (d_tempVal != NULL)
    CUDA_CALL(cudaMemcpy(d_val_, d_tempVal, to_copy*sizeof(T),
        cudaMemcpyDeviceToDevice));
  nvals_ = nsize;

  free(h_tempVal);
  CUDA_CALL(cudaFree(d_tempVal));

  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::fill(T val) {
  for (Index i = 0; i < nvals_; i++ )
    h_val_[i] = val;

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::fillAscending(Index nvals) {
  for (Index i = 0; i < nvals_; i++)
    h_val_[i] = i;

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::print(bool force_update) {
  std::cout << nvals_ << " x 1: " << nvals_ << " nnz\n";
  CUDA_CALL(cudaDeviceSynchronize());
  CHECK(gpuToCpu(force_update));
  printArray("val", h_val_, std::min(nvals_, 40));
  return GrB_SUCCESS;
}

// Count number of unique numbers
template <typename T>
Info DenseVector<T>::countUnique(Index* count) {
  CHECK(gpuToCpu());
  std::unordered_set<Index> unique;
  for (Index block = 0; block < nvals_; block++) {
    if (unique.find(h_val_[block]) == unique.end())
      unique.insert(h_val_[block]);
  }
  *count = unique.size();

  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::allocateCpu() {
  // Host malloc
  if (nvals_ > 0 && h_val_ == NULL) {
    h_val_ = reinterpret_cast<T*>(malloc(nvals_*sizeof(T)));
  } else {
    // std::cout << "Error: DeVec Host allocation unsuccessful!\n";
  }

  if (nvals_ > 0 && h_val_ == NULL) {
    std::cout << "Error: CPU DeVec Out of memory!\n";
    // return GrB_OUT_OF_MEMORY;
  }

  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::allocateGpu() {
  // GPU malloc
  if (nvals_ > 0 && d_val_ == NULL) {
    CUDA_CALL(cudaMalloc(&d_val_, nvals_*sizeof(T)));
    printMemory("DeVec");
  } else {
    // std::cout << "Error: DeVec Device allocation unsuccessful!\n";
  }

  if (nvals_ > 0 && d_val_ == NULL) {
    std::cout << "Error: DeVec Out of memory!\n";
    // return GrB_OUT_OF_MEMORY;
  }

  return GrB_SUCCESS;
}

// Allocate just enough (different from CPU impl since kcap_ratio=1.)
template <typename T>
Info DenseVector<T>::allocate() {
  CHECK(allocateCpu());
  CHECK(allocateGpu());
  return GrB_SUCCESS;
}

// Copies graph to GPU
template <typename T>
Info DenseVector<T>::cpuToGpu() {
  CUDA_CALL(cudaMemcpy(d_val_, h_val_, nvals_*sizeof(T),
      cudaMemcpyHostToDevice));
  return GrB_SUCCESS;
}

// Copies graph to CPU
template <typename T>
Info DenseVector<T>::gpuToCpu(bool force_update) {
  if (need_update_ || force_update)
    CUDA_CALL(cudaMemcpy(h_val_, d_val_, nvals_*sizeof(T),
        cudaMemcpyDeviceToHost));
  need_update_ = false;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseVector<T>::swap(DenseVector* rhs) {  // NOLINT(build/include_what_you_use)
  // Swap scalars
  std::swap(nvals_, rhs->nvals_);

  // Swap CPU pointers
  std::swap(h_val_, rhs->h_val_);

  // Swap GPU pointers
  std::swap(d_val_, rhs->d_val_);

  std::swap(need_update_, rhs->need_update_);
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_DENSE_VECTOR_HPP_
