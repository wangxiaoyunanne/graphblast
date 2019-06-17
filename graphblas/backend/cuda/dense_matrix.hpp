#ifndef GRAPHBLAS_BACKEND_CUDA_DENSE_MATRIX_HPP_
#define GRAPHBLAS_BACKEND_CUDA_DENSE_MATRIX_HPP_

#include <vector>
#include <iostream>
#include <algorithm>

#include "graphblas/types.hpp"

namespace graphblas {
namespace backend {

template <typename T>
class SparseMatrix;

template <typename T>
class DenseMatrix {
 public:
  DenseMatrix()
      : nrows_(0), ncols_(0), nvals_(0),
        h_denseVal_(NULL), d_denseVal_(NULL), need_update_(0) {}
  explicit DenseMatrix(Index nrows, Index ncols)
      : nrows_(nrows), ncols_(ncols), nvals_(nrows*ncols),
        h_denseVal_(NULL), d_denseVal_(NULL), need_update_(0) {}

  ~DenseMatrix() {}

  // C API Methods
  Info nnew(Index nrows, Index ncols);
  Info dup(const DenseMatrix* rhs);
  Info clear();
  Info nrows(Index* nrows_t) const;
  Info ncols(Index* ncols_t) const;
  Info nvals(Index* nvals_t) const;
  template <typename BinaryOpT>
  Info build(const std::vector<Index>* row_indices,
             const std::vector<Index>* col_indices,
             const std::vector<T>*     values,
             Index                     nvals,
             BinaryOpT                 dup);
  Info build(const std::vector<T>* values,
             Index                 nvals);
  Info setElement(Index row_index, Index col_index);
  Info extractElement(T* val, Index row_index, Index col_index);
  Info extractTuples(std::vector<Index>* row_indices,
                     std::vector<Index>* col_indices,
                     std::vector<T>*     values,
                     Index*              n);
  Info extractTuples(std::vector<T>* values,
                     Index*          n);

  // Handy methods
  const T operator[](Index ind);
  Info print(bool force_update);
  Info setNrows(Index nrows);
  Info setNcols(Index ncols);
  Info resize(Index nrows, Index ncols);
  template <typename U>
  Info fill(Index axis, Index nvals, U start);
  template <typename U>
  Info fillAscending(Index axis, Index nvals, U start);
  Info swap(DenseMatrix* rhs);

 private:
  Info allocate();
  Info printDense() const;
  Info cpuToGpu();
  Info gpuToCpu(bool force_update = false);

 private:
  Index nrows_;
  Index ncols_;
  Index nvals_;

  // Dense format
  T*    h_denseVal_;
  T*    d_denseVal_;

  bool  need_update_;
};

template <typename T>
Info DenseMatrix<T>::nnew(Index nrows, Index ncols) {
  nrows_ = nrows;
  ncols_ = ncols;
  nvals_ = nrows_*ncols_;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::dup(const DenseMatrix* rhs) {
  if (nrows_ != rhs->nrows_) return GrB_DIMENSION_MISMATCH;
  if (ncols_ != rhs->ncols_) return GrB_DIMENSION_MISMATCH;
  nvals_ = rhs->nvals_;

  Info err = allocate();
  if (err != GrB_SUCCESS) return err;

  CUDA_CALL(cudaMemcpy(d_denseVal_, rhs->d_denseVal_, nvals_*sizeof(T),
      cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaDeviceSynchronize());

  need_update_ = true;
  return err;
}

template <typename T>
Info DenseMatrix<T>::clear() {
  if (h_denseVal_) free(h_denseVal_);
  if (d_denseVal_) CUDA_CALL(cudaFree(d_denseVal_));
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::nrows(Index* nrows_t) const {
  *nrows_t = nrows_;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::ncols(Index* ncols_t) const {
  *ncols_t = ncols_;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::nvals(Index* nvals_t) const {
  *nvals_t = nvals_;
  return GrB_SUCCESS;
}

template <typename T>
template <typename BinaryOpT>
Info DenseMatrix<T>::build(const std::vector<Index>* row_indices,
                           const std::vector<Index>* col_indices,
                           const std::vector<T>*     values,
                           Index                     nvals,
                           BinaryOpT                 dup) {
  std::cout << "Dense Matrix Build from sparse input\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::build(const std::vector<T>* values,
                           Index                 nvals) {
  if (nvals > nvals_) return GrB_DIMENSION_MISMATCH;
  CHECK(allocate());

  // Host copy
  for (Index i = 0; i < nvals_; i++)
    h_denseVal_[i] = (*values)[i];

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::setElement(Index row_index, Index col_index) {
  std::cout << "Dense Matrix setElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::extractElement(T* val, Index row_index, Index col_index) {
  std::cout << "Dense Matrix setElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::extractTuples(std::vector<Index>* row_indices,
                                   std::vector<Index>* col_indices,
                                   std::vector<T>*     values,
                                   Index*              n) {
  std::cout << "Dense Matrix extractTuples\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::extractTuples(std::vector<T>* values, Index* n) {
  Info err = gpuToCpu();
  values->clear();
  if (*n > nvals_) {
    err = GrB_UNINITIALIZED_OBJECT;
    *n  = nvals_;
  } else if (*n < nvals_) {
    err = GrB_INSUFFICIENT_SPACE;
  }

  for (Index i = 0; i < *n; i++)
    values->push_back(h_denseVal_[i]);

  return err;
}

template <typename T>
const T DenseMatrix<T>::operator[](Index ind) {
  return T(0);
}

template <typename T>
Info DenseMatrix<T>::print(bool force_update) {
  std::cout << nrows_ << " x " << nrows_ << ": " << nrows_*ncols_ << " nnz\n";
  CHECK(gpuToCpu(force_update));
  printArray("denseVal", h_denseVal_, std::min(nvals_, 40));
  printDense();
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::setNrows(Index nrows) {
  nrows_ = nrows;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::setNcols(Index ncols) {
  ncols_ = ncols;
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::resize(Index nrows, Index ncols) {
  return GrB_SUCCESS;
}

template <typename T>
template <typename U>
Info DenseMatrix<T>::fill(Index axis, Index nvals, U start) {
  std::cout << "Dense Matrix fill\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
template <typename U>
Info DenseMatrix<T>::fillAscending(Index axis, Index nvals, U start) {
  std::cout << "Dense Matrix fillAscending\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::swap(DenseMatrix* rhs) {
  // Swap scalars
  std::swap(nrows_, rhs->nrows_);
  std::swap(ncols_, rhs->ncols_);
  std::swap(nvals_, rhs->nvals_);

  // Swap CPU pointers
  std::swap(h_denseVal_, rhs->h_denseVal_);
  
  // Swap GPU pointers
  std::swap(d_denseVal_, rhs->d_denseVal_);

  std::swap(need_update_, rhs->need_update_);
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::allocate() {
  // Host alloc
  if (nvals_ != 0 && h_denseVal_ == NULL)
    h_denseVal_ = reinterpret_cast<T*>(malloc(nvals_*sizeof(T)));

  for (Index i = 0; i < nvals_; i++)
    h_denseVal_[i] = (T) 0;

  if (nvals_ != 0 && d_denseVal_ == NULL)
    CUDA_CALL(cudaMalloc(&d_denseVal_, nvals_*sizeof(T)));

  if (h_denseVal_ == NULL || d_denseVal_ == NULL) return GrB_OUT_OF_MEMORY;

  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::printDense() const {
  Index row_length = std::min(20, nrows_);
  Index col_length = std::min(20, ncols_);

  for (Index row = 0; row < row_length; row++) {
    for (Index col = 0; col < col_length; col++) {
      // Print row major order matrix in row major order by default
      // if( major_type_ == GrB_ROWMAJOR )
      // {
        if (h_denseVal_[row*ncols_+col] != 0.0)
          std::cout << "x ";
        else
          std::cout << "0 ";
      // Print column major order matrix in row major order (Transposition)
      /*}
      else if (major_type_ == GrB_COLMAJOR )
      {
        if( h_denseVal_[col*nrows_+row]!=0.0 ) std::cout << "x ";
        else std::cout << "0 ";
      }*/
    }
    std::cout << std::endl;
  }
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::cpuToGpu() {
  CUDA_CALL(cudaMemcpy(d_denseVal_, h_denseVal_, nvals_*sizeof(T),
      cudaMemcpyHostToDevice));
  return GrB_SUCCESS;
}

template <typename T>
Info DenseMatrix<T>::gpuToCpu(bool force_update) {
  if (need_update_ || force_update)
    CUDA_CALL(cudaMemcpy(h_denseVal_, d_denseVal_, nvals_*sizeof(T),
        cudaMemcpyDeviceToHost));
  need_update_ = false;
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_DENSE_MATRIX_HPP_
