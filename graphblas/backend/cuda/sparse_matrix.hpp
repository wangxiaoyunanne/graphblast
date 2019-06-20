#ifndef GRAPHBLAS_BACKEND_CUDA_SPARSE_MATRIX_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPARSE_MATRIX_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>

#include "graphblas/util.hpp"

namespace graphblas {
namespace backend {

template <typename T>
class DenseMatrix;

template <typename T>
class Vector;

template <typename T>
class SparseMatrix {
 public:
  SparseMatrix()
      : nrows_(0), ncols_(0), nvals_(0), ncapacity_(0), nempty_(0),
        h_csrRowPtr_(NULL), h_csrColInd_(NULL), h_csrVal_(NULL),
        h_cscColPtr_(NULL), h_cscRowInd_(NULL), h_cscVal_(NULL),
        d_csrRowPtr_(NULL), d_csrColInd_(NULL), d_csrVal_(NULL),
        d_cscColPtr_(NULL), d_cscRowInd_(NULL), d_cscVal_(NULL),
        need_update_(0), symmetric_(0) {
    format_ = getEnv("GRB_SPARSE_MATRIX_FORMAT", GrB_SPARSE_MATRIX_CSRCSC);
  }

  explicit SparseMatrix(Index nrows, Index ncols)
      : nrows_(nrows), ncols_(ncols), nvals_(0), ncapacity_(0), nempty_(0),
        h_csrRowPtr_(NULL), h_csrColInd_(NULL), h_csrVal_(NULL),
        h_cscColPtr_(NULL), h_cscRowInd_(NULL), h_cscVal_(NULL),
        d_csrRowPtr_(NULL), d_csrColInd_(NULL), d_csrVal_(NULL),
        d_cscColPtr_(NULL), d_cscRowInd_(NULL), d_cscVal_(NULL),
        need_update_(0), symmetric_(0) {
    format_ = getEnv("GRB_SPARSE_MATRIX_FORMAT", GrB_SPARSE_MATRIX_CSRCSC);
  }

  ~SparseMatrix();

  // C API Methods
  Info nnew(Index nrows, Index ncols);
  Info dup(const SparseMatrix* rhs);
  Info clear();     // 1 way to free: (1) clear
  Info nrows(Index* nrows_t) const;
  Info ncols(Index* ncols_t) const;
  Info nvals(Index* nvals_t) const;
  template <typename BinaryOpT>
  Info build(const std::vector<Index>* row_indices,
             const std::vector<Index>* col_indices,
             const std::vector<T>*     values,
             Index                     nvals,
             BinaryOpT                 dup,
             char*                     dat_name);
  Info build(char* dat_name);
  Info build(const std::vector<T>* values,
             Index                 nvals);
  Info build(Index* row_ptr,
             Index* col_ind,
             T*     values,
             Index  nvals);
  Info setElement(Index row_index,
                  Index col_index);
  Info extractElement(T*    val,
                      Index row_index,
                      Index col_index);
  Info extractTuples(std::vector<Index>* row_indices,
                     std::vector<Index>* col_indices,
                     std::vector<T>*     values,
                     Index*              n);
  Info extractTuples(std::vector<T>* values,
                     Index*          n);

  // Handy methods
  const T operator[](Index ind);
  Info print(bool force_update);
  Info check();
  Info setNrows(Index nrows);
  Info setNcols(Index ncols);
  Info setNvals(Index nvals);
  Info getFormat(SparseMatrixFormat* format) const;
  Info getSymmetry(bool* symmetry) const;
  Info resize(Index nrows, Index ncols);
  template <typename U>
  Info fill(Index axis, Index nvals, U start);
  template <typename U>
  Info fillAscending(Index axis, Index nvals, U start);
  Info swap(SparseMatrix* rhs);
  Info rebuild(T identity, Descriptor* desc);

 private:
  Info allocateCpu();
  Info allocateGpu();
  Info allocate();  // 3 ways to allocate: (1) dup, (2) build, (3) spgemm
                    //                     (4) fill,(5) fillAscending
  Info printCSR(const char* str);  // private method for pretty printing
  Info printCSC(const char* str);
  Info cpuToGpu();
  Info gpuToCpu(bool force_update = false);
  Info syncCpu();   // synchronizes CSR and CSC representations
  Info validate();  // validates CSR structure on CPU

 private:
  const T kcap_ratio_    = 1.2f;  // Note: nasty bug if this is set to 1.f!
  const T kresize_ratio_ = 1.2f;

  Index nrows_;
  Index ncols_;
  Index nvals_;     // 3 ways to set: (1) dup (2) build (3) nnew
  Index ncapacity_;
  Index nempty_;

  Index* h_csrRowPtr_;  // CSR format
  Index* h_csrColInd_;
  T*     h_csrVal_;
  Index* h_cscColPtr_;  // CSC format
  Index* h_cscRowInd_;
  T*     h_cscVal_;

  Index* d_csrRowPtr_;  // GPU CSR format
  Index* d_csrColInd_;
  T*     d_csrVal_;
  Index* d_cscColPtr_;  // GPU CSC format
  Index* d_cscRowInd_;
  T*     d_cscVal_;

  // GPU variables
  bool need_update_;
  // Symmetric can mean 2 things:
  // 1) structure is symmetric i.e. d_cscColPtr_/d_cscRowInd_ are same as 
  //    d_csrRowPtr_/d_csrColInd_, so d_cscColPtr_ and d_cscRowInd_ can be freed
  // 2) values are symmetric i.e.
  //    [0 1 2]     [0 1 2]
  //    [1 0 0] vs. [3 0 4]
  //    [2 0 0]     [5 6 0]
  //
  //    In the case on the left, d_cscVal_ is same d_csrVal_ and can be freed,
  //    but this is not true for the case on the right and cannot be freed
  //
  // 3) user requests GRB_SPARSE_MATRIX_FORMAT = CSRONLY
  //
  // Current plan:
  // Detect Case 1 (when structure is symmetric)
  // -keep h_cscColPtr_ and h_cscRowInd_ (needed for csr2csc)
  // -free d_cscColPtr_ and d_cscRowInd_ (saves GPU memory)
  //
  // No specialization for Case 2 (when value is also symmetric)
  // -since cost savings of free'ing d_cscVal_ is not as big
  //
  // Memory efficient for Case 3
  // -free d_cscVal_ in addition to d_cscColPtr_ and d_cscRowInd_
  bool symmetric_;

  SparseMatrixFormat format_;
};

template <typename T>
SparseMatrix<T>::~SparseMatrix() {
  if (h_csrRowPtr_) free(h_csrRowPtr_);
  if (h_csrColInd_) free(h_csrColInd_);
  if (h_csrVal_   ) free(h_csrVal_);
  if (d_csrRowPtr_) CUDA_CALL(cudaFree(d_csrRowPtr_));
  if (d_csrColInd_) CUDA_CALL(cudaFree(d_csrColInd_));
  if (d_csrVal_   ) CUDA_CALL(cudaFree(d_csrVal_   ));

  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    if (h_cscColPtr_) free(h_cscColPtr_);
    if (h_cscRowInd_) free(h_cscRowInd_);
    if (h_cscVal_   ) free(h_cscVal_);
    if (d_cscVal_   ) CUDA_CALL(cudaFree(d_cscVal_));

    if (!symmetric_) {
      if (d_cscColPtr_) CUDA_CALL(cudaFree(d_cscColPtr_));
      if (d_cscRowInd_) CUDA_CALL(cudaFree(d_cscRowInd_));
    }
  }
}

template <typename T>
Info SparseMatrix<T>::nnew(Index nrows, Index ncols) {
  nrows_ = nrows;
  ncols_ = ncols;

  // We do not need to allocate here as in DenseVector, since nvals_ may not be
  // known
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::dup(const SparseMatrix* rhs) {
  if (nrows_ != rhs->nrows_) return GrB_DIMENSION_MISMATCH;
  if (ncols_ != rhs->ncols_) return GrB_DIMENSION_MISMATCH;
  nvals_     = rhs->nvals_;
  symmetric_ = rhs->symmetric_;
  format_    = rhs->format_;

  CHECK(allocate());

  CUDA_CALL(cudaMemcpy(d_csrRowPtr_, rhs->d_csrRowPtr_, (nrows_+1)*
      sizeof(Index), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(d_csrColInd_, rhs->d_csrColInd_, nvals_*sizeof(Index),
      cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(d_csrVal_,    rhs->d_csrVal_,    nvals_*sizeof(T),
      cudaMemcpyDeviceToDevice));

  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    CUDA_CALL(cudaMemcpy(d_cscVal_, rhs->d_cscVal_, nvals_*sizeof(T),
        cudaMemcpyDeviceToDevice));
    if (!symmetric_ && format_ == GrB_SPARSE_MATRIX_CSRCSC) {
      CUDA_CALL(cudaMemcpy(d_cscColPtr_, rhs->d_cscColPtr_, 
          (ncols_+1)*sizeof(Index), cudaMemcpyDeviceToDevice));
      CUDA_CALL(cudaMemcpy(d_cscRowInd_, rhs->d_cscRowInd_,
          nvals_*sizeof(Index), cudaMemcpyDeviceToDevice));
    }
  }

  need_update_ = true;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::clear() {
  nvals_     = 0;
  ncapacity_ = 0;

  if (h_csrRowPtr_) free(h_csrRowPtr_);
  if (h_csrColInd_) free(h_csrColInd_);
  if (h_csrVal_   ) free(h_csrVal_);
  if (d_csrRowPtr_) CUDA_CALL(cudaFree(d_csrRowPtr_));
  if (d_csrColInd_) CUDA_CALL(cudaFree(d_csrColInd_));
  if (d_csrVal_   ) CUDA_CALL(cudaFree(d_csrVal_));

  h_csrRowPtr_ = NULL;
  h_csrColInd_ = NULL;
  h_csrVal_    = NULL;
  d_csrRowPtr_ = NULL;
  d_csrColInd_ = NULL;
  d_csrVal_    = NULL;

  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    if (h_cscColPtr_) free(h_cscColPtr_);
    if (h_cscRowInd_) free(h_cscRowInd_);
    if (h_cscVal_   ) free(h_cscVal_);
    if (d_cscVal_   ) CUDA_CALL(cudaFree(d_cscVal_));

    if (!symmetric_) {
      if (d_cscColPtr_) CUDA_CALL(cudaFree(d_cscColPtr_));
      if (d_cscRowInd_) CUDA_CALL(cudaFree(d_cscRowInd_));
    }
  }
  return GrB_SUCCESS;
}

template <typename T>
inline Info SparseMatrix<T>::nrows(Index* nrows_t) const {
  *nrows_t = nrows_;
  return GrB_SUCCESS;
}

template <typename T>
inline Info SparseMatrix<T>::ncols(Index* ncols_t) const {
  *ncols_t = ncols_;
  return GrB_SUCCESS;
}

template <typename T>
inline Info SparseMatrix<T>::nvals(Index* nvals_t) const {
  *nvals_t = nvals_;
  return GrB_SUCCESS;
}

template <typename T>
template <typename BinaryOpT>
Info SparseMatrix<T>::build(const std::vector<Index>* row_indices,
                            const std::vector<Index>* col_indices,
                            const std::vector<T>*     values,
                            Index                     nvals,
                            BinaryOpT                 dup,
                            char*                     dat_name) {
  nvals_ = nvals;
  CHECK(allocateCpu());

  if (dat_name != NULL) {
    char* pch = strstr(dat_name, ".ud.");
    if (pch == NULL)
      symmetric_ = false;
    else
      symmetric_ = true;
  }

  coo2csr(h_csrRowPtr_, h_csrColInd_, h_csrVal_,
          *row_indices, *col_indices, *values, nrows_, ncols_);

  if (format_ == GrB_SPARSE_MATRIX_CSRONLY) {
    //if (symmetric_ || format_ == GrB_SPARSE_MATRIX_CSRONLY) {
    if (h_cscColPtr_ != NULL) free(h_cscColPtr_);
    if (h_cscRowInd_ != NULL) free(h_cscRowInd_);
    if (h_cscVal_    != NULL) free(h_cscVal_);
    h_cscColPtr_ = h_csrRowPtr_;
    h_cscRowInd_ = h_csrColInd_;
    h_cscVal_    = h_csrVal_;
  } else {
    coo2csc(h_cscColPtr_, h_cscRowInd_, h_cscVal_,
            *row_indices, *col_indices, *values, nrows_, ncols_);
  }

  if (dat_name != NULL) {
    if (!exists(dat_name)) {
      std::ofstream ofs(dat_name, std::ios::out | std::ios::binary);
      if (ofs.fail()) {
        std::cout << "Error: Unable to open file for writing!\n";
      } else {
        printf("Writing %s\n", dat_name);
        ofs.write(reinterpret_cast<char*>(&nrows_), sizeof(Index));
        if (ncols_ != nrows_)
          std::cout << "Error: nrows not equal to ncols!\n";
        ofs.write(reinterpret_cast<char*>(&nvals_), sizeof(Index));
        ofs.write(reinterpret_cast<char*>(h_csrRowPtr_),
            (nrows_+1)*sizeof(Index));
        ofs.write(reinterpret_cast<char*>(h_csrColInd_),
            nvals_*sizeof(Index));
        ofs.close();
      }
    }
    free(dat_name);
  }

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::build(char* dat_name) {
  if (dat_name != NULL && exists(dat_name)) {
    // The size of the file in bytes is in results.st_size
    // -unserialize vector
    std::ifstream ifs(dat_name, std::ios::in | std::ios::binary);
    if (ifs.fail()) {
      std::cout << "Error: Unable to open file for reading!\n";
    } else {
      printf("Reading %s\n", dat_name);
      char* pch = strstr(dat_name, ".ud.");
      if (pch == NULL)
        symmetric_ = false;
      else
        symmetric_ = true;

      ifs.read(reinterpret_cast<char*>(&nrows_), sizeof(Index));
      if (ncols_ != nrows_)
        std::cout << "Error: nrows not equal to ncols!\n";
      ifs.read(reinterpret_cast<char*>(&nvals_), sizeof(Index));
      CHECK(allocateCpu());

      ifs.read(reinterpret_cast<char*>(h_csrRowPtr_),
          (nrows_+1)*sizeof(Index));

      ifs.read(reinterpret_cast<char*>(h_csrColInd_),
          nvals_*sizeof(Index));

      for (Index i = 0; i < nvals_; i++)
        h_csrVal_[i] = static_cast<T>(1);

      if (format_ == GrB_SPARSE_MATRIX_CSRONLY) {
      //if (symmetric_ || format_ == GrB_SPARSE_MATRIX_CSRONLY) {
        if (h_cscColPtr_ != NULL) free(h_cscColPtr_);
        if (h_cscRowInd_ != NULL) free(h_cscRowInd_);
        if (h_cscVal_    != NULL) free(h_cscVal_);
        h_cscColPtr_ = h_csrRowPtr_;
        h_cscRowInd_ = h_csrColInd_;
        h_cscVal_    = h_csrVal_;
      } else {
        csr2csc(h_cscColPtr_, h_cscRowInd_, h_cscVal_,
                h_csrRowPtr_, h_csrColInd_, h_csrVal_, nrows_, ncols_);
      }

      CHECK(cpuToGpu());
    }
    free(dat_name);
  } else {
    std::cout << "Error: Unable to read file!\n";
  }
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::build(const std::vector<T>* values,
                            Index                 nvals) {
  std::cout << "SparseMatrix Build from dense input\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::build(Index* row_ptr,
                            Index* col_ind,
                            T*     values,
                            Index  nvals) {
  d_csrRowPtr_ = row_ptr;
  d_csrColInd_ = col_ind;
  d_csrVal_    = values;

  nvals_       = nvals;
  need_update_ = true;

  // Don't forget CPU must still be allocated
  CHECK(allocateCpu());

  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::setElement(Index row_index, Index col_index) {
  std::cout << "SparseMatrix setElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::extractElement(T* val, Index row_index, Index col_index) {
  std::cout << "SparseMatrix extractElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::extractTuples(std::vector<Index>* row_indices,
                                    std::vector<Index>* col_indices,
                                    std::vector<T>*     values,
                                    Index*              n) {
  CHECK(gpuToCpu());
  row_indices->clear();
  col_indices->clear();
  values->clear();

  if (*n > nvals_) {
    std::cout << "Error: Too many tuples requested!\n";
    return GrB_UNINITIALIZED_OBJECT;
  }

  if (*n < nvals_) {
    std::cout << "Error: Insufficient space!\n";
    return GrB_INSUFFICIENT_SPACE;
  }

  Index count = 0;
  for (Index row = 0; row < nrows_; row++) {
    for (Index ind = h_csrRowPtr_[row]; ind < h_csrRowPtr_[row+1]; ind++) {
      if (h_csrVal_[ind] != 0 && count < *n) {
        count++;
        row_indices->push_back(row);
        col_indices->push_back(h_csrColInd_[ind]);
        values->push_back(h_csrVal_[ind]);
      }
    }
  }

  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::extractTuples(std::vector<T>* values, Index* n) {
  std::cout << "SparseMatrix extractTuples to dense vector\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
const T SparseMatrix<T>::operator[](Index ind) {
  CHECKVOID(gpuToCpu(true));
  if (ind >= nvals_) std::cout << "Error: index out of bounds!\n";
  return h_csrColInd_[ind];
}

template <typename T>
Info SparseMatrix<T>::print(bool force_update) {
  std::cout << nrows_ << " x " << ncols_ << ": " << nvals_ << " nnz\n";
  CHECK(gpuToCpu(force_update));
  printArray("csrColInd", h_csrColInd_, std::min(nvals_, 40));
  printArray("csrRowPtr", h_csrRowPtr_, std::min(nrows_+1, 40));
  printArray("csrVal",    h_csrVal_,    std::min(nvals_, 40));
  CHECK(printCSR("pretty print"));
  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    if (!h_cscRowInd_ || !h_cscColPtr_ || !h_cscVal_)
      syncCpu();
    printArray("cscRowInd", h_cscRowInd_, std::min(nvals_, 40));
    printArray("cscColPtr", h_cscColPtr_, std::min(ncols_+1, 40));
    printArray("cscVal",    h_cscVal_,    std::min(nvals_, 40));
    CHECK(printCSC("pretty print"));
  }
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::check() {
  CHECK(gpuToCpu());
  std::cout << "Begin check:\n";
  // printArray( "rowptr", h_csrRowPtr_ );
  // printArray( "colind", h_csrColInd_+23 );
  // Check csrRowPtr is monotonically increasing
  for (Index row = 0; row < nrows_; row++) {
    // std::cout << "Comparing " << h_csrRowPtr_[row+1] << " >= " << h_csrRowPtr_[row] << std::endl;
    assert(h_csrRowPtr_[row+1] >= h_csrRowPtr_[row]);
  }

  // Check that: 1) there are no -1's in ColInd
  //             2) monotonically increasing
  for (Index row = 0; row < nrows_; row++) {
    Index row_start = h_csrRowPtr_[row];
    Index row_end   = h_csrRowPtr_[row+1];
    Index p_end     = h_csrRowPtr_[row+1];
    // std::cout << row << " " << row_end-row_start << std::endl;
    // printArray( "colind", h_csrColInd_+row_start, p_end-row_start );
    // printArray( "val", h_csrVal_+row_start, p_end-row_start );
    for (Index col = row_start; col < row_end-1; col++) {
      // std::cout << "Comparing " << h_csrColInd_[col+1] << " >= " << h_csrColInd_[col] << "\n";
      assert(h_csrColInd_[col] != -1);
      assert(h_csrColInd_[col+1] >= h_csrColInd_[col]);
      assert(h_csrVal_[col] > 0);
    }
    for (Index col = row_end; col < p_end; col++)
      assert(h_csrColInd_[col] == -1);
  }
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::setNrows(Index nrows) {
  nrows_ = nrows;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::setNcols(Index ncols) {
  ncols_ = ncols;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::setNvals(Index nvals) {
  nvals_ = nvals;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::getFormat(SparseMatrixFormat* format) const {
  *format = format_;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::getSymmetry(bool* symmetry) const {
  *symmetry = symmetric_;
  *symmetry = false;
  return GrB_SUCCESS;
}

// Note: has different meaning from sequential resize
//      -that one makes SparseMatrix bigger
//      -this one accounts for smaller nrows
template <typename T>
Info SparseMatrix<T>::resize(Index nrows, Index ncols) {
  if (nrows <= nrows_)
    nrows_ = nrows;
  else
    return GrB_PANIC;
  if (ncols <= ncols_)
    ncols_ = ncols;
  else
    return GrB_PANIC;

  return GrB_SUCCESS;
}

template <typename T>
template <typename U>
Info SparseMatrix<T>::fill(Index axis, Index nvals, U start) {
  CHECK(setNvals(nvals));
  CHECK(allocate());

  if (axis == 0) {
    for (Index i = 0; i < nvals; i++)
      h_csrRowPtr_[i] = (Index) start;
  } else if (axis == 1) {
    for (Index i = 0; i < nvals; i++)
      h_csrColInd_[i] = (Index) start;
  } else if (axis == 2) {
    for (Index i = 0; i < nvals; i++)
      h_csrVal_[i] = (T) start;
  }

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
template <typename U>
Info SparseMatrix<T>::fillAscending(Index axis, Index nvals, U start) {
  CHECK(setNvals(nvals));
  CHECK(allocate());

  if (axis == 0) {
    for (Index i = 0; i < nvals; i++)
      h_csrRowPtr_[i] = i+(Index) start;
  } else if (axis == 1) {
    for (Index i = 0; i < nvals; i++)
      h_csrColInd_[i] = i+(Index) start;
  } else if (axis == 2) {
    for (Index i = 0; i < nvals; i++)
      h_csrVal_[i] = (T)i+start;
  }

  CHECK(cpuToGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::swap(SparseMatrix* rhs) {  // NOLINK(build/include_what_you_use)
  // Swap scalars
  std::swap(nrows_,     rhs->nrows_);
  std::swap(ncols_,     rhs->ncols_);
  std::swap(nvals_,     rhs->nvals_);
  std::swap(ncapacity_, rhs->ncapacity_);
  std::swap(nempty_,    rhs->nempty_);

  // Swap CPU pointers
  std::swap(h_csrRowPtr_, rhs->h_csrRowPtr_);
  std::swap(h_csrColInd_, rhs->h_csrColInd_);
  std::swap(h_csrVal_,    rhs->h_csrVal_);
  std::swap(h_cscColPtr_, rhs->h_cscColPtr_);
  std::swap(h_cscRowInd_, rhs->h_cscRowInd_);
  std::swap(h_cscVal_,    rhs->h_cscVal_);

  // Swap GPU pointers
  std::swap(d_csrRowPtr_, rhs->d_csrRowPtr_);
  std::swap(d_csrColInd_, rhs->d_csrColInd_);
  std::swap(d_csrVal_,    rhs->d_csrVal_);
  std::swap(d_cscColPtr_, rhs->d_cscColPtr_);
  std::swap(d_cscRowInd_, rhs->d_cscRowInd_);
  std::swap(d_cscVal_,    rhs->d_cscVal_);

  std::swap(need_update_, rhs->need_update_);
  std::swap(symmetric_,   rhs->symmetric_);
  std::swap(format_,      rhs->format_);
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::rebuild(T identity, Descriptor* desc) {
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

  if (desc->debug()) {
    printDevice("d_csrColInd", d_csrColInd_, nvals_);
    printDevice("d_csrVal",    d_csrVal_,    nvals_);
  }

  // Calculate memory consumption
  // 1) d_flag: nvals_
  // 2) temp_row: nrows_+1 use d_flag and d_csrRowPtr_ to compute segmented
  //    reduction
  // 3) d_scan: nvals_ (can overwrite temp_row)
  // 4) temp_ind: nvals_ (can overwrite d_flag) use d_scan and d_csrColInd_
  // 5) temp_val: nvals_ use d_scan and d_csrVal_

  // Prune vals (0.f's) from matrix
  desc->resize(3*nvals_*sizeof(Index), "buffer");
  Index* d_flag = reinterpret_cast<Index*>(desc->d_buffer_);
  Index* d_scan = reinterpret_cast<Index*>(desc->d_buffer_)+nvals_;
  Index* temp_row = reinterpret_cast<Index*>(desc->d_buffer_)+nvals_;
  Index* temp_ind = reinterpret_cast<Index*>(desc->d_buffer_);
  T*     temp_val = reinterpret_cast<T*>(desc->d_buffer_)+2*nvals_;
  Index temp_nvals = 0;
  Index temp = 0;

  // 1) Mark values we want to keep
  updateFlagKernel<<<NB, NT>>>(d_flag, identity, d_csrVal_, nvals_);

  // 2) Calculate segmented reduction
  // Note: Must store total into d_csrRowPtr_+nrows_ or this will not result in
  // valid CSR data structure
  reduceMatrixCommon(d_scan, NULL, PlusMonoid<Index>(), d_csrRowPtr_, d_flag,
      nrows_, desc);
  mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(d_scan, nrows_, (Index) 0,
      mgpu::plus<Index>(),  // NOLINT(build/include_what_you_use)
      d_csrRowPtr_+nrows_, &temp, d_csrRowPtr_, temp_ind+2*nvals_,
      *(desc->d_context_));

  // 3) Compute scan (prefix-sum)
  mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(d_flag, nvals_, (Index) 0,
      mgpu::plus<Index>(),  // NOLINT(build/include_what_you_use)
      reinterpret_cast<Index*>(0), &temp_nvals, d_scan, temp_ind+2*nvals_,
      *(desc->d_context_));

  if (desc->debug()) {
    printDevice("d_flag", d_flag, nvals_);
    printDevice("d_scan", d_scan, nvals_);
    std::cout << "Pre-build size: " << nvals_ << std::endl;
    std::cout << "Post-build size: " << temp_nvals << std::endl;
    std::cout << "temp: " << temp << " == " << temp_nvals << ": temp_nvals\n";
  }

  // 4) Stream compact
  streamCompactSparseKernel<<<NB, NT>>>(temp_ind, temp_val, d_scan, (T)identity,
      d_csrColInd_, d_csrVal_, nvals_);
  CUDA_CALL(cudaMemcpy(d_csrColInd_, temp_ind, temp_nvals*sizeof(Index),
      cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(d_csrVal_, temp_val, temp_nvals*sizeof(T),
      cudaMemcpyDeviceToDevice));

  nvals_ = temp_nvals;
  if (desc->debug()) {
    printDevice("d_csrColInd", d_csrColInd_, nvals_);
    printDevice("d_csrVal",    d_csrVal_,    nvals_);
    validate();
  }
  need_update_ = true;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::allocateCpu() {
  // Allocate
  ncapacity_ = kcap_ratio_*nvals_;

  // Host malloc
  if (nrows_ > 0 && h_csrRowPtr_ == NULL)
    h_csrRowPtr_ = reinterpret_cast<Index*>(malloc((nrows_+1)*sizeof(Index)));
  if (nvals_ > 0 && h_csrColInd_ == NULL)
    h_csrColInd_ = reinterpret_cast<Index*>(malloc(ncapacity_*sizeof(Index)));
  if (nvals_ > 0 && h_csrVal_ == NULL)
    h_csrVal_    = reinterpret_cast<T*>(malloc(ncapacity_*sizeof(T)));

  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    if (ncols_ > 0 && h_cscColPtr_ == NULL) {
      h_cscColPtr_ = reinterpret_cast<Index*>(malloc((ncols_+1)*sizeof(Index)));
      std::cout << "Allocate " << ncols_ + 1 << std::endl;
    } else {
      std::cout << "Do not allocate " << ncols_ << " " << h_cscColPtr_ << "\n";
    }

    if (nvals_ > 0 && h_cscRowInd_ == NULL) {
      h_cscRowInd_ = reinterpret_cast<Index*>(malloc(ncapacity_*sizeof(Index)));
      std::cout << "Allocate " << ncapacity_ << std::endl;
    } else {
      std::cout << "Do not allocate " << nvals_ << " " << h_cscRowInd_ << "\n";
    }

    if (nvals_ > 0 && h_cscVal_ == NULL) {
      h_cscVal_    = reinterpret_cast<T*>(malloc(ncapacity_*sizeof(T)));
      std::cout << "Allocate " << ncapacity_ << std::endl;
    } else {
      std::cout << "Do not allocate " << nvals_ << " " << h_cscVal_ << "\n";
    }
  }

  // TODO(@ctcyang): does not need to be so strict since mxm may need to
  // only set storage type, but not allocate yet since nvals_ not known
  // if( h_csrRowPtr_==NULL || h_csrColInd_==NULL || h_csrVal_==NULL )
  //   return GrB_OUT_OF_MEMORY;

  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::allocateGpu() {
  // GPU malloc
  if (nrows_ > 0 && d_csrRowPtr_ == NULL)
    CUDA_CALL(cudaMalloc(&d_csrRowPtr_, (nrows_+1)*sizeof(Index)));
  if (nvals_ > 0 && d_csrColInd_ == NULL)
    CUDA_CALL(cudaMalloc(&d_csrColInd_, ncapacity_*sizeof(Index)));
  if (nvals_ > 0 && d_csrVal_ == NULL) {
    CUDA_CALL(cudaMalloc(&d_csrVal_, ncapacity_*sizeof(T)));
    printMemory("csrVal");
  }
  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    if (nvals_ > 0 && d_cscVal_ == NULL) {
      CUDA_CALL(cudaMalloc(&d_cscVal_, ncapacity_*sizeof(T)));
      printMemory("cscVal");

    if (!symmetric_) {
      if (nrows_ > 0 && d_cscColPtr_ == NULL)
        CUDA_CALL(cudaMalloc(&d_cscColPtr_, (ncols_+1)*sizeof(Index)));
      if (nvals_ > 0 && d_cscRowInd_ == NULL)
        CUDA_CALL(cudaMalloc(&d_cscRowInd_, ncapacity_*sizeof(Index)));
      }
    }
  }

  // TODO(@ctcyang): same reason as above for allocateCpu()
  // if( d_csrRowPtr_==NULL || d_csrColInd_==NULL || d_csrVal_==NULL )
  //   return GrB_OUT_OF_MEMORY;

  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::allocate() {
  CHECK(allocateCpu());
  CHECK(allocateGpu());
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::printCSR(const char* str) {
  Index row_length = std::min(20, nrows_);
  Index col_length = std::min(20, ncols_);
  std::cout << str << ":\n";

  for (Index row = 0; row < row_length; row++) {
    Index col_start = h_csrRowPtr_[row];
    Index col_end   = h_csrRowPtr_[row+1];
    for (Index col = 0; col < col_length; col++) {
      Index col_ind = h_csrColInd_[col_start];
      if (col_start < col_end && col_ind == col && h_csrVal_[col_start] > 0) {
        std::cout << "x ";
        col_start++;
      } else {
        std::cout << "0 ";
      }
    }
    std::cout << std::endl;
  }
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::printCSC(const char* str) {
  Index row_length = std::min(20, nrows_);
  Index col_length = std::min(20, ncols_);
  std::cout << str << ":\n";

  for (Index row = 0; row < col_length; row++) {
    Index col_start = h_cscColPtr_[row];
    Index col_end   = h_cscColPtr_[row+1];
    for (Index col = 0; col < row_length; col++) {
      Index col_ind = h_cscRowInd_[col_start];
      if (col_start < col_end && col_ind == col && h_cscVal_[col_start] > 0) {
        std::cout << "x ";
        col_start++;
      } else {
        std::cout << "0 ";
      }
    }
    std::cout << std::endl;
  }
  return GrB_SUCCESS;
}

// Copies graph to GPU
template <typename T>
Info SparseMatrix<T>::cpuToGpu() {
  CHECK(allocateGpu());

  CUDA_CALL(cudaMemcpy(d_csrRowPtr_, h_csrRowPtr_, (nrows_+1)*sizeof(Index),
      cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_csrColInd_, h_csrColInd_, nvals_*sizeof(Index),
      cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_csrVal_,    h_csrVal_,    nvals_*sizeof(T),
      cudaMemcpyHostToDevice));

  if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
    CUDA_CALL(cudaMemcpy(d_cscVal_, h_cscVal_, nvals_*sizeof(T),
        cudaMemcpyHostToDevice));

    if (!symmetric_) {
      CUDA_CALL(cudaMemcpy(d_cscColPtr_, h_cscColPtr_, (ncols_+1)*sizeof(Index),
          cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(d_cscRowInd_, h_cscRowInd_, nvals_*sizeof(Index),
          cudaMemcpyHostToDevice));
    } else {
      d_cscColPtr_ = d_csrRowPtr_;
      d_cscRowInd_ = d_csrColInd_;
    }
  }

  return GrB_SUCCESS;
}

// Copies graph to CPU
template <typename T>
Info SparseMatrix<T>::gpuToCpu(bool force_update) {
  if (need_update_ || force_update) {
    CUDA_CALL(cudaMemcpy(h_csrRowPtr_, d_csrRowPtr_, (nrows_+1)*sizeof(Index),
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_csrColInd_, d_csrColInd_, nvals_*sizeof(Index),
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_csrVal_,    d_csrVal_,    nvals_*sizeof(T),
        cudaMemcpyDeviceToHost));

    if (format_ == GrB_SPARSE_MATRIX_CSRCSC) {
      // Must account for combination of:
      // 1) CSRCSC
      // 2) sparse matrix being output of matrix-matrix multiply
      // In this case, the CSC copy does not exist, which causes an error.
      if (d_cscVal_ && d_cscColPtr_ && d_cscRowInd_ && 
          h_cscVal_ && h_cscColPtr_ && h_cscRowInd_) {
        CUDA_CALL(cudaMemcpy(h_cscVal_, d_cscVal_, nvals_*sizeof(T),
            cudaMemcpyDeviceToHost));
        if (!symmetric_) {
          CUDA_CALL(cudaMemcpy(h_cscColPtr_, d_cscColPtr_,
              (ncols_+1)*sizeof(Index), cudaMemcpyDeviceToHost));
          CUDA_CALL(cudaMemcpy(h_cscRowInd_, d_cscRowInd_, nvals_*sizeof(Index),
              cudaMemcpyDeviceToHost));
        }
      }
    }
  }
  need_update_ = false;
  return GrB_SUCCESS;
}

// Synchronizes CSR with CSC representation
template <typename T>
Info SparseMatrix<T>::syncCpu() {
  CHECK(allocateCpu());
  if (h_csrRowPtr_ && h_csrColInd_ && h_csrVal_ && 
      h_cscColPtr_ && h_cscRowInd_ && h_cscVal_)
    csr2csc(h_cscColPtr_, h_cscRowInd_, h_cscVal_,
            h_csrRowPtr_, h_csrColInd_, h_csrVal_, nrows_, ncols_);
  else
    return GrB_INVALID_OBJECT;
  return GrB_SUCCESS;
}

template <typename T>
Info SparseMatrix<T>::validate() {
  CHECK(gpuToCpu(true));
  //printArray("rowptr", h_csrRowPtr_, 30, false);
  //printArray("colind", h_csrColInd_, 1000, false);
  // Check csrRowPtr is monotonically increasing
  for (graphblas::Index row = 0; row < nrows_; row++) {
    //std::cout << "Comparing " << h_csrRowPtr_[row+1] << " >= " << h_csrRowPtr_[row] << std::endl;
    BOOST_ASSERT(h_csrRowPtr_[row+1] >= h_csrRowPtr_[row]);
  }

  // Check that: 1) there are no -1's in ColInd
  //             2) monotonically increasing
  for (graphblas::Index row = 0; row < nrows_; row++) {
    graphblas::Index row_start = h_csrRowPtr_[row];
    graphblas::Index row_end   = h_csrRowPtr_[row+1];
    // graphblas::Index row_end = row_start+h_rowLength_[row];
    // std::cout << row << " ";
    // printArray("colind", h_csrColInd_+row_start, h_rowLength_[row]);
    for (graphblas::Index col = row_start; col < row_end - 1; col++) {
      if (h_csrColInd_[col+1] <= h_csrColInd_[col]) {
        std::cout << "Error: on row " << row << " nnz " << col << ": ";
        std::cout << h_csrColInd_[col+1] << " > " << h_csrColInd_[col] << "\n";
      }
      BOOST_ASSERT(h_csrColInd_[col] != -1);
      BOOST_ASSERT(h_csrColInd_[col+1] > h_csrColInd_[col]);
    }
  }
  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPARSE_MATRIX_HPP_
