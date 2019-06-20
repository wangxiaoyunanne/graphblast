#ifndef GRAPHBLAS_BACKEND_CUDA_SPMSPV_INNER_HPP_
#define GRAPHBLAS_BACKEND_CUDA_SPMSPV_INNER_HPP_

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

#include <iostream>
#include <algorithm>

#include "graphblas/backend/cuda/kernels/kernels.hpp"

namespace graphblas {
namespace backend {
// Memory requirements: (4|V|+5|E|)*desc->memusage()
//   -desc->memusage() is defined in graphblas/types.hpp
//
//  -> d_csrColBad    |V|*desc->memusage()
//  -> d_csrColGood   |V|*desc->memusage()
//  -> d_csrColDiff   |V|*desc->memusage()
//  -> d_index        |V|*desc->memusage()
//  -> d_csrVecInd    |E|*desc->memusage() (u_ind)
//  -> d_csrSwapInd   |E|*desc->memusage()
//  -> d_csrVecVal    |E|*desc->memusage()
//  -> d_csrTempVal   |E|*desc->memusage() (u_val)
//  -> d_csrSwapVal   |E|*desc->memusage()
//  -> w_ind          |E|*desc->memusage()
//  -> w_val          |E|*desc->memusage()
//  -> d_temp_storage runtime constant
template <typename W, typename a, typename U,
          typename BinaryOpT, typename SemiringT>
Info spmspvApspie(Index*       w_ind,
                  W*           w_val,
                  Index*       w_nvals,
                  BinaryOpT    accum,
                  SemiringT    op,
                  Index        A_nrows,
                  Index        A_nvals,
                  const Index* A_csrRowPtr,
                  const Index* A_csrColInd,
                  const a*     A_csrVal,
                  const Index* u_ind,
                  const U*     u_val,
                  const Index* u_nvals,
                  Descriptor*  desc) {
  return GrB_SUCCESS;
}

// Memory requirements: 2|E|*desc->memusage()
//   -desc->memusage() is defined in graphblas/types.hpp
//
//  -> d_csrSwapInd   |E|*desc->memusage() [2*A_nrows: 1*|E|*desc->memusage()]
//  -> d_csrSwapVal   |E|*desc->memusage() [2*A_nrows+ 2*|E|*desc->memusage()]
//  -> d_temp_storage runtime constant
//
// TODO(@ctcyang): can lower 2|E| * desc->memusage() memory requirement further
// by doing external memory sorting
template <typename W, typename a, typename U,
          typename BinaryOpT, typename SemiringT>
Info spmspvApspieMerge(Index*       w_ind,
                       W*           w_val,
                       Index*       w_nvals,
                       BinaryOpT    accum,
                       SemiringT    op,
                       Index        A_nrows,
                       Index        A_nvals,
                       const Index* A_csrRowPtr,
                       const Index* A_csrColInd,
                       const a*     A_csrVal,
                       const Index* u_ind,
                       const U*     u_val,
                       const Index* u_nvals,
                       Descriptor*  desc) {
  // Get descriptor parameters for nthreads
  Desc_value ta_mode, tb_mode, nt_mode;
  CHECK(desc->get(GrB_TA, &ta_mode));
  CHECK(desc->get(GrB_TB, &tb_mode));
  CHECK(desc->get(GrB_NT, &nt_mode));

  const int ta = static_cast<int>(ta_mode);
  const int tb = static_cast<int>(tb_mode);
  const int nt = static_cast<int>(nt_mode);

  dim3 NT, NB;
  NT.x = nt;
  NT.y = 1;
  NT.z = 1;
  NB.x = (*u_nvals+nt-1)/nt;
  NB.y = 1;
  NB.z = 1;

  // Step 0) Must compute how many elements are in the selected region in the
  // worst-case. This is a global reduce.
  //  -> d_temp_nvals |V|
  //  -> d_scan       |V|+1
  int    size        = static_cast<float>(A_nvals)*desc->memusage()+1;
  void* d_temp_nvals = reinterpret_cast<void*>(w_ind);
  void* d_scan       = reinterpret_cast<void*>(w_val);
  void* d_temp       = desc->d_buffer_+2*A_nrows*sizeof(Index);

  if (desc->struconly())
    d_scan = desc->d_buffer_+(A_nrows+size)*sizeof(Index);

  if (desc->debug()) {
    assert(*u_nvals <= A_nrows);
    std::cout << "NT: " << NT.x << " NB: " << NB.x << std::endl;
  }

  indirectScanKernel<<<NB, NT>>>(reinterpret_cast<Index*>(d_temp_nvals),
      A_csrRowPtr, u_ind, *u_nvals);
  // Note: cannot use op.add_op() here
  mgpu::ScanPrealloc<mgpu::MgpuScanTypeExc>(reinterpret_cast<Index*>(
      d_temp_nvals), *u_nvals, (Index)0, mgpu::plus<Index>(),
      reinterpret_cast<Index*>(d_scan)+*u_nvals, w_nvals,
      reinterpret_cast<Index*>(d_scan), reinterpret_cast<Index*>(d_temp),
      *(desc->d_context_));

  if (desc->debug()) {
    printDevice("d_temp_nvals", reinterpret_cast<Index*>(d_temp_nvals),
        *u_nvals);
    printDevice("d_scan", reinterpret_cast<Index*>(d_scan), *u_nvals+1);

    std::cout << "u_nvals: " << *u_nvals << std::endl;
    std::cout << "w_nvals: " << *w_nvals << std::endl;
  }

  if (desc->struconly() && !desc->sort())
    CUDA_CALL(cudaMemset(w_ind, 0, A_nrows*sizeof(Index)));

  // No neighbors is one possible stopping condition
  if (*w_nvals == 0)
    return GrB_SUCCESS;

  // Step 1) Gather from CSR graph into one big array  |     |  |
  // Step 2) Vector Portion
  //   -IntervalExpand into frontier-length list
  //      1. Gather the elements indexed by d_csrVecInd
  //      2. Expand the elements to memory set by d_csrColGood
  //   -Element-wise multiplication with frontier
  // Step 3) Matrix Structure Portion
  // Step 4) Element-wise multiplication
  // Step 1-4) custom kernel method (1 single kernel)
  //   modify spmvCsrIndirectBinary() to stop after expand phase
  //   output: 1) expanded index array 2) expanded value array
  //   -> d_csrSwapInd |E| x desc->memusage()
  //   -> d_csrSwapVal |E| x desc->memusage()
  void* d_csrSwapInd;
  void* d_csrSwapVal;

  if (desc->struconly()) {
    d_csrSwapInd = desc->d_buffer_+   A_nrows      *sizeof(Index);
    d_temp       = desc->d_buffer_+(  A_nrows+size)*sizeof(Index);
  } else {
    d_csrSwapInd = desc->d_buffer_+ 2*A_nrows        *sizeof(Index);
    d_csrSwapVal = desc->d_buffer_+(2*A_nrows+  size)*sizeof(Index);
    d_temp       = desc->d_buffer_+(2*A_nrows+2*size)*sizeof(Index);
  }

  if (!desc->struconly_) {
  /*!
   * \brief IntervalExpand, which takes a scan of the output, values and 
   *        expands values into an output specified by the scan.
   *
   *        IntervalExpand:
   *        values  =  0,  1,  2,  3,  4,  5,  6,  7,  8
   *        counts  =  1,  2,  1,  0,  4,  2,  3,  0,  2
   *        d_scan  =  0,  1,  3,  4,  4,  8, 10, 13, 13 (moveCount = 15).
   * Expand values[i] by counts[i]:
   * d_temp  =  0, 1, 1, 2, 4, 4, 4, 4, 5, 5, 6, 6, 6, 8, 8
   *
   */
    IntervalExpand(*w_nvals, reinterpret_cast<Index*>(d_scan), u_val, *u_nvals,
        reinterpret_cast<T*>(d_temp), *(desc->d_context_));
    if (desc->debug())
      printDevice("d_temp", reinterpret_cast<T*>(d_temp), *w_nvals);
  }

  /*!
   * \brief IntervalGatherIndirect is a modification of IntervalGather, which
   *        takes some interval starts, interval lengths and gathers them into
   *        a single output.
   *
   *        IntervalGatherIndirect differs, because it replaces *some* 
   *        interval starts with *all* interval starts, and indices that we 
   *        care about that are used to index into interval starts. This lets
   *        the user in a single function call do 2 IntervalGather's.
   */ 
  IntervalGatherIndirect(*w_nvals, A_csrRowPtr,
      reinterpret_cast<Index*>(d_scan), *u_nvals, A_csrColInd, u_ind,
      reinterpret_cast<Index*>(d_csrSwapInd), *(desc->d_context_));
  if (!desc->struconly()) {
    IntervalGatherIndirect(*w_nvals, A_csrRowPtr,
        reinterpret_cast<Index*>(d_scan), *u_nvals, A_csrVal, u_ind,
        reinterpret_cast<T*>(d_csrSwapVal), *(desc->d_context_));

  // Step 4) Element-wise multiplication
    NB.x = (*w_nvals+nt-1)/nt;
    eWiseMultKernel<<<NB, NT>>>(reinterpret_cast<T*>(d_csrSwapVal),
        extractAdd(op), op.identity(), extractMul(op),
        reinterpret_cast<T*>(d_csrSwapVal), reinterpret_cast<T*>(d_temp),
        *w_nvals);
  }

  if (desc->debug()) {
    printDevice("SwapInd", reinterpret_cast<Index*>(d_csrSwapInd), *w_nvals);
    if (!desc->struconly())
      printDevice("SwapVal", reinterpret_cast<T*>(d_csrSwapVal), *w_nvals);
  }

  // Step 5) Sort step
  //   -> d_csrTempInd |E| x desc->memusage()
  //   -> d_csrTempVal |E| x desc->memusage()
  size_t temp_storage_bytes = 0;
  void* d_csrTempInd;
  void* d_csrTempVal;

  int endbit = sizeof(Index)*8;
  if (desc->endbit())
    endbit = std::min(endbit,
        static_cast<int>(log2(static_cast<float>(A_nrows)))+1);

  if (desc->struconly()) {
    if (desc->sort()) {
      d_csrTempInd = desc->d_buffer_+(A_nrows+size)*sizeof(Index);

      if (!desc->split())
        CUDA_CALL(cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes,
            reinterpret_cast<Index*>(d_csrSwapInd),
            reinterpret_cast<Index*>(d_csrTempInd), *w_nvals, 0, endbit));
      else
        temp_storage_bytes = desc->d_temp_size_;

      if (desc->debug())
        std::cout << temp_storage_bytes << " bytes required!\n";

      desc->resize(temp_storage_bytes, "temp");

      CUDA_CALL(cub::DeviceRadixSort::SortKeys(desc->d_temp_,
          temp_storage_bytes, reinterpret_cast<Index*>(d_csrSwapInd),
          reinterpret_cast<Index*>(d_csrTempInd), *w_nvals, 0, endbit));

      if (desc->debug())
        printDevice("TempInd", reinterpret_cast<Index*>(d_csrTempInd),
            *w_nvals);
    }
  } else {
    d_csrTempInd = desc->d_buffer_+(2*A_nrows+2*size)*sizeof(Index);
    d_csrTempVal = desc->d_buffer_+(2*A_nrows+3*size)*sizeof(Index);

    if (!desc->split())
      CUDA_CALL(cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes,
          reinterpret_cast<Index*>(d_csrSwapInd),
          reinterpret_cast<Index*>(d_csrTempInd),
          reinterpret_cast<T*>(d_csrSwapVal),
          reinterpret_cast<T*>(d_csrTempVal), *w_nvals, 0, endbit));
    else
      temp_storage_bytes = desc->d_temp_size_;

    if (desc->debug())
      std::cout << temp_storage_bytes << " bytes required!\n";

    desc->resize(temp_storage_bytes, "temp");

    CUDA_CALL(cub::DeviceRadixSort::SortPairs(desc->d_temp_, temp_storage_bytes,
        reinterpret_cast<Index*>(d_csrSwapInd),
        reinterpret_cast<Index*>(d_csrTempInd),
        reinterpret_cast<T*>(d_csrSwapVal),
        reinterpret_cast<T*>(d_csrTempVal), *w_nvals, 0, endbit));

    // MergesortKeys(d_csrVecInd, total, mgpu::less<int>(), desc->d_context_);

    if (desc->debug()) {
      printDevice("TempInd", reinterpret_cast<Index*>(d_csrTempInd), *w_nvals);
      printDevice("TempVal", reinterpret_cast<T*>    (d_csrTempVal), *w_nvals);
    }
  }

  if (desc->debug()) {
    printf("Endbit: %d\n", endbit);
    printf("Current iteration: %d nonzero vector, %d edges\n", *u_nvals,
      *w_nvals);
  }

  // Step 6) Segmented Reduce By Key
  if (desc->struconly()) {
    if (!desc->sort()) {
      NB.x = (*w_nvals+nt-1)/nt;
      scatter<<<NB, NT>>>(w_ind, reinterpret_cast<Index*>(d_csrSwapInd),
          (Index)1, *w_nvals);
      *w_nvals = A_nrows;

      if (desc->debug())
        printDevice("scatter", w_ind, *w_nvals);
    } else {
      Index  w_nvals_t = 0;
      ReduceByKey(reinterpret_cast<Index*>(d_csrTempInd),
          reinterpret_cast<T*>(d_csrSwapInd), *w_nvals,
          op.identity(), extractAdd(op), mgpu::equal_to<Index>(), w_ind,
          w_val, &w_nvals_t, reinterpret_cast<int*>(0), *(desc->d_context_));
      *w_nvals = w_nvals_t;
    }
  } else {
    Index  w_nvals_t = 0;
    ReduceByKey(reinterpret_cast<Index*>(d_csrTempInd),
        reinterpret_cast<T*>(d_csrTempVal), *w_nvals, op.identity(),
        extractAdd(op),
        mgpu::equal_to<Index>(),  // NOLINT(build/include_what_you_use)
        w_ind, w_val, &w_nvals_t, reinterpret_cast<int*>(0),
        *(desc->d_context_));
    *w_nvals = w_nvals_t;
  }

  return GrB_SUCCESS;
}
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_SPMSPV_INNER_HPP_
