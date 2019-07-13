#ifndef GRAPHBLAS_BACKEND_CUDA_UTIL_HPP_
#define GRAPHBLAS_BACKEND_CUDA_UTIL_HPP_

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                               \
  cudaError err = call;                                                 \
  if (cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_CALL(call) do {                                            \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if (cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
     exit(EXIT_FAILURE);                                                \
  } } while (0)

#define CURAND_CALL(x) do {                                             \
  if ((x)!=CURAND_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "CURAND error in file '%s' in line %i.\n",        \
                __FILE__, __LINE__);                                    \
      exit(EXIT_FAILURE);                                               \
  } } while (0)

#include <cstdlib>

namespace graphblas {
namespace backend {

void printMemory(const char* str) {
  size_t free, total;
  if (GrB_MEMORY) {
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    std::cout << str << ": " << free << " bytes left out of " << total <<
        " bytes\n";
  }
}

template <typename T>
void printDevice(const char* str, const T* array, int length = 40,
                 bool limit = true) {
  if (limit && length > 40) length = 40;

  // Allocate array on host
  T *temp = reinterpret_cast<T*>(malloc(length*sizeof(T)));
  CUDA_CALL(cudaMemcpy(temp, array, length*sizeof(T), cudaMemcpyDeviceToHost));
  printArray(str, temp, length, limit);

  // Cleanup
  if (temp) free( temp );
}

template <typename T>
void printCode(const char* str, const T* array, int length) {
  // Allocate array on host
  T *temp = reinterpret_cast<T*>(malloc(length*sizeof(T)));
  CUDA_CALL(cudaMemcpy(temp, array, length*sizeof(T), cudaMemcpyDeviceToHost));

  // Traverse array, printing out move
  // Followed by manual reordering:
  // 1) For each dst block, find final move to that block. Mark its src.
  // 2) For all moves to that dst block, change dst to src.
  for (Index i = length-1; i >= 0; i--)
    if (temp[i] != i)
      printf("  count += testMerge( state, %d, %d, true );\n", temp[i], i);

  // Cleanup
  if (temp) free(temp);
}

void printState(bool use_mask, bool use_accum, bool use_scmp, bool use_repl,
                bool use_tran) {
  std::cout << "Mask: " << use_mask  << std::endl;
  std::cout << "Accum:" << use_accum << std::endl;
  std::cout << "SCMP: " << use_scmp  << std::endl;
  std::cout << "Repl: " << use_repl  << std::endl;
  std::cout << "Tran: " << use_tran  << std::endl;
}

/*! 
 * \brief constexpr variant of std::min, since we may not be using C++14
 */
template<typename T> constexpr
T const& min(T const& a, T const& b) {
  return a < b ? a : b;
}

/*!
 * \brief  constexpr variant of std::max, since we may not be using C++14
 */
template<typename T> constexpr
T const& max(T const& a, T const& b) {  // NOLINT(build/include_what_you_use)
  return a > b ? a : b;
}

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
  }

  void Stop() {
    cudaEventRecord(stop, 0);
  }

  float ElapsedMillis() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};
}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_UTIL_HPP_
