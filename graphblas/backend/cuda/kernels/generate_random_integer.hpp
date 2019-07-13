#ifndef GRAPHBLAS_BACKEND_CUDA_KERNELS_GENERATE_RANDOM_INTEGER_HPP_
#define GRAPHBLAS_BACKEND_CUDA_KERNELS_GENERATE_RANDOM_INTEGER_HPP_

namespace graphblas {
namespace backend {

// Initialize curand states
__global__ void initCurandStates(curandState_t* states,
                                 unsigned int   seed, 
                                 Index          size) {

  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (; id < size; id += gridDim.x * blockDim.x) {
    curand_init(seed, id, 0, &states[id]);
  }
}

// Generate uniformly distributed integers in range of 0 <= k < max_val
__global__ void generateRandomIntegersUniform(curandState_t* states,
                                              Index*         vec,
                                              Index          size,
                                              Index          max_val) {

  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (; id < size; id += gridDim.x * blockDim.x) {
    vec[id] = (Index)ceilf(curand_uniform(&states[id]) * max_val);
    // printf("vec[%d]: %d", id, vec[id]);
  }
}

}  // namespace backend
}  // namespace graphblas

#endif  // GRAPHBLAS_BACKEND_CUDA_KERNELS_GENERATE_RANDOM_INTEGER_HPP_
