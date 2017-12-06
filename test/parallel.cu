#define THREADS_PER_BLOCK 256

int calc_numblock(int limit, int threadsPerBlock)
{
  return (limit + threadsPerBlock - 1) / threadsPerBlock;
}

__device__ int check_limit(int limit)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < limit)
    return i;
  return -1;
}
