#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cudaprint.h"

#include "justprint.cu"
#include "Torus.cu"
void cudaPrint()
{
  kernel_print<<<2,3>>>();
  cudaDeviceSynchronize();
}
