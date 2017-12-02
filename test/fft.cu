#include "depend.h"

__global__ void printfft(cufftDoubleComplex *data, int len)
{
  for(int i=0; i<len; i++)
  {
    printf("%f %f \n", data[i].x, data[i].y);
  }
}

void fft()
{
  int len = 10;
  cufftDoubleComplex A[len];
  for(int i=0; i<len; i++)
    A[i] = make_cuDoubleComplex(i,0.0);
  
  cufftDoubleComplex* cudamem;
  cudaMalloc(&cudamem, sizeof(cufftDoubleComplex) * len);
  cudaMemcpy(cudamem, A, sizeof(cufftDoubleComplex)*len, 
      cudaMemcpyHostToDevice);

  cufftDoubleComplex *data;
  cudaMalloc(&data, sizeof(cufftDoubleComplex) * len);

  cufftHandle plan;
  cufftPlan2d(&plan, 2, 5, CUFFT_Z2Z);

  cufftExecZ2Z(plan, cudamem, data, CUFFT_FORWARD);

  cudaDeviceSynchronize();

  cufftExecZ2Z(plan, data, cudamem, CUFFT_INVERSE); 

  cudaDeviceSynchronize();

  printfft<<<1,1>>>(cudamem,len);

  cudaDeviceSynchronize();


  printf("done\n");

  
}
