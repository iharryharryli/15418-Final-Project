#include "depend.h"

__global__ void printfft(cufftComplex *data, int len)
{
  for(int i=0; i<len; i++)
  {
    printf("%f %f \n", data[i].x, data[i].y);
  }
}

void fft()
{
  int len = 10;
  cufftComplex A[len];
  for(int i=0; i<len; i++)
    A[i] = make_cuFloatComplex(i,0.0);
  
  cufftComplex* cudamem;
  cudaMalloc(&cudamem, sizeof(cufftComplex) * len);
  cudaMemcpy(cudamem, A, sizeof(cufftComplex)*len, cudaMemcpyHostToDevice);

  cufftComplex *data;
  cudaMalloc(&data, sizeof(cufftComplex) * len);

  cufftHandle plan;
  cufftPlan1d(&plan, len, CUFFT_C2C, 1);

  cufftExecC2C(plan, cudamem, data, CUFFT_FORWARD);

  cudaDeviceSynchronize();

  printfft<<<1,1>>>(data,len);

  cudaDeviceSynchronize();


  printf("done\n");

  
}
