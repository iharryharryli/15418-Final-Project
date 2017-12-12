#include "depend.h"
// #include "Torus.cu"

__global__ void printfft(cufftDoubleComplex *data, int len)
{
  for(int i=0; i<len; i++)
  {
    printf("%f %f \n", data[i].x, data[i].y);
  }
}

//*********** not tested! ***********
// __global__ void fftshift(cufftDoubleComplex *data)
// // The thing is: fftshift for even and odd dimensional arrays 
// // are really different -- the even case is much simpler than the odd case
// // To save ourselves the trouble we will only implement the even fftshift
// // and give an error when the input has odd dimension
// {
//   int xs = torus.resx / 2;
//   int ys = torus.resy / 2;
//   int zs = torus.resz / 2;
//   int len = torus.resx * torus.resy * torus.resz;
//   int x, y, z = 0;
//   int j;

//   if (len % 2 == 1){
//     printf("Error: fftshift only supports even sized grid!\n");
//     return;
//   }

//   for (int i=0; i<len/2; i++)
//   {
//     cufftDoubleComplex temp = data[i];
//     getCoords(i, &x, &y, &z);
//     x = (x + xs) % torus.resx;
//     y = (y + ys) % torus.resy;
//     z = (z + zs) % torus.resz;
//     j = index3d(x, y, z);
//     data[i] = data[j];
//     data[j] = temp;
//   }
// }

// //*********** not tested! ***********
// __global__ void ifftshift(cufftDoubleComplex *data)
// // Since we are only working with even-sized arrays
// // ifftshift is equivalent with fftshift
// {
//   int xs = torus.resx / 2;
//   int ys = torus.resy / 2;
//   int zs = torus.resz / 2;
//   int len = torus.resx * torus.resy * torus.resz;
//   int x, y, z = 0;
//   int j;

//   if (len % 2 == 1){
//     printf("Error: fftshift only supports even sized grid!\n");
//     return;
//   }

//   for (int i=0; i<len/2; i++)
//   {
//     cufftDoubleComplex temp = data[i];
//     getCoords(i, &x, &y, &z);
//     x = (x + xs) % torus.resx;
//     y = (y + ys) % torus.resy;
//     z = (z + zs) % torus.resz;
//     j = index3d(x, y, z);
//     data[i] = data[j];
//     data[j] = temp;
//   }
// }

void fft()
// Test code to make sure cufft works the same as in MATLAB
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
