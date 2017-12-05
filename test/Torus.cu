#include "depend.h"
#include "mycomplex.cu"
struct Torus
{
  int resx,resy,resz;
  int sizex,sizey,sizez;
  double dx,dy,dz;

  double* vx;
  double* vy;
  double* vz;

  int plen;

  double* div;
  cuDoubleComplex* fftbuf;
};

void Torus_calc_ds(Torus* t)
{
  t -> dx = ((double)t -> sizex) / (t -> resx);
  t -> dy = ((double)t -> sizey) / (t -> resy);
  t -> dz = ((double)t -> sizez) / (t -> resz);
}

__constant__ Torus torus;
Torus torus_cpu;

__device__  __inline__  int 
index3d(int i, int j, int k)
{
  return (k + j*torus.resz + i*torus.resz*torus.resy);
}

__device__  __inline__  void 
getCoords(int i, int *x, int *y, int *z)
{
  *x = i / (torus.resz * torus.resy);
  *y = (i % (torus.resz * torus.resy)) / torus.resz;
  *z = (i % (torus.resz * torus.resy)) % torus.resz;
}

__global__ void Torus_Div ()
{

  double dx2 = torus.dx * torus.dx;
  double dy2 = torus.dy * torus.dy;
  double dz2 = torus.dz * torus.dz;

  double* vx = torus.vx;
  double* vy = torus.vy;
  double* vz = torus.vz;

  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ixm = (i - 1 + torus.resx) % torus.resx;
        int iym = (j - 1 + torus.resy) % torus.resy;
        int izm = (k - 1 + torus.resz) % torus.resz;

        int normal_index = index3d(i,j,k);
        
        torus.div[normal_index] = 
          (vx[normal_index] - vx[index3d(ixm,j,k)])/dx2;
        torus.div[normal_index] +=
          (vy[normal_index] - vy[index3d(i,iym,k)])/dy2;
        torus.div[normal_index] +=
          (vz[normal_index] - vz[index3d(i,j,izm)])/dz2;

      }
    }
  }

}

__global__ void Torus_printfft()
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        printf("%f %f\n", torus.fftbuf[ind].x / torus.plen, 
              torus.fftbuf[ind].y / torus.plen);
      }
    }
  }

}

__global__ void Torus_printdouble(double* f)
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        printf("%f\n", f[ind]);
      }
    }
  }
}

__global__ void Torus_div2buf()
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        torus.fftbuf[ind] = make_cuDoubleComplex(torus.div[ind],0.0);
       }
    }
  }
}

__global__ void PoissonSolve_main()
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);   
        double sx = sin(M_PI*i/torus.resx) / torus.dx;
        double sy = sin(M_PI*j/torus.resy) / torus.dy;  
        double sz = sin(M_PI*k/torus.resz) / torus.dz;
        double denom = sx * sx + sy * sy + sz * sz;
        double fac = 0.0;
        if(ind > 0)
        {
          fac = -0.25 / denom;
        }
        //mul_mycomplex(&torus.fftbuf[ind], fac);
        torus.fftbuf[ind].x *= fac;
        torus.fftbuf[ind].y *= fac;
      }
    }
  }    
        
}

//*********** not tested! ***********
__global__ void fftshift(cufftDoubleComplex *data)
// The thing is: fftshift for even and odd dimensional arrays 
// are really different -- the even case is much simpler than the odd case
// To save ourselves the trouble we will only implement the even fftshift
// and give an error when the input has odd dimension
{
  int xs = torus.resx / 2;
  int ys = torus.resy / 2;
  int zs = torus.resz / 2;
  int len = torus.resx * torus.resy * torus.resz;
  int x, y, z = 0;
  int j;

  if (len % 2 == 1){
    printf("Error: fftshift only supports even sized grid!\n");
    return;
  }

  for (int i=0; i<len/2; i++)
  {
    cufftDoubleComplex temp = data[i];
    getCoords(i, &x, &y, &z);
    x = (x + xs) % torus.resx;
    y = (y + ys) % torus.resy;
    z = (z + zs) % torus.resz;
    j = index3d(x, y, z);
    data[i] = data[j];
    data[j] = temp;
  }
}

//*********** not tested! ***********
__global__ void ifftshift(cufftDoubleComplex *data)
// Since we are only working with even-sized arrays
// ifftshift is equivalent with fftshift
{
  int xs = torus.resx / 2;
  int ys = torus.resy / 2;
  int zs = torus.resz / 2;
  int len = torus.resx * torus.resy * torus.resz;
  int x, y, z = 0;
  int j;

  if (len % 2 == 1){
    printf("Error: fftshift only supports even sized grid!\n");
    return;
  }

  for (int i=0; i<len/2; i++)
  {
    cufftDoubleComplex temp = data[i];
    getCoords(i, &x, &y, &z);
    x = (x + xs) % torus.resx;
    y = (y + ys) % torus.resy;
    z = (z + zs) % torus.resz;
    j = index3d(x, y, z);
    data[i] = data[j];
    data[j] = temp;
  }
}

cufftHandle fftn(cufftDoubleComplex *data)
// Returns the cufft plan created
{
  cufftHandle plan;
  cufftPlan3d(&plan, torus_cpu.resx, torus_cpu.resy, torus_cpu.resz, CUFFT_Z2Z);
  cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  return plan;
}

void ifftn(cufftDoubleComplex *data, cufftHandle plan)
// Destorys the cufft plan after finshing
{
  cufftExecZ2Z(plan, data, data, CUFFT_INVERSE); 
  cudaDeviceSynchronize();
  cufftDestroy(plan);
}

void Torus_PoissonSolve()
// TODO: This is a crazy amount of passing data back and forth...
{
  Torus_div2buf<<<1,1>>>();
  cudaDeviceSynchronize(); 
 

  //Torus_printfft<<<1,1>>>(); cudaDeviceSynchronize(); 

  // fft

  cufftHandle plan = fftn(torus_cpu.fftbuf);

  // Do work in the fourier space
  PoissonSolve_main<<<1,1>>>();
  cudaDeviceSynchronize();   

  // ifft

  ifftn(torus_cpu.fftbuf, plan);
  

  //Torus_printfft<<<1,1>>>();
  //Torus_printdouble<<<1,1>>>(f);
}

__global__ void StaggeredSharp_kernel()
{
  for(int i=0; i<torus.plen; i++)
  {
    torus.vx[i] /= torus.dx;
    torus.vy[i] /= torus.dy;
    torus.vz[i] /= torus.dz;

    //printf("%f\n", torus.vx[i]);
  }
}

void Torus_StaggeredSharp()
{
  StaggeredSharp_kernel<<<1,1>>>();
  cudaDeviceSynchronize();
}


