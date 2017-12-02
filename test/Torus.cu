#include "depend.h"
#include "mycomplex.cu"
struct Torus
{
  int resx,resy,resz;
  int sizex,sizey,sizez;
  double dx,dy,dz;

  int plen;

  double* out;
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

__global__ void Torus_Div (double* vx, double* vy, double* vz)
{

  double dx2 = torus.dx * torus.dx;
  double dy2 = torus.dy * torus.dy;
  double dz2 = torus.dz * torus.dz;

  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ixm = (i - 1) % torus.resx;
        int iym = (j - 1) % torus.resy;
        int izm = (k - 1) % torus.resz;

        int normal_index = index3d(i,j,k);
        
        torus.out[normal_index] = 
          (vx[normal_index] - vx[index3d(ixm,j,k)])/dx2;
        torus.out[normal_index] +=
          (vy[normal_index] - vy[index3d(i,iym,k)])/dy2;
        torus.out[normal_index] +=
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

__global__ void Torus_f2buf(double* f)
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        torus.fftbuf[ind] = make_cuDoubleComplex(f[ind],0.0);
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
        if(denom > 1e-16)
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

void Torus_PoissonSolve(double* f)
{
  Torus_f2buf<<<1,1>>>(f);
  cudaDeviceSynchronize(); 
 

  //Torus_printfft<<<1,1>>>(); cudaDeviceSynchronize(); 


  // fft
  cufftHandle plan;
  cufftPlan3d(&plan, torus_cpu.resx, 
              torus_cpu.resy, torus_cpu.resz, CUFFT_Z2Z);
  cufftExecZ2Z(plan, torus_cpu.fftbuf, 
                       torus_cpu.fftbuf, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  
  PoissonSolve_main<<<1,1>>>();
  cudaDeviceSynchronize();   

  // ifft
  cufftExecZ2Z(plan, torus_cpu.fftbuf, 
                       torus_cpu.fftbuf, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  cufftDestroy(plan);
  

  //Torus_printfft<<<1,1>>>();
  //Torus_printdouble<<<1,1>>>(f);
  //cudaDeviceSynchronize();
  

}

