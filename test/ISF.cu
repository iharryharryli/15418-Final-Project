#include "Torus.cu"

struct mycomplex
{
  float r;
  float i;
};

struct ISF
{
  float hbar;
  float dt;
  mycomplex* mask;
};

__constant__ ISF isf;

__device__ mycomplex exp_mycomplex(mycomplex inp)
{
  mycomplex res;
  res.r = exp(inp.r) * cos(inp.i);
  res.i = exp(inp.r) * sin(inp.i);
  return res;
}

__global__ void ISF_BuildSchroedinger()
{
  float nx = torus.resx, ny = torus.resy, nz = torus.resz;
  float fac = -4.0 * M_PI * M_PI * isf.hbar;
  
  for(int i=0; i<dim_info.d1; i++)
  {
    for(int j=0; j<dim_info.d2; j++)
    {
      for(int k=0; k<dim_info.d3; k++)
      {
        float kx = (i - nx / 2) / torus.sizex;
        float ky = (j - ny / 2) / torus.sizey;
        float kz = (k - nz / 2) / torus.sizez;
        float lambda = fac * (kx * kx + ky * ky + kz * kz);
        
        int ind = index3d(i,j,k);
        
        mycomplex inp;
        inp.r = 0;
        inp.i = lambda * isf.dt / 2;
        
        isf.mask[index3d(i,j,k)] = exp_mycomplex(inp);
        
      }
    }
  }
}

