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

__device__ void div_mycomplex(mycomplex* n, float d)
{
  n -> r /= d;
  n -> i /= d;
}

__global__ void ISF_Normalize(mycomplex* psi1, mycomplex* psi2)
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        float psi_norm = 
          sqrt(psi1->r*psi1->r+psi1->i*psi1->i+
               psi2->r*psi2->r+psi2->i*psi2->i);
        
        int ind = index3d(i,j,k);
        div_mycomplex(&psi1[ind], psi_norm);
        div_mycomplex(&psi2[ind], psi_norm);
      }
    }
  }
}

__global__ void ISF_BuildSchroedinger()
{
  float nx = torus.resx, ny = torus.resy, nz = torus.resz;
  float fac = -4.0 * M_PI * M_PI * isf.hbar;
  
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
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

        //printf("%f %f \n", isf.mask[index3d(i,j,k)].r,isf.mask[index3d(i,j,k)].i);
        
      }
    }
  }

  printf("Done ISF_BuildSchroedinger \n"); 
}

