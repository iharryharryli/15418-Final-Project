#include "depend.h"
struct Torus
{
  int resx,resy,resz;
  int sizex,sizey,sizez;
  float dx,dy,dz;

  int plen;

  float* out;
};

void Torus_calc_ds(Torus* t)
{
  t -> dx = ((float)t -> sizex) / (t -> resx);
  t -> dy = ((float)t -> sizey) / (t -> resy);
  t -> dz = ((float)t -> sizez) / (t -> resz);
}

__constant__ Torus torus;

__device__  __inline__  int 
index3d(int i, int j, int k)
{
  return (k + j*torus.resz + i*torus.resz*torus.resy);
}

__global__ void Torus_Div (float* vx, float* vy, float* vz)
{

  float dx2 = torus.dx * torus.dx;
  float dy2 = torus.dy * torus.dy;
  float dz2 = torus.dz * torus.dz;

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
          (vz[normal_index] - vz[index3d(j,j,izm)])/dz2;

      }
    }
  }

}

