#include "Torus.cu"



struct ISF
{
  float hbar;
  float dt;
  cuFloatComplex* mask;
  
  float* vx;
  float* vy;
  float* vz;
};

__constant__ ISF isf;
ISF isf_cpu;


__global__ void ISF_Normalize(cuFloatComplex* psi1, cuFloatComplex* psi2)
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        float psi_norm = 
          sqrt(psi1->x*psi1->x+psi1->y*psi1->y+
               psi2->x*psi2->x+psi2->y*psi2->y);
        
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
        
        cuFloatComplex inp;
        inp.x = 0;
        inp.y = lambda * isf.dt / 2;
        
        isf.mask[index3d(i,j,k)] = exp_mycomplex(inp);

        //printf("%f %f \n", isf.mask[index3d(i,j,k)].x,isf.mask[index3d(i,j,k)].y);
        
      }
    }
  }

  printf("Done ISF_BuildSchroedinger \n"); 
}

__global__ void ISF_VelocityOneForm(cuFloatComplex* psi1, 
                                    cuFloatComplex* psi2, 
                                  float hbar)
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ixp = (i + 1) % torus.resx;
        int iyp = (j + 1) % torus.resy;
        int izp = (k + 1) % torus.resz;

        int ind = index3d(i,j,k);
        int vxi = index3d(ixp,j,k);
        int vyi = index3d(i,iyp,k);
        int vzi = index3d(i,j,izp);
        
        cuFloatComplex vxraw = cuCaddf(
          cuCmulf(cuConjf(psi1[ind]),psi1[vxi]),
          cuCmulf(cuConjf(psi2[ind]),psi2[vxi])
          );
        cuFloatComplex vyraw = cuCaddf(
          cuCmulf(cuConjf(psi1[ind]),psi1[vyi]),
          cuCmulf(cuConjf(psi2[ind]),psi2[vyi])
          );
        cuFloatComplex vzraw = cuCaddf(
          cuCmulf(cuConjf(psi1[ind]),psi1[vzi]),
          cuCmulf(cuConjf(psi2[ind]),psi2[vzi])
          );

        isf.vx[ind] = angle_mycomplex(vxraw);
        isf.vy[ind] = angle_mycomplex(vyraw);
        isf.vz[ind] = angle_mycomplex(vzraw);

        /*if(isf.vx[ind] > 1.2)
          printf("%f\n", isf.vx[ind]);*/

      }
    }
  }
}

__global__ void ISF_Neg_Normal_GaugeTransform(cuFloatComplex* psi1,
                          cuFloatComplex* psi2, cuFloatComplex* q)
{
  cuFloatComplex negi = make_cuFloatComplex(0.0, -1.0 / torus.plen);

  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        cuFloatComplex eiq = 
          exp_mycomplex( cuCmulf(negi, q[ind]) );

        psi1[ind] = cuCmulf(psi1[ind], eiq);
        psi2[ind] = cuCmulf(psi2[ind], eiq);

      }
    }
  }
}

void ISF_PressureProject(cuFloatComplex* psi1,
                          cuFloatComplex* psi2)
{
  ISF_VelocityOneForm<<<1,1>>>(psi1, psi2, 1.0);
  cudaDeviceSynchronize();
  Torus_Div<<<1,1>>>(isf_cpu.vx, isf_cpu.vy, isf_cpu.vz); 
  cudaDeviceSynchronize();

  Torus_PoissonSolve(torus_cpu.out);

  ISF_Neg_Normal_GaugeTransform<<<1,1>>>(psi1,psi2,torus_cpu.fftbuf);
  cudaDeviceSynchronize(); 

}





