#include "Torus.cu"



struct ISF
{
  double hbar;
  double dt;
  cuDoubleComplex* mask;
  
  double* vx;
  double* vy;
  double* vz;
};

__constant__ ISF isf;
ISF isf_cpu;


__global__ void ISF_Normalize(cuDoubleComplex* psi1, cuDoubleComplex* psi2)
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        double psi_norm = 
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
// Initializes the complex components of the field
{
  double nx = torus.resx, ny = torus.resy, nz = torus.resz;
  double fac = -4.0 * M_PI * M_PI * isf.hbar;
  
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        double kx = (i - nx / 2) / torus.sizex;
        double ky = (j - ny / 2) / torus.sizey;
        double kz = (k - nz / 2) / torus.sizez;
        double lambda = fac * (kx * kx + ky * ky + kz * kz);
        
        int ind = index3d(i,j,k);
        
        cuDoubleComplex inp;
        inp.x = 0;
        inp.y = lambda * isf.dt / 2;
        
        isf.mask[index3d(i,j,k)] = exp_mycomplex(inp);

        //printf("%f %f \n", isf.mask[index3d(i,j,k)].x,isf.mask[index3d(i,j,k)].y);
        
      }
    }
  }

  printf("Done ISF_BuildSchroedinger \n"); 
}

// function [psi1,psi2] = SchroedingerFlow(obj,psi1,psi2)
//         % solves Schroedinger equation for dt time.
//         %
//             psi1 = fftshift(fftn(psi1)); psi2 = fftshift(fftn(psi2));
//             psi1 = psi1.*obj.SchroedingerMask;
//             psi2 = psi2.*obj.SchroedingerMask;
//             psi1 = ifftn(fftshift(psi1)); psi2 = ifftn(fftshift(psi2));
//         end

__global__ void ISF_SchroedingerFlow(cuDoubleComplex* psi1,
                                     cuDoubleComplex* psi2)
// Solves Schroedinger equation for dt time
// TODO: Implement this!
{
  // psi1 = fftshift(fftn(psi1)); psi2 = fftshift(fftn(psi2));
  // psi1 = psi1.*obj.SchroedingerMask;
  // psi2 = psi2.*obj.SchroedingerMask;
  // psi1 = ifftn(fftshift(psi1)); psi2 = ifftn(fftshift(psi2));
}

__global__ void ISF_VelocityOneForm(cuDoubleComplex* psi1, 
                                    cuDoubleComplex* psi2, 
                                  double hbar)
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
        
        cuDoubleComplex vxraw = cuCadd(
          cuCmul(cuConj(psi1[ind]),psi1[vxi]),
          cuCmul(cuConj(psi2[ind]),psi2[vxi])
          );
        cuDoubleComplex vyraw = cuCadd(
          cuCmul(cuConj(psi1[ind]),psi1[vyi]),
          cuCmul(cuConj(psi2[ind]),psi2[vyi])
          );
        cuDoubleComplex vzraw = cuCadd(
          cuCmul(cuConj(psi1[ind]),psi1[vzi]),
          cuCmul(cuConj(psi2[ind]),psi2[vzi])
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

__global__ void ISF_Neg_Normal_GaugeTransform(cuDoubleComplex* psi1,
                          cuDoubleComplex* psi2, cuDoubleComplex* q)
{
  cuDoubleComplex negi = make_cuDoubleComplex(0.0, -1.0 / torus.plen);

  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        cuDoubleComplex eiq = 
          exp_mycomplex( cuCmul(negi, q[ind]) );

        psi1[ind] = cuCmul(psi1[ind], eiq);
        psi2[ind] = cuCmul(psi2[ind], eiq);

      }
    }
  }
}

void ISF_PressureProject(cuDoubleComplex* psi1,
                          cuDoubleComplex* psi2)
{
  ISF_VelocityOneForm<<<1,1>>>(psi1, psi2, 1.0);
  cudaDeviceSynchronize();
  Torus_Div<<<1,1>>>(isf_cpu.vx, isf_cpu.vy, isf_cpu.vz); 
  cudaDeviceSynchronize();

  Torus_PoissonSolve(torus_cpu.out);

  ISF_Neg_Normal_GaugeTransform<<<1,1>>>(psi1,psi2,torus_cpu.fftbuf);
  cudaDeviceSynchronize(); 

}





