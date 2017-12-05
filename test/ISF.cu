#include "Torus.cu"

struct ISF
{
  double hbar;
  double dt;
  cuDoubleComplex* mask;
};

struct para_t
{
  double jet_velocity[3];

  char* isJet;

  cuDoubleComplex* psi1;
  cuDoubleComplex* psi2;


  double kvec[3];
  double omega;
  double* phase;
  

};

__constant__ para_t para;
para_t para_cpu;
__constant__ ISF isf;
ISF isf_cpu;

__global__ void ISF_Normalize_kernel()
{
  cuDoubleComplex* psi1 = para.psi1;
  cuDoubleComplex* psi2 = para.psi2;
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

void ISF_Normalize()
{
  ISF_Normalize_kernel<<<1,1>>>();
}

__global__ void ISF_BuildSchroedinger_kernel()
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

void ISF_BuildSchroedinger()
{
  ISF_BuildSchroedinger_kernel<<<1,1>>>();
}

// function [psi1,psi2] = SchroedingerFlow(obj,psi1,psi2)
//         % solves Schroedinger equation for dt time.
//         %
//             psi1 = fftshift(fftn(psi1)); psi2 = fftshift(fftn(psi2));
//             psi1 = psi1.*obj.SchroedingerMask;
//             psi2 = psi2.*obj.SchroedingerMask;
//             psi1 = ifftn(fftshift(psi1)); psi2 = ifftn(fftshift(psi2));
//         end


//*********** not tested! ***********
void ISF_SchroedingerFlow()
// Solves Schroedinger equation for dt time
{
  cudaMemcpyFromSymbol(para_cpu.psi1, para.psi1, 
    sizeof(cuDoubleComplex) * torus_cpu.plen);
  cudaMemcpyFromSymbol(para_cpu.psi2, para.psi2, 
    sizeof(cuDoubleComplex) * torus_cpu.plen);
  cufftHandle plan1 = fftn(para_cpu.psi1);
  cufftHandle plan2 = fftn(para_cpu.psi2);
  cudaDeviceSynchronize();
  fftshift<<<1,1>>>(para_cpu.psi1); 
  fftshift<<<1,1>>>(para_cpu.psi2);
  cudaDeviceSynchronize();

  int len = torus_cpu.resx * torus_cpu.resy * torus_cpu.resz;

  // Elementwise multiplication, easy to make parallel
  for (int i=0; i<len; i++)
  {
    para_cpu.psi1[i] = cuCmul(para_cpu.psi1[i], isf_cpu.mask[i]);
    para_cpu.psi2[i] = cuCmul(para_cpu.psi2[i], isf_cpu.mask[i]);
  }

  // Matlab code used fftshift here, which I believe is wrong
  // Doesn't matter here when we use even sized grid though
  ifftshift<<<1,1>>>(para_cpu.psi1);
  ifftshift<<<1,1>>>(para_cpu.psi2);
  cudaDeviceSynchronize();
  ifftn(para_cpu.psi1, plan1);
  ifftn(para_cpu.psi2, plan2);
  cudaDeviceSynchronize();
  cudaMemcpyToSymbol(para.psi1, para_cpu.psi1, 
    sizeof(cuDoubleComplex) * torus_cpu.plen);
  cudaMemcpyToSymbol(para.psi2, para_cpu.psi2, 
    sizeof(cuDoubleComplex) * torus_cpu.plen);
}

__global__ void ISF_VelocityOneForm_kernel()
{
  cuDoubleComplex* psi1 = para.psi1;
  cuDoubleComplex* psi2 = para.psi2;
  double hbar = isf.hbar;
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

        torus.vx[ind] = angle_mycomplex(vxraw) * hbar;
        torus.vy[ind] = angle_mycomplex(vyraw) * hbar;
        torus.vz[ind] = angle_mycomplex(vzraw) * hbar;

        /*if(isf.vx[ind] > 1.2)
          printf("%f\n", isf.vx[ind]);*/

      }
    }
  }
}

void ISF_VelocityOneForm()
{
  ISF_VelocityOneForm_kernel<<<1,1>>>();
}

__global__ void ISF_Neg_Normal_GaugeTransform()
{
  cuDoubleComplex negi = make_cuDoubleComplex(0.0, -1.0 / torus.plen);
  cuDoubleComplex* psi1 = para.psi1;
  cuDoubleComplex* psi2 = para.psi2;
  cuDoubleComplex* q = torus.fftbuf;
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

void ISF_PressureProject()
{
  ISF_VelocityOneForm();
  cudaDeviceSynchronize();
  Torus_Div<<<1,1>>>(); 
  cudaDeviceSynchronize();

  Torus_PoissonSolve();

  ISF_Neg_Normal_GaugeTransform<<<1,1>>>();
  cudaDeviceSynchronize(); 

}





