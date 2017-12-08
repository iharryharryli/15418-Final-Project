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
  int ind = check_limit(torus.plen);
  if(ind<0)return;
  cuDoubleComplex* psi1 = para.psi1;
  cuDoubleComplex* psi2 = para.psi2;
  /*for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
*/
        double psi_norm = 
          sqrt(psi1[ind].x*psi1[ind].x+psi1[ind].y*psi1[ind].y+
               psi2[ind].x*psi2[ind].x+psi2[ind].y*psi2[ind].y);
        
        div_mycomplex(&psi1[ind], psi_norm);
        div_mycomplex(&psi2[ind], psi_norm);
/*      }
    }
  }*/
}

void ISF_Normalize()
{
  tpstart(4);
  int nb = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK);
  ISF_Normalize_kernel<<<nb,THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
  tpend(4);

}

__global__ void ISF_BuildSchroedinger_kernel()
// Initializes the complex components of the field
{
  int ind = check_limit(torus.plen);
  if(ind<0)return;
  double nx = torus.resx, ny = torus.resy, nz = torus.resz;
  double fac = -4.0 * M_PI * M_PI * isf.hbar;

  int i,j,k;
  getCoords(ind,&i,&j,&k);
  
        double kx = (i - nx / 2) / torus.sizex;
        double ky = (j - ny / 2) / torus.sizey;
        double kz = (k - nz / 2) / torus.sizez;
        double lambda = fac * (kx * kx + ky * ky + kz * kz);
        
        
        cuDoubleComplex inp;
        inp.x = 0;
        inp.y = lambda * isf.dt / 2;
        
        isf.mask[index3d(i,j,k)] = exp_mycomplex(inp);

       
}

void ISF_BuildSchroedinger()
{
  int nb = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK);
  ISF_BuildSchroedinger_kernel<<<nb,THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
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

__global__ void ISF_ElementProduct(cuDoubleComplex* a, 
                                cuDoubleComplex* b)
{
  int i = check_limit(torus.plen);
  if(i<0) return;
   
  a[i] = cuCmul(a[i], b[i]);
}
        
void ISF_SchroedingerFlow()
// Solves Schroedinger equation for dt time
{
  fftn(para_cpu.psi1);
  fftn(para_cpu.psi2);
  cudaDeviceSynchronize();

  tpstart(11);
  int nb = calc_numblock(torus_cpu.plen/2, THREADS_PER_BLOCK);
  fftshift<<<nb,THREADS_PER_BLOCK>>>(para_cpu.psi1); 
  fftshift<<<nb,THREADS_PER_BLOCK>>>(para_cpu.psi2);
  cudaDeviceSynchronize();
  tpend(11);


  tpstart(12);
  // Elementwise multiplication, easy to make parallel
  int nb2 = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK);
  ISF_ElementProduct<<<nb2,THREADS_PER_BLOCK>>>
    (para_cpu.psi1, isf_cpu.mask);
  ISF_ElementProduct<<<nb2,THREADS_PER_BLOCK>>>
    (para_cpu.psi2, isf_cpu.mask);
  cudaDeviceSynchronize();
  tpend(12);
  

  tpstart(11);
  // Matlab code used fftshift here, which I believe is wrong
  // Doesn't matter here when we use even sized grid though
  fftshift<<<nb,THREADS_PER_BLOCK>>>(para_cpu.psi1);
  fftshift<<<nb,THREADS_PER_BLOCK>>>(para_cpu.psi2);
  cudaDeviceSynchronize();
  tpend(11);

  ifftn(para_cpu.psi1);
  ifftn(para_cpu.psi2);
  cudaDeviceSynchronize();
}

__global__ void ISF_VelocityOneForm_kernel(double hbar)
{
  int ind = check_limit(torus.plen);
  if(ind<0) return;
  cuDoubleComplex* psi1 = para.psi1;
  cuDoubleComplex* psi2 = para.psi2;
  int i,j,k;
  getCoords(ind, &i, &j, &k);

  /*for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {*/
        int ixp = (i + 1) % torus.resx;
        int iyp = (j + 1) % torus.resy;
        int izp = (k + 1) % torus.resz;

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

 /*     }
    }
  }*/
}

void ISF_VelocityOneForm(double hbar)
{
  tpstart(3);
  int nb = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK); 
  ISF_VelocityOneForm_kernel<<<nb,THREADS_PER_BLOCK>>>
    (hbar);
  cudaDeviceSynchronize();
  tpend(3);
}

__global__ void ISF_Neg_Normal_GaugeTransform()
{
  int ind = check_limit(torus.plen);
  if(ind<0)return;
  cuDoubleComplex negi = make_cuDoubleComplex(0.0, -1.0 / torus.plen);
  cuDoubleComplex* psi1 = para.psi1;
  cuDoubleComplex* psi2 = para.psi2;
  cuDoubleComplex* q = torus.fftbuf;
  /*for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);*/
        cuDoubleComplex eiq = 
          exp_mycomplex( cuCmul(negi, q[ind]) );

        psi1[ind] = cuCmul(psi1[ind], eiq);
        psi2[ind] = cuCmul(psi2[ind], eiq);

   /*   }
    }
  }*/
}

void ISF_PressureProject()
{

  ISF_VelocityOneForm(1.0);
  
  tpstart(0);
  int nb = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK);
  Torus_Div<<<nb,THREADS_PER_BLOCK>>>(); 
  cudaDeviceSynchronize();
  tpend(0);

  Torus_PoissonSolve();

  tpstart(2);
  ISF_Neg_Normal_GaugeTransform<<<nb,THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize(); 
  tpend(2);
  

}







