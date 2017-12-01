#include "ISF.cu"


struct nozzle_t
{
  float center[3];
  float len;
  float rad;
};

struct para_t
{
  float jet_velocity[3];
  nozzle_t nozzle;

  char* isJet;

  cuFloatComplex* psi1;
  cuFloatComplex* psi2;


  float kvec[3];
  float omega;
  float* phase;
  

};

__constant__ para_t para;
para_t para_cpu;

__global__ void set_nozzle_and_phase_and_psi()
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
        float px = i * torus.dx;
        float py = j * torus.dy;
        float pz = k * torus.dz;

        float dx = px - para.nozzle.center[0];
        float dy = py - para.nozzle.center[1];
        float dz = pz - para.nozzle.center[2];

        if(abs(dx) <= para.nozzle.len / 2.0
            && (dy * dy + dz * dz) <= para.nozzle.rad * para.nozzle.rad)
        {
          para.isJet[ind] = 1;
          //printf("%d %d %d \n", i, j ,k);
        }
        else
        {
          para.isJet[ind] = 0;
        }

        para.phase[ind] = 
          para.kvec[0]*px + para.kvec[1]*py + para.kvec[2]*pz;

        para.psi1[ind] = make_cuFloatComplex(1.0, 0.0);
        para.psi2[ind] = make_cuFloatComplex(0.01, 0.0);

      }
    }
  }

}



void para_init(Torus* p, ISF* q, para_t* t)
{
  cudaMalloc(&(t -> psi1),
         sizeof(cuFloatComplex) * (p -> plen));
  cudaMalloc(&(t -> psi2),
         sizeof(cuFloatComplex) * (p -> plen));
  
  cudaMalloc(&(t -> isJet),
         sizeof(char) * (p -> plen));

  cudaMalloc(&(t -> phase),
         sizeof(float) * (p -> plen));

  (t -> jet_velocity)[0] = 1.0;
  (t -> jet_velocity)[1] = 0.0;
  (t -> jet_velocity)[2] = 0.0;
  
  (t -> nozzle).center[0] = 2.0 - 1.7;
  (t -> nozzle).center[1] = 1.0 - 0.034;
  (t -> nozzle).center[2] = 1.0 + 0.066;
  (t -> nozzle).len = 0.5;
  (t -> nozzle).rad = 0.5;

  (t -> kvec)[0] = (t -> jet_velocity)[0] / (q -> hbar);
  (t -> kvec)[1] = (t -> jet_velocity)[1] / (q -> hbar); 
  (t -> kvec)[2] = (t -> jet_velocity)[2] / (q -> hbar); 

  t -> omega = 0.0;
  for(int i=0; i<3; i++)
  {
    t -> omega += ((t -> jet_velocity)[i])*((t -> jet_velocity)[i]);
  }

  t -> omega /= 2.0 * (q -> hbar);
  

  cudaMemcpyToSymbol(para, t, sizeof(para_t));
}

void isf_init(Torus* p, ISF* q)
{
  p -> resx = 64;
  p -> resy = 32;
  p -> resz = 32;
  p -> sizex = 4;
  p -> sizey = 2;
  p -> sizez = 2;
  p -> plen = (p -> resx) * (p -> resy) * (p -> resz);
  Torus_calc_ds(p);
  cudaMalloc(&(p -> out), sizeof(float) * (p -> plen));
  cudaMalloc(&(p -> fftbuf), sizeof(cuFloatComplex) * (p -> plen));


  q -> hbar = 0.1;
  q -> dt = 1.0 / 48.0;
  cudaMalloc(&(q -> mask), 
        sizeof(cuFloatComplex) * (p -> plen));


  cudaMalloc(&(q -> vx),
        sizeof(float) * (p -> plen));
  cudaMalloc(&(q -> vy),
        sizeof(float) * (p -> plen));
  cudaMalloc(&(q -> vz),
        sizeof(float) * (p -> plen));
  
  
  cudaMemcpyToSymbol(torus, p, sizeof(Torus));
  cudaMemcpyToSymbol(isf, q, sizeof(ISF));

}

__global__ void constrain_velocity_iter()
{
    for(int i=0; i<torus.resx; i++)
    {
      for(int j=0; j<torus.resy; j++)
      {
        for(int k=0; k<torus.resz; k++)
        {
          int ind = index3d(i,j,k);
          
          if(para.isJet[ind] == 1)
          {
            float amp1 = cuCabsf(para.psi1[ind]);
            float amp2 = cuCabsf(para.psi2[ind]);
            
            para.psi1[ind] = exp_mycomplex( 
                     make_cuFloatComplex(0.0, para.phase[ind]));
            mul_mycomplex(&para.psi1[ind], amp1);

            /*if(para.psi1[ind].x < -0.34)
              printf("%d %d %d %f\n",i,j,k,para.psi1[ind].x);*/

            para.psi2[ind] = exp_mycomplex( 
                     make_cuFloatComplex(0.0, para.phase[ind]));
            mul_mycomplex(&para.psi2[ind], amp2);

          }

        }
      }
    }
}

void constrain_velocity()
{
  for(int i=0; i<1; i++)
  {
    constrain_velocity_iter<<<1,1>>>();
    cudaDeviceSynchronize();
    ISF_PressureProject(para_cpu.psi1, para_cpu.psi2);

    printf("iteration success \n");
  }
}


void jet_setup()
{

  isf_init(&torus_cpu, &isf_cpu);
  para_init(&torus_cpu, &isf_cpu, &para_cpu);

  ISF_BuildSchroedinger<<<1,1>>>();

  set_nozzle_and_phase_and_psi<<<1,1>>>();

  cudaDeviceSynchronize();

  constrain_velocity();

  cudaDeviceSynchronize();

  
}
