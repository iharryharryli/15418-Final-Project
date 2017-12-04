#include "ISF.cu"


struct nozzle_t
{
  double center[3];
  double len;
  double rad;
};

struct para_t
{
  double jet_velocity[3];
  nozzle_t nozzle;

  char* isJet;

  cuDoubleComplex* psi1;
  cuDoubleComplex* psi2;


  double kvec[3];
  double omega;
  double* phase;
  

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
        double px = i * torus.dx;
        double py = j * torus.dy;
        double pz = k * torus.dz;

        double dx = px - para.nozzle.center[0];
        double dy = py - para.nozzle.center[1];
        double dz = pz - para.nozzle.center[2];

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

        para.psi1[ind] = make_cuDoubleComplex(1.0, 0.0);
        para.psi2[ind] = make_cuDoubleComplex(0.01, 0.0);

      }
    }
  }

}

void para_init(Torus* p, ISF* q, para_t* t)
{
  cudaMalloc(&(t -> psi1),
         sizeof(cuDoubleComplex) * (p -> plen));
  cudaMalloc(&(t -> psi2),
         sizeof(cuDoubleComplex) * (p -> plen));
  
  cudaMalloc(&(t -> isJet),
         sizeof(char) * (p -> plen));

  cudaMalloc(&(t -> phase),
         sizeof(double) * (p -> plen));

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
// Includes a bunch of hard-coded values
{
  p -> resx = 64;
  p -> resy = 32;
  p -> resz = 32;
  p -> sizex = 4;
  p -> sizey = 2;
  p -> sizez = 2;
  p -> plen = (p -> resx) * (p -> resy) * (p -> resz);
  Torus_calc_ds(p);
  cudaMalloc(&(p -> out), sizeof(double) * (p -> plen));
  cudaMalloc(&(p -> fftbuf), sizeof(cuDoubleComplex) * (p -> plen));


  q -> hbar = 0.1;
  q -> dt = 1.0 / 48.0;
  cudaMalloc(&(q -> mask), 
        sizeof(cuDoubleComplex) * (p -> plen));


  cudaMalloc(&(q -> vx),
        sizeof(double) * (p -> plen));
  cudaMalloc(&(q -> vy),
        sizeof(double) * (p -> plen));
  cudaMalloc(&(q -> vz),
        sizeof(double) * (p -> plen));
  
  
  cudaMemcpyToSymbol(torus, p, sizeof(Torus));
  cudaMemcpyToSymbol(isf, q, sizeof(ISF));

}

__global__ void constrain_velocity_iter()
// A special procedure we need to do in order for the jet dynamics to work
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
            double amp1 = cuCabs(para.psi1[ind]);
            double amp2 = cuCabs(para.psi2[ind]);
            
            para.psi1[ind] = exp_mycomplex( 
                     make_cuDoubleComplex(0.0, para.phase[ind]));
            mul_mycomplex(&para.psi1[ind], amp1);

            /*if(para.psi1[ind].x < -0.34)
              printf("%d %d %d %f\n",i,j,k,para.psi1[ind].x);*/

            //printf("%d %d %d %f\n",i,j,k,para.psi1[ind].x);

            para.psi2[ind] = exp_mycomplex( 
                     make_cuDoubleComplex(0.0, para.phase[ind]));
            mul_mycomplex(&para.psi2[ind], amp2);

          }

        }
      }
    }
}

__global__ void print_psi()
{
    for(int i=0; i<torus.resx; i++)
    {
      for(int j=0; j<torus.resy; j++)
      {
        for(int k=0; k<torus.resz; k++)
        {
          int ind = index3d(i,j,k);
          printf("%f %f\n", para.psi1[ind].x, para.psi1[ind].y);
        }
      }
    }
}


void constrain_velocity()
{
  for(int i=0; i<10; i++)
  {
    constrain_velocity_iter<<<1,1>>>();
    cudaDeviceSynchronize();
    ISF_PressureProject(para_cpu.psi1, para_cpu.psi2);

    printf("iteration success \n");
  }

  //print_psi<<<1,1>>>(); cudaDeviceSynchronize(); 

}


void jet_setup()
{

  // Basic setup

  isf_init(&torus_cpu, &isf_cpu);
  para_init(&torus_cpu, &isf_cpu, &para_cpu);

  ISF_BuildSchroedinger<<<1,1>>>();

  // Jet-specific setup

  set_nozzle_and_phase_and_psi<<<1,1>>>();

  cudaDeviceSynchronize();

  constrain_velocity();

  // Main algorithm
  int itermax = 24;
  for (int i=0; i<itermax; i++)
  {
    // Simulate Incompressible Schroedinger Flow
    // ISF_SchroedingerFlow(para_cpu.psi1, para_cpu.psi2);
    // ISF_Normalize(para_cpu.psi1, para_cpu.psi2);
    // ISF_PressureProject(para_cpu.psi1, para_cpu.psi2);

    // Do particle advection


  }
}
