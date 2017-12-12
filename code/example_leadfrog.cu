#include "ISF.cu"
#include "particle.cu"

int n_particles = 10000;
int resx = 128;
int resy = 64;
int resz = 64;
int sizex = 10;
int sizey = 10;
int sizez = 10;
double hbar = 0.1;
double dt = 1.0/24.0;
int tmax = 85;
double bgv_x = -0.2;
double bgv_y = 0;
double bgv_z = 0;
double r1 = 1.5;
double r2 = 0.9;

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

  // (t -> jet_velocity)[0] = 0.0;
  // (t -> jet_velocity)[1] = 0.0;
  // (t -> jet_velocity)[2] = 0.0;
  
  n->center[0] = 2.0 - 1.7;
  n->center[1] = 1.0 - 0.034;
  n->center[2] = 1.0 + 0.066;
  n->len = 0.5;
  n->rad = 0.5;

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
  cudaMemcpyToSymbol(nozzle, n, sizeof(nozzle_t));
}

void isf_init(Torus* p, ISF* q)
// Includes a bunch of hard-coded values
{
  p -> resx = resx;
  p -> resy = resy;
  p -> resz = resz;
  p -> sizex = sizex;
  p -> sizey = sizey;
  p -> sizez = sizez;
  p -> plen = (p -> resx) * (p -> resy) * (p -> resz);
  p -> yzlen = (p -> resy) * (p -> resz);
  Torus_calc_ds(p);
  cudaMalloc(&(p -> div), sizeof(double) * (p -> plen));
  cudaMalloc(&(p -> fftbuf), sizeof(cuDoubleComplex) * (p -> plen));


  q -> hbar = hbar;
  q -> dt = dt;
  cudaMalloc(&(q -> mask), 
        sizeof(cuDoubleComplex) * (p -> plen));

  cudaMalloc(&(p -> vx),
        sizeof(double) * (p -> plen));
  cudaMalloc(&(p -> vy),
        sizeof(double) * (p -> plen));
  cudaMalloc(&(p -> vz),
        sizeof(double) * (p -> plen));
  
  
  cudaMemcpyToSymbol(torus, p, sizeof(Torus));
  cudaMemcpyToSymbol(isf, q, sizeof(ISF));

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

__global__ void print_particles()
{
  for(int i=0; i<particles.num_particles; i++)
  {
    printf("%f %f %f\n", particles.x[i],
              particles.y[i], particles.z[i]);
  }
}

void run_leapfrog()
{

  // Basic setup

  isf_init(&torus_cpu, &isf_cpu);
  para_init(&torus_cpu, &isf_cpu, &para_cpu);

  ISF_BuildSchroedinger();

  // Jet-specific setup
  
 
  set_nozzle_and_phase_and_psi();

  cudaDeviceSynchronize();

  for(int i=0; i<10; i++)
  {
    constrain_velocity(0.0);
    printf("iteration success \n");
  }

  //print_psi<<<1,1>>>();
  //cudaDeviceSynchronize(); 

  // generate particles
  particle_birth(50);

  // Main algorithm
  for (int i=0; i<5; i++)
  {
    // Simulate Incompressible Schroedinger Flow
    ISF_SchroedingerFlow();
    ISF_Normalize();
    ISF_PressureProject();

    constrain_velocity((i+1) * isf_cpu.dt);

    // Particle birth
    // rt = rand(n_particles,1)*2*pi;
    // newx = nozzle_cen(1)*ones(size(rt));
    // newy = nozzle_cen(2) + 0.9*nozzle_rad*cos(rt);
    // newz = nozzle_cen(3) + 0.9*nozzle_rad*sin(rt);
    // particle.x = [particle.x;newx];
    // particle.y = [particle.y;newy];
    // particle.z = [particle.z;newz];

    // Do particle advection

    ISF_VelocityOneForm(isf_cpu.hbar);
    Torus_StaggeredSharp();
    StaggeredAdvect();

    printf("Iteration %d done!\n", i);

  }

  //print_psi<<<1,1>>>(); cudaDeviceSynchronize();  
  print_particles<<<1,1>>>(); cudaDeviceSynchronize();
}
