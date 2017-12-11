#include "ISF.cu"
#include "particle.cu"
#include "collect.cu"

struct nozzle_t
{
	double center[3];
	double len;
	double rad;
};

__constant__ nozzle_t nozzle;
nozzle_t nozzle_cpu;

__global__ void set_nozzle_and_phase_and_psi_kernel()
{
  int ind = check_limit(torus.plen);
  if(ind<0)return;

  int i,j,k;
  getCoords(ind, &i, &j, &k);
  
        double px = i * torus.dx;
        double py = j * torus.dy;
        double pz = k * torus.dz;

        double dx = px - nozzle.center[0];
        double dy = py - nozzle.center[1];
        double dz = pz - nozzle.center[2];

        if(abs(dx) <= nozzle.len / 2.0
            && (dy * dy + dz * dz) <= nozzle.rad * nozzle.rad)
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

void set_nozzle_and_phase_and_psi()
{
  int nb = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK);
  set_nozzle_and_phase_and_psi_kernel<<<nb,THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
}

void para_init(Torus* p, ISF* q, para_t* t, nozzle_t* n)
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
  p -> resx = 64;
  p -> resy = 32;
  p -> resz = 32;
  p -> sizex = 4;
  p -> sizey = 2;
  p -> sizez = 2;
  p -> plen = (p -> resx) * (p -> resy) * (p -> resz);
  p -> yzlen = (p -> resy) * (p -> resz);
  Torus_calc_ds(p);
  cudaMalloc(&(p -> div), sizeof(double) * (p -> plen));
  cudaMalloc(&(p -> fftbuf), sizeof(cuDoubleComplex) * (p -> plen));
  cufftPlan3d(&(p -> fftplan), 
      torus_cpu.resx, torus_cpu.resy, torus_cpu.resz, CUFFT_Z2Z);


  q -> hbar = 0.1;
  q -> dt = 1.0 / 48.0;
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

__global__ void constrain_velocity_iter(double t)
// A special procedure we need to do in order for the jet dynamics to work
{

    int ind = check_limit(torus.plen);
    if(ind < 0) return;
    /*for(int i=0; i<torus.resx; i++)
    {
      for(int j=0; j<torus.resy; j++)
      {
        for(int k=0; k<torus.resz; k++)
        {
          int ind = index3d(i,j,k);*/
          
          if(para.isJet[ind] == 1)
          {
            double amp1 = cuCabs(para.psi1[ind]);
            double amp2 = cuCabs(para.psi2[ind]);
            
            para.psi1[ind] = exp_mycomplex( 
                     make_cuDoubleComplex(0.0, para.phase[ind] - para.omega * t));
            mul_mycomplex(&para.psi1[ind], amp1);

            para.psi2[ind] = exp_mycomplex( 
                     make_cuDoubleComplex(0.0, para.phase[ind] - para.omega * t));
            mul_mycomplex(&para.psi2[ind], amp2);

          }
/*
        }
      }
    }*/
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
		// px[i] = particles.x[i];
		// py[i] = particles.y[i];
		// pz[i] = particles.z[i];
		// printf("%f %f %f\n", particles.x[i],
		// 					particles.y[i], particles.z[i]);
	}
}

void constrain_velocity(double t)
{

    tpstart(5);
    int nb = calc_numblock(torus_cpu.plen, THREADS_PER_BLOCK); 
    constrain_velocity_iter<<<nb,THREADS_PER_BLOCK>>>(t);
    cudaDeviceSynchronize();
    tpend(5);
    ISF_PressureProject();
}


__global__ void
particle_birth_kernel()
{
	for(int i=0; i<particles.num_particles; i++)
	{
		double rt = ((double)i) / particles.num_particles;
		rt *= 2 * M_PI;

		particles.x[i] = nozzle.center[0];
		particles.y[i] = nozzle.center[1] + 0.9 * nozzle.rad * cos(rt);
		particles.z[i] = nozzle.center[2] + 0.9 * nozzle.rad * sin(rt);
	}
}

void particle_birth(int num)
{

  particles_cpu.num_particles = num;
  cudaMalloc(&(particles_cpu.x), sizeof(double) * num);
  cudaMalloc(&(particles_cpu.y), sizeof(double) * num);
  cudaMalloc(&(particles_cpu.z), sizeof(double) * num);
  
  cudaMalloc(&(particles_cpu.pixel_index), sizeof(int) * num);

  cudaMemcpyToSymbol(particles, &particles_cpu, sizeof(particles_t)); 
  
  particle_birth_kernel<<<1,1>>>();
  cudaDeviceSynchronize();
}


void jet_setup(int particleCount)
{
  // init timer
  tpinit();

  // Basic setup

  isf_init(&torus_cpu, &isf_cpu);
  para_init(&torus_cpu, &isf_cpu, &para_cpu, &nozzle_cpu);

  ISF_BuildSchroedinger();

  // Jet-specific setup
  
 
  set_nozzle_and_phase_and_psi();

  cudaDeviceSynchronize();

  for(int i=0; i<10; i++)
  {
    constrain_velocity(0.0);
  }

  printf("Initialization Done! \n");

  //print_psi<<<1,1>>>();
  //cudaDeviceSynchronize(); 

  // generate particles
  particle_birth(particleCount);

  // Main algorithm
  for (int i=0; i<500; i++)
  {
    // Simulate Incompressible Schroedinger Flow
    ISF_SchroedingerFlow();
    ISF_Normalize();
    ISF_PressureProject();

    constrain_velocity((i+1) * isf_cpu.dt);

    // Do particle advection

    ISF_VelocityOneForm(isf_cpu.hbar);
    Torus_StaggeredSharp();
    StaggeredAdvect();

    //printf("Iteration %d done!\n", i);
  }


  tpsummary();

  //print_psi<<<1,1>>>(); cudaDeviceSynchronize();  
  //print_particles<<<1,1>>>(); cudaDeviceSynchronize();
  
}
