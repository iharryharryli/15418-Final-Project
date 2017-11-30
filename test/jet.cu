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

  mycomplex* psi1;
  mycomplex* psi2;

};

__constant__ para_t para;

__global__ void set_nozzle()
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

        if(abs(dx) < para.nozzle.len / 2.0
            && (dy * dy + dz * dz) < para.nozzle.rad * para.nozzle.rad)
        {
          para.isJet[ind] = 1;
          printf("%d %d %d \n", i, j ,k);
        }
        else
        {
          para.isJet[ind] = 0;
        }
      }
    }
  }

}

__global__ void psi_init_cuda()
{
  for(int i=0; i<torus.resx; i++)
  {
    for(int j=0; j<torus.resy; j++)
    {
      for(int k=0; k<torus.resz; k++)
      {
        int ind = index3d(i,j,k);
      }
    }
  }
}


void para_init(Torus* p, para_t* t)
{
  cudaMalloc(&(t -> psi1),
         sizeof(mycomplex) * (p -> plen));
  cudaMalloc(&(t -> psi2),
         sizeof(mycomplex) * (p -> plen));
  
  cudaMalloc(&(t -> isJet),
         sizeof(char) * (p -> plen));

  (t -> jet_velocity)[0] = 1.0;
  (t -> jet_velocity)[1] = 0.0;
  (t -> jet_velocity)[2] = 0.0;
  
  (t -> nozzle).center[0] = 2.0 - 1.7;
  (t -> nozzle).center[1] = 1.0 - 0.034;
  (t -> nozzle).center[2] = 1.0 + 0.066;
  (t -> nozzle).len = 0.5;
  (t -> nozzle).rad = 0.5;
  

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

  q -> hbar = 0.1;
  q -> dt = 1.0 / 48.0;
  cudaMalloc(&(q -> mask), 
        sizeof(mycomplex) * (p -> plen));

  cudaMemcpyToSymbol(torus, p, sizeof(Torus));
  cudaMemcpyToSymbol(isf, q, sizeof(ISF));

}



void jet_setup()
{
  Torus localtorus;
  ISF localISF;
  para_t localPara;

  isf_init(&localtorus, &localISF);
  para_init(&localtorus, &localPara);

  ISF_BuildSchroedinger<<<1,1>>>();

  set_nozzle<<<1,1>>>();



  cudaDeviceSynchronize();
}
