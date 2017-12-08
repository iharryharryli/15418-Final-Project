#define THREADS_PER_BLOCK 1024
#define PROFILE_N 100

int calc_numblock(int limit, int threadsPerBlock)
{
  return (limit + threadsPerBlock - 1) / threadsPerBlock;
}

__device__ __inline__ int check_limit(int limit)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < limit)
    return i;
  return -1;
}


// performance mearsurement

struct time_profile_t
{
  int count[PROFILE_N];
  double total[PROFILE_N];
  
  double s_time[PROFILE_N];
};

time_profile_t time_profile;
void tpstart(int i)
{
  time_profile.s_time[i] = CycleTimer::currentSeconds();
}

void tpend(int i)
{
  double e_time = CycleTimer::currentSeconds();
  time_profile.count[i] ++;
  time_profile.total[i] += e_time - time_profile.s_time[i];
}

void tpinit()
{
  for(int i=0; i<PROFILE_N; i++)
  {
    time_profile.count[i] = 0;
    time_profile.total[i] = 0.0;
  }
}



const char* profileNames[]=
{
  "Torus_Div", //0
  "PoissonSolveMain",
  "ISF_Neg_Normal_GaugeTransform",
  "ISF_VelocityOneForm",
  "ISF_Normalize",
  "constrain_velocity_iter", //5
  "fft", //6
  "ifft", //7
  "div2buf",
  "Torus_StaggeredSharp", //9
  "StaggeredAdvect", //10
  "fftshift", //11
  "ISF_ElementProduct", //12

};

void tpsummary()
{
  double total = 0.0;
  for(int i=0; i<PROFILE_N; i++)
  {
    total += time_profile.total[i];
  }
  printf("\n\n***** summary *****\n total time: %f \n\n", total);

  for(int i=0; i<PROFILE_N; i++)
  {
    if(time_profile.count[i] == 0) continue;
    printf("Profile for %s\n",profileNames[i]);
    printf("Total Percentage %f\n", 100 * time_profile.total[i] / total);
    printf("Avg Time %.10f\n\n", 
        time_profile.total[i] / time_profile.count[i]);
  }
}
