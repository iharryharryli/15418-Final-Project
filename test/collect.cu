
struct division_t
{
  int from;
  int to;
};

struct collect_data_t
{
  int content_len;
  int division_len;
  int* content;
  division_t* division;
};

__constant__ collect_data_t collector;
collect_data_t collector_cpu;

void collect_create(int content_len, int division_len)
{
  collector_cpu.content_len = content_len;
  collector_cpu.division_len = division_len;
  cudaMalloc(&(collector_cpu.content), sizeof(int) * content_len);
  cudaMalloc(&(collector_cpu.division), 
      sizeof(division_t) * division_len);

  cudaMemcpyToSymbol(collector, &collector_cpu, sizeof(collect_data_t));
}

__global__ void 
collect_init_kernel()
{
  int ind = check_limit(collector.content_len);
  if(ind < 0) return;
  collector.content[ind] = ind;
}

void collect_init()
{
  int nb = calc_numblock(collector_cpu.content_len, THREADS_PER_BLOCK);
  collect_init_kernel<<<nb, THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
}

void collect_main(int* keys)
{
  thrust::device_ptr<int> dev_keys(keys);
  thrust::device_ptr<int> dev_values(collector_cpu.content);
  thrust::sort_by_key(dev_keys, dev_keys + collector_cpu.content_len, 
      dev_values);
}

// *********** TEST ************

__global__ void
collect_print()
{
  for(int i=0; i<collector.content_len; i++)
  {
    printf("%d \n", collector.content[i]);
  }
}

__global__ void
collect_test_kernel(int* keys)
{
  keys[0] = 2;
  keys[1] = 3;
  keys[2] = 10;
  keys[3] = 6;
  keys[4] = 1;
  keys[5] = 8;
  keys[6] = 7;
  keys[7] = 5;
  keys[8] = 4;
  keys[9] = 9;
}

void collect_test()
{
  int len = 10;
  collect_create(len, 10);
  collect_init(); 

  int* keys;
  cudaMalloc(&keys, sizeof(int) * len);
  collect_test_kernel<<<1,1>>>(keys); cudaDeviceSynchronize();
  
  collect_main(keys);

  collect_print<<<1,1>>>();  cudaDeviceSynchronize(); 
}



