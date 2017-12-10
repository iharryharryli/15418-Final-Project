
struct collect_data_t
{
  int content_len;
  int division_len;
  int* content;

  int* division;
  
  int* helper;
};

__constant__ collect_data_t collector;
collect_data_t collector_cpu;

void collect_create(int content_len, int division_len)
{
  collector_cpu.content_len = content_len;
  collector_cpu.division_len = division_len;
  cudaMalloc(&(collector_cpu.content), sizeof(int) * content_len);
  cudaMalloc(&(collector_cpu.helper), sizeof(int) * content_len);
  cudaMalloc(&(collector_cpu.division), sizeof(int) * division_len);

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

void collect_sort(int* keys)
{
  thrust::device_ptr<int> dev_keys(keys);
  thrust::device_ptr<int> dev_values(collector_cpu.content);
  thrust::sort_by_key(dev_keys, dev_keys + collector_cpu.content_len, 
      dev_values);
}

__global__ void 
collect_break_kernel(int* keys)
{
  int ind = check_limit(collector.content_len); 
  if(ind < 0) return;
  if(ind == 0)
    collector.helper[ind] = 0;
  else
  {
    //printf("%d %d %d \n", collector.content[ind - 1],
    //    collector.content[ind], ind);
    if(keys[ind - 1] !=
         keys[ind])
    {
      collector.helper[ind] = ind;
    }
    else
      collector.helper[ind] = -1;
  }
}

void collect_break(int* keys)
{
  int nb = calc_numblock(collector_cpu.content_len, THREADS_PER_BLOCK);
  collect_break_kernel<<<nb, THREADS_PER_BLOCK>>>(keys);
  cudaDeviceSynchronize();
}

struct is_nonnegative
{
  __host__ __device__
    bool operator()(const int x)
    {
      return (x >= 0);
    }
};

int collect_result()
{
  thrust::device_ptr<int> helper_ptr(collector_cpu.helper);
  thrust::device_ptr<int> division_ptr(collector_cpu.division);
  thrust::device_ptr<int> division_end = thrust::copy_if(helper_ptr, 
      helper_ptr + collector_cpu.content_len,
      division_ptr, is_nonnegative());
  return (division_end - division_ptr);
}

int collect_main(int* keys)
{
  collect_init();
  collect_sort(keys);
  collect_break(keys);
  return collect_result();
}



// *********** TEST ************

__global__ void
collect_print(int* keys)
{
  for(int i=0; i<collector.division_len; i++)
  {
    printf("%d \n", collector.division[i]);
  }
}

__global__ void
collect_test_kernel(int* keys)
{
  keys[0] = 4;
  keys[1] = 2;
  keys[2] = 1;
  keys[3] = 1;
  keys[4] = 2;
  keys[5] = 4;
  keys[6] = 1;
  keys[7] = 1;
  keys[8] = 3;
  keys[9] = 1;
}

void collect_test()
{
  int len = 10;
  collect_create(len, 10);

  int* keys;
  cudaMalloc(&keys, sizeof(int) * len);
  collect_test_kernel<<<1,1>>>(keys); cudaDeviceSynchronize();
  
  printf("result: %d \n", collect_main(keys));

  collect_print<<<1,1>>>(keys);  cudaDeviceSynchronize(); 
}



