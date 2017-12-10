
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
  tpstart(13);
  int nb = calc_numblock(collector_cpu.content_len, THREADS_PER_BLOCK);
  collect_init_kernel<<<nb, THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
  tpend(13);
}

void collect_sort(int* keys)
{
  tpstart(14);
  thrust::device_ptr<int> dev_keys(keys);
  thrust::device_ptr<int> dev_values(collector_cpu.content);
  thrust::sort_by_key(dev_keys, dev_keys + collector_cpu.content_len, 
      dev_values);
  tpend(14);
}

__global__ void 
collect_break_kernel(int* keys)
{
  int ind = check_limit(collector.content_len); 
  if(ind < 0) return;
  if(ind == 0)
    collector.helper[ind] = ind;
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
  tpstart(15);
  int nb = calc_numblock(collector_cpu.content_len, THREADS_PER_BLOCK);
  collect_break_kernel<<<nb, THREADS_PER_BLOCK>>>(keys);
  cudaDeviceSynchronize();
  tpend(15);
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
  //int* out;
  //cudaMalloc(&out, sizeof(int) * collector_cpu.content_len);
  tpstart(16);
  int* division_end = thrust::copy_if(
      thrust::device, collector_cpu.helper, 
      collector_cpu.helper + collector_cpu.content_len,
      collector_cpu.division, is_nonnegative());
  /*thrust::inclusive_scan(thrust::device, helper_ptr, 
      helper_ptr + collector_cpu.content_len, out);*/
  tpend(16);
  //return 0;
  return (division_end - collector_cpu.division);
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
collect_test_kernel(int* keys, int len, int div)
{
  for(int i=0; i<len; i++)
    keys[i] = i % 100;
}

void collect_test()
{
  int len = 1000000, div = 262144;
  collect_create(len, div);

  int* keys;
  cudaMalloc(&keys, sizeof(int) * len);
  collect_test_kernel<<<1,1>>>(keys, len, div); cudaDeviceSynchronize();
 
  tpinit();

  printf("result: %d \n", collect_main(keys));

  //collect_print<<<1,1>>>(keys);  cudaDeviceSynchronize(); 

  tpsummary();
}



