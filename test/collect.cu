
struct render_info_t
{
  int index;
  int value;
};

struct collect_data_t
{
  int content_len;
  int division_len;

  int* division;
  int* helper;

  render_info_t* output;
};

__constant__ collect_data_t collector;
collect_data_t collector_cpu;

void collect_create(int content_len, int division_len)
{
  collector_cpu.content_len = content_len;
  collector_cpu.division_len = division_len;
  cudaMalloc(&(collector_cpu.helper), sizeof(int) * content_len);
  cudaMalloc(&(collector_cpu.division), sizeof(int) * division_len);
  cudaMalloc(&(collector_cpu.output), 
              sizeof(collect_data_t) * division_len);

  cudaMemcpyToSymbol(collector, &collector_cpu, sizeof(collect_data_t));
}


void collect_sort(int* keys)
{
  tpstart(14);
  thrust::sort(thrust::device, keys, 
      keys + collector_cpu.content_len); 
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

__global__ void
collect_result_kernel(int limit, int* keys)
{
  int ind = check_limit(limit);
  if(ind<0) return;

  collector.output[ind].index = 
    keys[collector.division[ind]];
  
  if(ind+1 == limit)
    collector.output[ind].value = 
      limit - collector.division[ind];
  else
    collector.output[ind].value = 
      collector.division[ind+1] - collector.division[ind];

}

int collect_result(int* keys)
{
  tpstart(16);
  int* division_end = thrust::copy_if(
      thrust::device, collector_cpu.helper, 
      collector_cpu.helper + collector_cpu.content_len,
      collector_cpu.division, is_nonnegative());
  tpend(16);
  
  int limit = (division_end - collector_cpu.division);
  
  tpstart(17);
  int nb = calc_numblock(limit, THREADS_PER_BLOCK);
  collect_result_kernel<<<nb, THREADS_PER_BLOCK>>>
    (limit, keys);
  cudaDeviceSynchronize();  
  tpend(17);


  return limit;
}

int collect_main(int* keys)
{
  collect_sort(keys);
  collect_break(keys);
  return collect_result(keys);
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



