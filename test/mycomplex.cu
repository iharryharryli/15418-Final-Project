


__device__ cuFloatComplex exp_mycomplex(cuFloatComplex inp)
{
  cuFloatComplex res;
  res.x = exp(inp.x) * cos(inp.y);
  res.y = exp(inp.x) * sin(inp.y);
  return res;
}

__device__ void div_mycomplex(cuFloatComplex* n, float d)
{
  n -> x /= d;
  n -> y /= d;
}

__device__ void mul_mycomplex(cuFloatComplex* n, float d)
{
  n -> x *= d;
  n -> y *= d;
}

__device__ float angle_mycomplex(cuFloatComplex inp)
{
  float res =  atan2(inp.y, inp.x);
  return res;
}


