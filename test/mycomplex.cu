


__device__ cuDoubleComplex exp_mycomplex(cuDoubleComplex inp)
{
  cuDoubleComplex res;
  res.x = exp(inp.x) * cos(inp.y);
  res.y = exp(inp.x) * sin(inp.y);
  return res;
}

__device__ void div_mycomplex(cuDoubleComplex* n, double d)
{
  n -> x /= d;
  n -> y /= d;
}

__device__ void mul_mycomplex(cuDoubleComplex* n, double d)
{
  n -> x *= d;
  n -> y *= d;
}

__device__ double angle_mycomplex(cuDoubleComplex inp)
{
  double res =  atan2(inp.y, inp.x);
  return res;
}


