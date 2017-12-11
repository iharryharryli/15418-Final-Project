#include "depend.h"

struct particles_t
{
  double *x;
  double *y;
  double *z;

  int spawn_rate;
  int curr_particles;
  int max_particles;
  int num_particles;

  int* pixel_index;
};

__constant__ particles_t particles;
particles_t particles_cpu;

__device__ __inline__ void 
StaggeredVelocity(double px, double py, double pz,
    double* ux, double* uy, double* uz)
// evaluates velocity at (px,py,pz) in the grid torus with staggered
// velocity vector field vx,vy,vz
{
    px = fmod(px, (double)torus.sizex);
    py = fmod(py, (double)torus.sizey);
    pz = fmod(pz, (double)torus.sizez);

    int ix = floor(px / torus.dx);
    int iy = floor(py / torus.dy);
    int iz = floor(pz / torus.dz);

    int ixp = (ix + 1) % torus.resx;
    int iyp = (iy + 1) % torus.resy;
    int izp = (iz + 1) % torus.resz;

    int ind0 =    index3d(ix, iy, iz);
    int indxp =   index3d(ixp, iy, iz);
    int indyp =   index3d(ix, iyp, iz);
    int indzp =   index3d(ix, iy, izp); 
    int indxpyp = index3d(ixp, iyp, iz);
    int indypzp = index3d(ix, iyp, izp);
    int indxpzp = index3d(ixp, iy, izp);    
    
    double wx = px - ix * torus.dx;
    double wy = py - iy * torus.dy;
    double wz = pz - iz * torus.dz;

    // Interpolate between velocities at grid points
    *ux = (1 - wz) * ((1 - wy) * torus.vx[ind0] + wy * torus.vx[indyp]) 
              + wz * ((1 - wy) * torus.vx[indzp] + wy * torus.vx[indypzp]);
    *uy = (1 - wx) * ((1 - wz) * torus.vy[ind0] + wz * torus.vy[indzp]) 
              + wx * ((1 - wz) * torus.vy[indxp] + wz * torus.vy[indxpzp]);
    *uz = (1 - wy) * ((1 - wx) * torus.vz[ind0] + wx * torus.vz[indxp]) 
              + wy * ((1 - wx) * torus.vz[indyp] + wx * torus.vz[indxpyp]);
}

__device__ __inline__ double
fma_lerp(double v0, double v1, double t)
{
  return fma(t, v1, fma(-t, v0, v0));
}

__device__ __inline__ void 
StaggeredVelocity_fma(double px, double py, double pz,
    double* ux, double* uy, double* uz)
// evaluates velocity at (px,py,pz) in the grid torus with staggered
// velocity vector field vx,vy,vz
{
    px = fmod(px, (double)torus.sizex);
    py = fmod(py, (double)torus.sizey);
    pz = fmod(pz, (double)torus.sizez);

    int ix = floor(px / torus.dx);
    int iy = floor(py / torus.dy);
    int iz = floor(pz / torus.dz);

    int ixp = (ix + 1) % torus.resx;
    int iyp = (iy + 1) % torus.resy;
    int izp = (iz + 1) % torus.resz;

    int ind0 =    index3d(ix, iy, iz);
    int indxp =   index3d(ixp, iy, iz);
    int indyp =   index3d(ix, iyp, iz);
    int indzp =   index3d(ix, iy, izp); 
    int indxpyp = index3d(ixp, iyp, iz);
    int indypzp = index3d(ix, iyp, izp);
    int indxpzp = index3d(ixp, iy, izp);    
    
    double wx = px - ix * torus.dx;
    double wy = py - iy * torus.dy;
    double wz = pz - iz * torus.dz;

    // Interpolate between velocities at grid points

    *ux = fma_lerp(fma_lerp(torus.vx[ind0], torus.vx[indyp], wy),
          fma_lerp(torus.vx[indzp], torus.vx[indypzp], wy), wz);
    *uy = fma_lerp(fma_lerp(torus.vy[ind0], torus.vy[indzp], wz),
          fma_lerp(torus.vy[indxp], torus.vy[indxpzp], wz), wx);
    *uz = fma_lerp(fma_lerp(torus.vz[ind0], torus.vz[indxp], wx),
          fma_lerp(torus.vz[indyp], torus.vz[indxpyp], wx), wy);

}


__global__ void StaggeredAdvect_kernel()
// advect particle positions using RK4 in a grid torus with
// staggered velocity vx,vy,vz, for dt period of time
{
    int i = check_limit(particles.num_particles);
    if(i<0)return;
    double k1x, k1y, k1z;
    double k2x, k2y, k2z;
    double k3x, k3y, k3z;
    double k4x, k4y, k4z;
    double *x = particles.x;
    double *y = particles.y;
    double *z = particles.z;
    double dt = isf.dt;
        
    // Fourth-order Runge-Kutta method
        StaggeredVelocity(x[i], y[i], z[i], &k1x, &k1y, &k1z);
        StaggeredVelocity(x[i]+dt*k1x/2, y[i]+dt*k1y/2, z[i]+dt*k1z/2, 
            &k2x, &k2y, &k2z);
        StaggeredVelocity(x[i]+dt*k2x/2, y[i]+dt*k2y/2, z[i]+dt*k2z/2,
            &k3x, &k3y, &k3z);
        StaggeredVelocity(x[i]+dt*k3x, y[i]+dt*k3y, z[i]+dt*k3z, 
            &k4x, &k4y, &k4z);
        x[i] += dt/6*(k1x+2*k2x+2*k3x+k4x);
        y[i] += dt/6*(k1y+2*k2y+2*k3y+k4y);
        z[i] += dt/6*(k1z+2*k2z+2*k3z+k4z);
}

__global__ void StaggeredAdvect_fma_kernel()
// advect particle positions using RK4 in a grid torus with
// staggered velocity vx,vy,vz, for dt period of time
{
    int i = check_limit(particles.num_particles);
    if(i<0)return;
    double k1x, k1y, k1z;
    double k2x, k2y, k2z;
    double k3x, k3y, k3z;
    double k4x, k4y, k4z;
    double *x = particles.x;
    double *y = particles.y;
    double *z = particles.z;
    double dt = isf.dt;
        
    // Fourth-order Runge-Kutta method
        StaggeredVelocity_fma(x[i], y[i], z[i], &k1x, &k1y, &k1z);
        StaggeredVelocity_fma(x[i]+dt*k1x/2, y[i]+dt*k1y/2, z[i]+dt*k1z/2, 
            &k2x, &k2y, &k2z);
        StaggeredVelocity_fma(x[i]+dt*k2x/2, y[i]+dt*k2y/2, z[i]+dt*k2z/2,
            &k3x, &k3y, &k3z);
        StaggeredVelocity_fma(x[i]+dt*k3x, y[i]+dt*k3y, z[i]+dt*k3z, 
            &k4x, &k4y, &k4z);
        x[i] += dt/6*(k1x+2*k2x+2*k3x+k4x);
        y[i] += dt/6*(k1y+2*k2y+2*k3y+k4y);
        z[i] += dt/6*(k1z+2*k2z+2*k3z+k4z);

}


void StaggeredAdvect()
{
  tpstart(10);
  int nb = calc_numblock(particles_cpu.num_particles, THREADS_PER_BLOCK);
  StaggeredAdvect_fma_kernel<<<nb,THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize(); 
  tpend(10); 
}
