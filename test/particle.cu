#include "depend.h"
#include "mycomplex.cu"

struct particles_t
{
  double *x;
  double *y;
  double *z;
};

__constant__ particles_t particles;
particles_t particles_cpu;

__global__ void StaggeredAdvect()
// advect particle positions using RK4 in a grid torus with
// staggered velocity vx,vy,vz, for dt period of time
{
    // [k1x,k1y,k1z] =...
    //     particle.StaggeredVelocity(...
    //         particle.x,particle.y,particle.z,...
    //         torus,vx,vy,vz);
    // [k2x,k2y,k2z] =...
    //     particle.StaggeredVelocity(...
    //         particle.x+k1x*dt/2,particle.y+k1y*dt/2,particle.z+k1z*dt/2,...
    //         torus,vx,vy,vz);
    // [k3x,k3y,k3z] =...
    //     particle.StaggeredVelocity(...
    //         particle.x+k2x*dt/2,particle.y+k2y*dt/2,particle.z+k2z*dt/2,...
    //         torus,vx,vy,vz);
    // [k4x,k4y,k4z] =...
    //     particle.StaggeredVelocity(...
    //         particle.x+k3x*dt,particle.y+k3y*dt,particle.z+k3z*dt,...
    //         torus,vx,vy,vz);
    // particle.x = particle.x + dt/6*(k1x+2*k2x+2*k3x+k4x);
    // particle.y = particle.y + dt/6*(k1y+2*k2y+2*k3y+k4y);
    // particle.z = particle.z + dt/6*(k1z+2*k2z+2*k3z+k4z);
}

__global__ void StaggeredVelocity()
// evaluates velocity at (px,py,pz) in the grid torus with staggered
// velocity vector field vx,vy,vz
{
    // px = mod(px,torus.sizex);
    // py = mod(py,torus.sizey);
    // pz = mod(pz,torus.sizez);
    
    // ix = floor(px/torus.dx) + 1;
    // iy = floor(py/torus.dy) + 1;
    // iz = floor(pz/torus.dz) + 1;
    // ixp = mod(ix,torus.resx)+1;
    // iyp = mod(iy,torus.resy)+1;
    // izp = mod(iz,torus.resz)+1;
    // ind0 = sub2ind([torus.resx,torus.resy,torus.resz],ix,iy,iz);
    // indxp = sub2ind([torus.resx,torus.resy,torus.resz],ixp,iy,iz);
    // indyp = sub2ind([torus.resx,torus.resy,torus.resz],ix,iyp,iz);
    // indzp = sub2ind([torus.resx,torus.resy,torus.resz],ix,iy,izp);
    // indxpyp = sub2ind([torus.resx,torus.resy,torus.resz],ixp,iyp,iz);
    // indypzp = sub2ind([torus.resx,torus.resy,torus.resz],ix,iyp,izp);
    // indxpzp = sub2ind([torus.resx,torus.resy,torus.resz],ixp,iy,izp);
    
    // wx = px - (ix-1)*torus.dx;
    // wy = py - (iy-1)*torus.dy;
    // wz = pz - (iz-1)*torus.dz;
    // ux = (1-wz).*((1-wy).*vx(ind0 )+wy.*vx(indyp  )) + ...
    //         wz .*((1-wy).*vx(indzp)+wy.*vx(indypzp));
    // uy = (1-wz).*((1-wx).*vy(ind0 )+wx.*vy(indxp  )) + ...
    //         wz .*((1-wx).*vy(indzp)+wx.*vy(indxpzp));
    // uz = (1-wy).*((1-wx).*vz(ind0 )+wx.*vz(indxp  )) + ...
    //         wy .*((1-wx).*vz(indyp)+wx.*vz(indxpyp));

}