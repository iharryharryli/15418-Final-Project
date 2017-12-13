# CUDA Implementation and Rendering of Schrodinger’s Smoke
by Yixiu Zhao (yixiuz) and Shangda Li (shangdal)

## Demo
https://www.youtube.com/watch?v=gY28SRlS48I

Here is the link to a video of our algorithm simulating and rendering Schrodinger’s Smoke with 5,000,000 particles in real-time at 48 FPS.

## Summary
We present our CUDA parallel implementation of a novel fluid simulation algorithm known as Incompressible Schrodinger Flow as well as a particle density map renderer that outputs high resolution images in real time.


## Background

![Alt text](1.png)

The paper “Schrodinger’s Smoke” (Chern et al.) describes a new approach for the purely Eulerian simulation of incompressible fluids. In it, the fluid state is represented by a  valued wave function evolving under the Schrödinger equation subject to incompressibility constraints. By solving the time evolution of Schrodinger’s equation, sometimes along with other constraints (like constant velocity constraint in case of simulating a jet), we obtain all the information of the evolution of the fluid. From the complex wave functions, the velocity field of the flow is computed. This is the Incompressible Schrodinger Flow (ISF) algorithm. This algorithm is not only elegant, but produces great results, as the comparison with real life smoke the original paper showed (Fig.1).

With the velocity field, we can visualize the fluid flow with particle advection. Particle advection is the technique for visualizing a vector field by having virtual particles follow the field vectors. To make relatively accurate trajectories in the discrete-time setting, the Fourth-order Runge-Kutta method is used, as is common in approximation problems like these. These advected particles can then be rendered each as transparent white dots with a constant alpha value. To sum up, there are three distinctive components to the whole algorithm: the ISF simulation itself, particle advection, and rendering (Fig. 2). The advantage of this approach is that it is a purely grid-based, Eulerian fluid simulator. We don't need to track the dynamics of the individual particles one by one, but we can still keep these fine, small-scale details in a way that rivals Lagrangian simulations, but without the huge additional costs.

The key data structures are three dimensional grids of numerical data of type double and cuDoubleComplex, where cuDoubleComplex is the CUDA implementation of complex numbers. Three main types of key operations are performed on the data in this algorithm. These operations are element-wise product, forward and inverse Fourier Transforms, and differentiation. Note that differentiation in the discrete grid setting is just taking the difference of value between adjacent grid points.

The ISF component of the algorithm takes in the size of the grid as well as initial state of the wave function and outputs the state of the wave function after a certain number of iterations. To simulate fluids under more complex constraints, additional parameters might be added to the simulation (like jet size and velocity in the case of a jet). The advection component takes in an array of particles with their three-dimensional positions, and outputs the new positions after a small amount of time. Finally, the rendering component takes in the positions of the particle, along with an output image size, and outputs an image based on the XY projections of the particles.

The ratio of computation time of the three components vary with key parameters like the resolution of the fluid grid solver and the number of particles. The computational cost of ISF increase with the number of voxels, while both particle advection and rendering time increase unsurprisingly with the number of particles. All three components are considerably parallel, while ISF is less parallel because of its multitude of computation steps which can each be done in a highly parallel way but depend on the results of the operations that come before it. ISF is also highly regular and localized in its array access patterns, unlike particle advection, which accesses the velocity array based on each particle’s position. Finally, the renderer is also highly parallel, similar to the circle renderer from assignment 2. However, there are a couple of differences. First, since the particles are points instead of circles, each particle only affects the color of one pixel (although it is an arbitrary design decision we can play around with), which reduces the complexity of the problem. On the other hand, the number of particles is typically several orders of magnitude higher than the number of circles (starting at 500k), which renders (excuse the pun) our original algorithm in the assignment intractable, since it had to allocate too much memory to keep track of the containment relationships between the particles and each pixel block. Last but not least, the nature of the problem demands consecutive rendering of images, preferably in real time, which means the initialization time of the renderer in between each frame should also be taken into consideration.


## Approaches and results
We start with the MATLAB code and Houdini scripts published by the original authors and rewrite the algorithm on the GPU platform using C++ and CUDA. In the implementation of the rendering part, we also use the Thrust library and the rendering framework from assignment 2. Note that we are only using the part of code that was given to us, and not reusing any of the kernels we wrote for the assignment. The core rendering algorithm is not included in the original paper and is entirely original.

### Part I. The main ISF algorithm
The main ISF algorithm is relatively easy to parallelize, as most computational steps are element wise, and we only need to map all the positions in the grid to the threads. We time all the steps of the computation and create a chart to identify the most time-consuming parts (Figure 3. left). Apart from the calls to Fourier Transforms in the cuFFT library, we can see that the functions VelocityOneForm_kernel (V1F) and PoissonSolveMain_kernel (Poisson) are most time consuming. Since VelocityOneForm_kernel also accesses adjacent elements of the current position, we assume that the reason for the high cost is memory related (which is a wrong assumption, as we will see later).

We make multiple attempts to parallelize the function using tricks learned on the CUDA website in the hope that the program can be more efficient in accessing global memory. The simple tricks like grid stride loops, and using the __restrict__ keyword yield no meaningful results, so we move on to more a more sophisticated method, which is using shared memory as a cache for memory access. Still, there is no improvement at all.

After all these fails, we realize that at the grid resolution of 128 * 64 * 64, the ISF component is compute bound, because trigonometric functions are very high in arithmetic intensity. Deleting all computations of trigonometric function in the VelocityOneForm_kernel decreased its run time by about 90%. There is a possibility that at higher grid resolutions the memory effects will become more apparent, however we observe that the cost of FFT dominates at higher grid resolutions. Therefore, we decide that optimizing for memory access and cache locality will not yield meaningful results.

Looking at the function PoissonSolveMain_kernel, we discover that it is just an element-wise product of a coefficient matrix and the divergence matrix. The construction of the coefficient matrix is highly complex, but its values do not change. Since we know the code is compute bound, we can save the values coefficient matrix for later use. This almost entirely eliminated the cost of the function and thus improved the runtime of ISF by 10% at our baseline resolution of 128 * 64 * 64 (Figure 3. right).

### Part II. Particle Advection
In this section we look at the relationship between the total cost of ISF and particle advection (not including rendering) and its relationship with the number of particles. As we can see in Figure 4, the cost of particle advection is roughly linearly proportional to the number of particles. At the order of magnitude of 10000000, the cost is entirely dominated by particle advection. Similar to ISF, particle advection is also compute bound, and the most costly operation in this part is the three-dimensional interpolation of velocity vectors. Using the Fused Multiply-Add (FMA) operation provided by CUDA, we are able to yield a small amount (about 7%) of speedup for particle advection (see Figure 5. below).

After this optimization, we decide to move on to the rendering part. There are two reasons for not focusing on particle advection. Firstly, since the adjacent particles stored in memory is not necessarily adjacent in grid space, array access is inherently random, which makes it hard to do further non-trivial optimizations. Secondly, because of the same reason, rendering is really hard, and the naive implementation of the rendering algorithm takes significantly more time that the advection itself. We therefore move on to the rendering part.

### Part III. Rendering








## Resources
PAAC paper: https://arxiv.org/pdf/1705.04862.pdf

Sourcecode: https://github.com/Alfredvc/paac

We will be using the processors in XuLab@CMU: https://sites.google.com/view/xulab/home


## Goals/Deliverables
The most important metrics of evaluation are training per second (TPS), prediction per second (PPS) and the overall score during training. There are two main criteria that we want to evaluate. Firstly we want the network to perform faster with the current number of workers (measured in TPS and PPS). Secondly we want to use more workers at roughly the same TPS and PPS and see if it leads to faster convergence (measured in score), because the experiences are drawn from more randomly distributed sources.

We hope to achieve at least a 1.5x speedup over the original paper in terms of training throughput, and we are happy with any improvement in terms of score vs. number of training data.

## Platform
We are using the python Tensorflow library for evaluation of deep neural networks within the algorithm, as it is the same platform used by the original paper.

## Schedule
Week 1: Read the paper and code from the original authors.

Week 2: Try to get the original code running on our machines.

Week 3: Try to get the original code running on our machines and do minor adjustments.

Week 4: Try to improve the original code using methods using pipelining.

Week 5: Try more optimization tricks if the methods don’t work. Start doing experiments.

Week 6: Work on the final writeup.

## References
[1] Alfredo V. Clemente, Humberto N. Castejón, Arjun Chandra. “Efficient Parallel Methods for Deep Reinforcement Learning”. arXiv:1705.04862 [cs.LG] (2017).
