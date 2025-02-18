# PainlessMG
This code is developed base on Huaming Wang's code base
http://web.cse.ohio-state.edu/~wang.3602/Wang-2015-ACS/Chebyshev_sim.zip

# Usage
Codes in *CUDA_Projective_Armadillo* is for a tet mesh simulation.\
Codes in *CUDA_Projective_Square* is for a mass-spring cloth simulation.

Settings of simulation can be hard coded. However to get rid of recompilation, we provide a way to load setting from files.

# Format of the config file
## For cloth simulations:
### Overview
[*cloth size*]

[*output directory of benckmark records*]

[*enable benchmark*]

[*number of iterations in one frame*] [*enable Hessian*] [*enable Chebyshev*] [*enable pre-factorization on the finest level*]

[*number of coarsened level*]

[*number of vertices in each coarsened level, begin from the coarsest level*]

[*save matrix in each level (including the finest level) in a dense matrix memory layout*]

[*save matrix in each level (including the finest level) in a LDU memory layout*]

[*number of multigrid operations*]

[*list of multigrid operations*]

[*parameter for Chebyshev*]

[*number of objects in the scene*]

[*list of objects in the scene*]

### Detail Explaination
*cloth size*:\
number of vertices on one side of a square cloth

*enable Chebyshev*:\
Chebyshev is an algorithm delivered by Huaming Wang. Here is the [PDF](https://web.cse.ohio-state.edu/~wang.3602/Wang-2015-ACS/Wang-2015-ACS.pdf). This option is only for comparison.

*enable pre-factorization on the finest level*:\
Enable it for direct solve on the finest level. Disable it for saving a ton of time in pre-computing.

*save matrix in each level (including the finest level) in a LDU memory layout*:\
LDU memory layout saves the lower part, the diagonal and the upper part of a matrix seperately.\
For performance issue, symmetric Gauss-Seidel iteration is only supported on this memory layout.

*list of multigrid operations*:\
To customize a multigrid procedure, we offer **3** kinds of operations: **DownSample**, **UpSample** and **Smoothing**.
- **DownSample**: This operation sets up the coarsened problem based on current level and switches to the next coarsened level. Use it as "DS".
- **UpSample**: This operation updates information in the fined problem based on current level and switches to the next fined level. Use it as "US".
- **Smoothing**: This operation performs smoothing iterations on the current level. So far we have implemented:
   - Jacobi iteration: could be used if matrix is not stored in dense, use "Jacobi [N]" for [N] iterations.
   - Gauss-Seidel iteration: could be used if matrix is not store in dense and stored in LDU memory layout, use "GS [N]" for [N] iterations.
   - Direct solve: could be used if matrix is in the coarsest level and stored in dense, or in the finest level with [*enable pre-factorization on the finest level*] on, use it as "Direct".

*list of objects in the scene*:\
This option is for simulation with collision. Define the **COLLISION** flag in code use enable it.
So far we support these objects:
- **Sphere**: use "Sphere [cx] [cy] [cz] [r]" as a sphere centered in position (cx, cy, cz) with radius r.
- **Plane**: use "Plane [nx] [ny] [nz] [b]" as a plane whose analytical form is (nx, ny, nz)\*(x, y, z)=b.
- **Cylinder**: use "Cylinder [cx] [cy] [r]" as a Cylinder which is infinitely long along z axis, centered in position (cx, cy, 0) and with radius r.

*boundary conditions*:\
We have a hard coded "fixed vectices" boundary conditions.

## For tet simulations:
### Overview
[*mesh file name*]

[*output directory of benckmark records*]

[*enable benchmark*]

[*scaling*] [*pre rotation*] [*post rotation*]

[*number of iterations in one frame*] [*enable Hessian*] [*enable Chebyshev*] [*enable pre-factorization on the finest level*]

[*relaxation parameter of Gauss-Seidel iteration*]

[*number of coarsened level*]

[*number of vertices in each coarsened level, begin from the coarsest level*]

[*save matrix in each level (including the finest level) in a dense matrix memory layout*]

[*save matrix in each level (including the finest level) in a LDU memory layout*]

[*number of multigrid operations*]

[*list of multigrid operations*]

[*parameter for Chebyshev*]

[*number of objects in the scene*]

[*list of objects in the scene*]

### Detail Explaination
Almost the same as the previous one, except:

*mesh file name*:\
we provide some sample meshes in our project. You can generate mesh files in the same format to simulate your own meshes.

*scaling*, *pre rotation*, *post rotation*:\
This is for getting rid of re-compilation while benchmarking. You can also hard code it for customization.

# We are still working...
This code has not been re-written yet for people to read it easily. Contact Zangyueyang.Xian.GR@dartmouth.edu if you have any problem.



# CMake
This is tested on CUDA 12.5 and VS2022. Only for softbody for now.
**Build**
```
cmake -B build
cmake --build build --config=Release
```
**Run**
cd to CUDA_Projective_Armadillo and run soft.exe