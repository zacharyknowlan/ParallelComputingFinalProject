# Building The Project
## MFEM 
* If on a SCOREC machine, which is where the results for this project were obtained, Run `source MFEM_Config.sh`. This will build the entire project with the default configuration.
* On a non-SCOREC system, you will need to edit the `-DCMAKE_CUDA_ARCHITECTURE` flag and substitute the list of loaded modules for your systems equivalent.
* To run all parts of this project you will have to build MFEM twice. Once with `-DMFEM_USE_CUDA=ON` to run `ParallelPartialAssembly` and once with `-DMFEM_USE_CUDA=OFF` to run `ParallelVTKWriter`. See Project Source Code for more information.
* Due to the way MFEM builds with CUDA (sets all files to be compiled with nvcc), the micromrophic problem cannot be solved on the GPU and subsequently written using MPI I/O, so the file `ParallelVTKWriter.cpp` writes a dummy problem and uses the MFEM library compiled serially.
* The default has been set to build MFEM with CUDA.
* **NOTE:** The shell script builds and runs all MFEM tests when `-DMFEM_ENABLE_TESTING=ON`.
If any tests return a failure the results of your build may not be consistent with the results reported in this project. 
The default is `-DMFEM_ENABLE_TESTING=OFF`.
## Project Source Code
* `mpich` is loaded in `ConfigBuild.sh`. 
If you are attempting to run this on a non-SCOREC machine, this module will have to be changed to the MPI module on your system. 
With that being said, it has only been tested for `mpich` on the SCOREC machines.
* If `-DMFEM_USE_CUDA=ON`, CMake will compile and link `ParallelPartialAssembly.cpp` which uses CUDA to solve the micromorphic finite element problem.
* If `-DMFEM_USE_CUDA=OFF`, CMake will compile and link `ParallelVTKWriter.cpp` which uses MPI I/O to write a VTK file for the micromorphic finite element problem.
* A sample mesh with 10000 elements (342000 DOF for the finite element spaces used) has been provided.
* Both `ParallelPartialAssembly` and `ParallelVTKWriter` use the **relative** path of the mesh file. 
These executables must be launched from the `build/` directory to be able to read the mesh.
