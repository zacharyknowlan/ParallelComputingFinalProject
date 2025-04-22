module use /opt/scorec/spack/rhel9/v0201_4/lmod/linux-rhel9-x86_64/Core/
module load gcc/12.3.0-iil3lno 
module load mpich/4.1.1-xpoyz4t
module load cuda/12.1.1-zxa4msk
module load openblas/0.3.23-wqm7iud
module load cmake/3.26.3-2duxfcd

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CXX_COMPILER=`which mpic++` \
    -DCMAKE_CUDA_COMPILER=`which nvcc` \
    -DMFEM_ROOT=mfem-4.7/build/install/
cmake --build build/ -j 8
