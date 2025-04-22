if grep -q '^MFEM_USA_CUDA:BOOL=ON' mfem-4.7/build/CMakeCache.txt; then
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CXX_COMPILER=`which g++` \
        -DCMAKE_CUDA_COMPILER=`which nvcc` \
        -DMFEM_ROOT=mfem-4.7/build/install/ 
else
    module load mpich/4.1.1-xpoyz4t # SCOREC module
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CXX_COMPILER=`which mpic++` \
        -DCMAKE_CUDA_COMPILER=`which nvcc` \
        -DMFEM_ROOT=mfem-4.7/build/install/ 
fi
cmake --build build/ -j 8
