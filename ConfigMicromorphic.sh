cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DCMAKE_CUDA_COMPILER=`which nvcc` \
    -DMFEM_ROOT=mfem-4.7/build/install/
cmake --build build/ -j 8
