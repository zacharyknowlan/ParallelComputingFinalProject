# Load the appropriate modules (the following are for the SCOREC system)
module use /opt/scorec/spack/rhel9/v0201_4/lmod/linux-rhel9-x86_64/Core/
module load gcc/12.3.0-iil3lno 
module load cuda/12.1.1-zxa4msk
module load cmake/3.26.3-2duxfcd

# Unzip and build MFEM (CUDA architecture is set for GeForce RTX 4060)
mkdir mfem-4.7/
#tar -xzvf mfem-4.7.tgz mfem-4.7/
cmake -S mfem-4.7/ -B mfem-4.7/build/ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=mfem-4.7/build/install/ \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DCMAKE_CUDA_COMPILER=`which nvcc` \
    -DMFEM_ENABLE_TESTING=OFF \
    -DMFEM_USE_MPI=OFF \
    -DMFEM_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build mfem-4.7/build/ -j 8 --target install

# Test MFEM
if grep -q '^MFEM_ENABLE_TESTING:BOOL=ON' mfem-4.7/build/CMakeCache.txt; then
    cd mfem-4.7/build/tests/ && make -j 8
    cd ../examples/ && make -j 8
    cd ../miniapps/ && make -j 8
    cd ../ && make test
    cd ../../
fi

# Configure the build
source ConfigBuild.sh
