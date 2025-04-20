module use /opt/scorec/spack/rhel9/v0201_4/lmod/linux-rhel9-x86_64/Core/
module load gcc/12.3.0-iil3lno 
#module load mpich/4.1.1-xpoyz4t
module load cuda/12.1.1-zxa4msk
module load openblas/0.3.23-wqm7iud
module load cmake/3.26.3-2duxfcd

# Unzip and build MFEM (MAKE SURE GPU ARCHITECTURE IS CORRECT)
mkdir mfem-4.7/
#tar -xzvf mfem-4.7.tgz mfem-4.7/
cmake -S mfem-4.7/ -B mfem-4.7/build/ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=mfem-4.7/build/install/ \
    -DCMAKE_C_COMPILER=`which gcc` \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DCMAKE_CUDA_COMPILER=`which nvcc` \
    -DMFEM_ENABLE_TESTING=ON \
    -DMFEM_USE_MPI=OFF \
    -DMFEM_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89
    #-DMETIS_DIR=metis-5.1.0/build/install/ \
    #-DHYPRE_DIR=hypre-2.33.0/build/install/ 
cmake --build mfem-4.7/build/ -j 8 --target install
cd mfem-4.7/build/ && make -j 8
#cd tests/ && make -j 8
#cd ../examples/ && make -j 8
#cd ../miniapps/ && make -j 8
#cd ../ && make test
cd ../../ && source ConfigMicromorphic.sh
