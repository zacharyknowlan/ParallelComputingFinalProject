cmake_minimum_required(VERSION 3.20)
project(MicromorphicProject CXX CUDA)
find_package(MFEM REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_FLAGS "--expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")

add_subdirectory(MicromorphicLibrary)

if(MFEM_USE_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(main main.cpp)
    target_link_libraries(main PUBLIC mfem MicromorphicLibrary)
elseif(NOT MFEM_USE_CUDA)
    add_executable(ParallelVTKWriter ParallelVTKWriter.cpp)
    target_link_libraries(ParallelVTKWriter PUBLIC mfem MicromorphicLibrary)
endif()