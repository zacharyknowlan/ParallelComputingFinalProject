set(SOURCES
    MicromorphicIntegrator.cpp 
    MicromorphicStressStrain.cpp 
    GlobalStaticCondensation.cpp
    MicromorphicPartialAssembly.cpp)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
add_library(MicromorphicLibrary ${SOURCES})
target_include_directories(MicromorphicLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(MicromorphicLibrary PUBLIC mfem)
