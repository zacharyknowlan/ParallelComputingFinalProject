add_library(VTKWriterLibrary VTKWriter.cpp)
target_include_directories(VTKWriterLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(VTKWriterLibrary PUBLIC mfem)
