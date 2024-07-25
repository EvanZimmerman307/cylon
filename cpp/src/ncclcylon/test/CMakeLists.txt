cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(create_cudf_table_test)
rapids_cuda_set_architectures(RAPIDS)

project(
  create_cudf_table_test
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(create_cudf_table_test src/create_cudf_table_test.cpp)
target_link_libraries(create_cudf_table_test PRIVATE cudf::cudf)
target_compile_features(create_cudf_table_test PRIVATE cxx_std_17)

install(TARGETS create_cudf_table_test DESTINATION bin/examples/libcudf)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/create_cudf_table_test.csv DESTINATION bin/examples/libcudf)