cmake_minimum_required(VERSION 3.10)
project(CylonTest)

find_package(MPI REQUIRED)
find_package(glog REQUIRED)
set(GLOG_LIBRARIES glog::glog)

add_library(mpi_channel /u/ewz9kg/cylon_nccl_dev/cpp/src/cylon/net/mpi/mpi_channel.cpp)

add_executable(mpi_channel_test src/mpi_channel_test.cpp)
target_link_libraries(mpi_channel_test cylon_context mpi_channel ${MPI_LIBRARIES})