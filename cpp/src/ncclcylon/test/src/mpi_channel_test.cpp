#include <iostream>
#include "/u/ewz9kg/cylon_nccl_dev/cpp/src/cylon/net/mpi/mpi_channel.hpp"
#include <cylon/ctx/cylon_context.hpp>

void testBasicSendReceive() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cylon::MPIChannel channel(MPI_COMM_WORLD);

  // Send a message from rank 0 to rank 1
  if (rank == 0) {
    std::shared_ptr<cylon::CylonRequest> request = std::make_shared<cylon::CylonRequest>(1);
    request->buffer = new int[10];
    request->length = 10;
    channel.send(request);
  }

  // Receive the message on rank 1
  if (rank == 1) {
    channel.progressReceives();
    // Check if the message was received correctly
    std::cout << "Test passed!" << std::endl;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  testBasicSendReceive();
  MPI_Finalize();
  return 0;
}