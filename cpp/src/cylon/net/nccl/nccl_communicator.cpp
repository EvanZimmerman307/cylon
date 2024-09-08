/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <memory>

#include <arrow/ipc/api.h>

#include "cylon/net/communicator.hpp"
#include "cylon/net/nccl/nccl_communicator.hpp"
#include "cylon/net/nccl/nccl_channel.hpp"
#include "cylon/scalar.hpp"
#include "cylon/util/macros.hpp"

#include "cylon/net/mpi/mpi_communicator.hpp"
#include "cylon/net/mpi/mpi_channel.hpp"
#include "cylon/net/mpi/mpi_operations.hpp"
#include "cylon/scalar.hpp"
#include "cylon/util/macros.hpp"

namespace cylon {
namespace net {

// configs
CommType NCCLConfig::Type() {
  return CommType::NCCL;
}

std::shared_ptr<NCCLConfig> NCCLConfig::Make(ncclComm_t ncclcomm, MPI_Comm mpicomm) {
  return std::make_shared<NCCLConfig>(ncclcomm, mpicomm);
}

NCCLConfig::NCCLConfig(ncclComm_t ncclcomm, MPI_Comm mpicomm) : ncclcomm_(ncclcomm), mpicomm_(mpicomm) {}

ncclComm_t NCCLConfig::GetNCCLComm() const {
  return ncclcomm_;
}

MPI_Comm NCCLConfig::GetMPIComm() const {
  return mpicomm_;
}
NCCLConfig::~NCCLConfig() = default;

std::unique_ptr<Channel> NCCLCommunicator::CreateChannel() const {
  return std::make_unique<NCCLChannel>(nccl_comm_);
}

int NCCLCommunicator::GetRank() const {
  return this->rank;
}
int NCCLCommunicator::GetWorldSize() const {
  return this->world_size;
}

Status NCCLCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                             MemoryPool *pool,
                             std::shared_ptr<Communicator> *mpi_out, std::shared_ptr<Communicator> *nccl_out) {
  int myRank, nRanks, ext_init;
  // check if MPI is initialized
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Initialized(&ext_init));
  auto nccl_comm = std::static_pointer_cast<NCCLConfig>(config)->GetNCCLComm();
  auto mpi_comm = std::static_pointer_cast<NCCLConfig>(config)->GetMPIComm();
  
  if (mpi_comm != MPI_COMM_NULL && !ext_init) {
    return {Code::Invalid, "non-null MPI_Comm passed without initializing MPI"};
  }

  if (!ext_init) { // if not initialized, init MPI
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }

  if (mpi_comm == MPI_COMM_NULL) { // set comm_ to world
    mpi_comm = MPI_COMM_WORLD;
  }

  // setting errors to return
  MPI_Comm_set_errhandler(mpi_comm, MPI_ERRORS_RETURN);

  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_rank(mpi_comm, &myRank));
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Comm_size(mpi_comm, &nRanks));

  ncclUniqueId id;
  if (myRank == 0) ncclGetUniqueId(&id);
  // Next, a single rank will create a unique ID and send it to all other ranks to make sure everyone has it:
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (myRank < 0 || nRanks < 0 || myRank >= nRanks) {
      return {Code::ExecutionError, "Malformed rank: " + std::to_string(myRank)
          + " or world size: " + std::to_string(nRanks)};
  }

  // Initialize NCCL
  RETURN_CYLON_STATUS_IF_NCCL_FAILED(ncclCommInitRank(&nccl_comm, nRanks, id, myRank));

  *mpi_out = std::make_shared<MPICommunicator>(pool, myRank, nRanks, mpi_comm, (bool) ext_init);
  *nccl_out = std::make_shared<NCCLCommunicator>(pool, myRank, nRanks, nccl_comm, mpi_out, (bool) ext_init);
  return Status::OK();

}

NCCLCommunicator::NCCLCommunicator(MemoryPool *pool,
                                 int32_t rank,
                                 int32_t world_size,
                                 ncclComm_t nccl_comm,
                                 MPICommunicator* mpi_communicator
                                 bool externally_init)
    : Communicator(pool, rank, world_size), nccl_comm_(nccl_comm), externally_init(externally_init), mpi_comm_(mpi_communicator) {}

void NCCLCommunicator::Finalize() {
  // Finalize only if we initialized NCCL
  if (!externally_init && !IsFinalized()) {
    if (nccl_comm_ != nullptr) {
      ncclResult_t result = ncclCommDestroy(nccl_comm_);
      if (result != ncclSuccess) {
        // Handle error (you might want to log this or throw an exception)
        std::cerr << "NCCL finalization failed: " << ncclGetErrorString(result) << std::endl;
      }
      nccl_comm_ = nullptr;
    }
  mpi_comm_->finalize();
  finalized = true;  
  }
}

bool NCCLCommunicator::IsFinalized() const {
  return finalized_;
}

CommType NCCLCommunicator::GetCommType() const {
  return NCCL;
}

MPI_Comm NCCLCommunicator::mpi_comm() const {
  return mpi_comm_;
}

ncclComm_t NCCLCommunicator::nccl_comm() const {
  return nccl_comm_;
}

}
}