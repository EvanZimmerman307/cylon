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

#ifndef CYLON_SRC_CYLON_COMM_NCCLCOMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_NCCLCOMMUNICATOR_H_

#include <nccl.h>
#include <mpi.h>

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>

namespace cylon {
namespace net {

class NCCLConfig : public CommConfig {
 public:
  explicit NCCLConfig(ncclComm_t ncclcomm = nullptr, MPI_Comm mpicomm = MPI_COMM_NULL);

  CommType Type() override;

  ~NCCLConfig() override;

  ncclComm_t GetNCCLComm() const;

  MPI_Comm GetMPIComm() const;

  static std::shared_ptr<NCCLConfig> Make(ncclComm_t comm = nullptr, MPI_Comm mpicomm = MPI_COMM_NULL);

 private:
  ncclComm_t ncclcomm_;
  MPI_Comm mpicomm_;
};


class NCCLCommunicator : public Communicator {
 public:
  NCCLCommunicator(MemoryPool *pool, int32_t rank, int32_t world_size, ncclComm_t nccl_comm , MPI_Comm mpi_comm, bool externally_init);
  ~MPICommunicator() override = default;
  std::unique_ptr<Channel> CreateChannel() const override;
  int GetRank() const override;
  int GetWorldSize() const override;
  void Finalize() override;
  void Barrier() override;
  CommType GetCommType() const override;

  MPI_Comm mpi_comm() const;

  ncclComm_t nccl_comm() const;

  static Status Make(const std::shared_ptr<CommConfig> &config,
                     MemoryPool *pool, std::shared_ptr<Communicator> *mpi_out, std::shared_ptr<Communicator> *nccl_out);

  private:
    MPI_Comm mpi_comm_ = MPI_COMM_NULL;
    ncclComm_t nccl_comm_ = nullptr;
    bool externally_init = false;

};

}
}
#endif //CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_


