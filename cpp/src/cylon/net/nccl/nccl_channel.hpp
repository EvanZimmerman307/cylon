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

#ifndef CYLON_CPP_SRC_CYLON_NET_NCCL_NCCL_CHANNEL_HPP_
#define CYLON_CPP_SRC_CYLON_NET_NCCL_NCCL_CHANNEL_HPP_

#include <unordered_map>
#include <queue>
#include <vector>
#include <memory

#include <nccl.h>
#include <cuda_runtime.h>
#include "cylon/net/channel.hpp"

namespace cylon {
namespace net {

enum SendStatus {
  SEND_INIT = 0,
  SEND_LENGTH_POSTED = 1,
  SEND_POSTED = 2,
  SEND_FINISH = 3,
  SEND_DONE = 4
};

enum ReceiveStatus {
  RECEIVE_INIT = 0,
  RECEIVE_LENGTH_POSTED = 1,
  RECEIVE_POSTED = 2,
  RECEIVED_FIN = 3
};

/**
 * Keep track about the length buffer to receive the length first
 */
struct PendingSend {
  //  we allow upto 8 ints for the header
  int header_buf[CYLON_CHANNEL_HEADER_SIZE]{};
  std::queue<std::shared_ptr<CylonRequest>> pending_data;
  SendStatus status = SEND_INIT;
  void* d_headerBuf; // d means device
  void* d_dataBuf;
  size_t dataSize;
  // the current send, if it is a actual send
  std::shared_ptr<CylonRequest> current_send{};

  PendingSend() {
    cudaMalloc(&d_headerBuf, CYLON_CHANNEL_HEADER_SIZE * sizeof(int)); // allocate header buffer on device
  }

  ~PendingSend() {
    cudaFree(d_headerBuf);
    cudaFree(d_dataBuf);
  }
};

struct PendingReceive {
  // we allow upto 8 integer header
  int header_buf[CYLON_CHANNEL_HEADER_SIZE]{};
  int recv_id{};
  std::shared_ptr<Buffer> data{};
  int length{};
  ReceiveStatus status = RECEIVE_INIT;
  void* d_headerBuf;
  void* d_dataBuf;

  PendingReceive() {
    cudaMalloc(&d_headerBuf, CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
  }

  ~PendingReceive() {
    cudaFree(d_headerBuf);
    cudaFree(d_dataBuf);
  }
};

class NcclChannel : public Channel {
 public:
  explicit NcclChannel(ncclComm_t comm, int rank, int world_size);  // do we need world size as well?

  void init(int edge, const std::vector<int> &receives, const std::vector<int> &sendIds,
            ChannelReceiveCallback *rcv, ChannelSendCallback *send, Allocator *alloc) override;

  /**
  * Send the message to the target.
  *
  * @param request the request
  * @return true if accepted
  */

  int send(std::shared_ptr<CylonRequest> request) override;

  /**
  * Send the message to the target.
  *
  * @param request the request
  * @return true if accepted
  */
  int sendFin(std::shared_ptr<CylonRequest> request) override;

  /**
   * This method, will send the messages, It will first send a message with length and then
   */
  void progressSends() override;

  /**
   * Progress the pending receivers
   */
  void progressReceives() override;

  void close() override;

  ~NCCLChannel() override = default;  // MPI has a destructor

 private:
  int edge;
  // keep track of the length buffers for each receiver
  std::unordered_map<int, PendingSend *> sends;
  // keep track of the posted receives
  std::unordered_map<int, PendingReceive *> pendingReceives;
  // we got finish requests
  std::unordered_map<int, std::shared_ptr<CylonRequest>> finishRequests;
  // receive callback function
  ChannelReceiveCallback *rcv_fn = nullptr;
  // send complete callback function
  ChannelSendCallback *send_comp_fn = nullptr;
  // allocator
  Allocator *allocator = nullptr;
  // GPU rank
  int rank;
  int world_size;
  ncclComm_t comm_;
  cudaStream_t stream_;

  /**
   * Send finish request
   * @param x the target, pendingSend pair
   */
  void sendFinishHeader(const std::pair<const int, PendingSend *> &x) const;

  /**
   * Send the length
   * @param x the target, pendingSend pair
   */
  void sendHeader(const std::pair<const int, PendingSend *> &x) const;
};
}


#endif //CYLON_CPP_SRC_CYLON_NET_GLOO_GLOO_CHANNEL_HPP_