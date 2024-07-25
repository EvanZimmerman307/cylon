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

#include <glog/logging.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <memory>
#include <utility>

#include <cylon/status.hpp>
#include <cylon/net/nccl/nccl_channel.hpp>
#include <cylon/net/cylon_request.hpp>

namespace cylon {

NcclChannel::NcclChannel(ncclComm_t comm, int world_size, int rank)
  : comm_(comm), world_size_(world_size), rank_(rank) {
  // Initialize CUDA stream
  cudaStreamCreate(&stream_);
}

void NcclChannel::init(int ed, const std::vector<int> &receives, const std::vector<int> &sendIds,
                      ChannelReceiveCallback *rcv, ChannelSendCallback *send_fn,
                      Allocator *alloc) {   
  edge = ed;
  rcv_fn = rcv;
  send_comp_fn = send_fn;
  allocator = alloc;

  //  could create stream here but we create it in the constructor
  
  // we need to post the length buffers
  for (int source : receives) {
    auto *buf = new PendingReceive();
    buf->receiveId = source;
    pendingReceives.insert(std::make_pair(source, buf));
    cudaError_t cuda_error = cudaMalloc(&buf->d_headerBuf, CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
    if (cuda_error != cudaSuccess) {
      LOG(FATAL) << "Failed to allocate device memory for header buffer for receives: " << cudaGetErrorString(cuda_error);
    }
    buf->status = RECEIVE_LENGTH_POSTED;
  }

  for (int target : sendIds) {
    sends[target] = new PendingSend();
    cudaError_t cuda_error = cudaMalloc(&sends[target]->d_headerBuf, CYLON_CHANNEL_HEADER_SIZE * sizeof(int));
    if (cuda_error != cudaSuccess) {
      LOG(FATAL) << "Failed to allocate device memory for header buffer for sends: " << cudaGetErrorString(cuda_error);
    }
  }
}

int NCCLChannel::send(std::shared_ptr<CylonRequest> request) {
  PendingSend *ps = sends[request->target];
  if (ps->pendingData.size() > MAX_PENDING) {
    return -1;
  }
  ps->pendingData.push(std::move(request));
  return 1;
}

int NCCLChannel::sendFin(std::shared_ptr<CylonRequest> request) {
  if (finishRequests.find(request->target) != finishRequests.end()) {
    return -1;
  }

  finishRequests.emplace(request->target, std::move(request));
  return 1;
}









