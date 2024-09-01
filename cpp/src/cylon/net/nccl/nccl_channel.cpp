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
#include <queue>
#include <unordered_map>

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

void NCCLChannel::progressSends() {
  for (auto& x : sends) {
    PendingSend* ps = x.second;
    switch (ps->status) {
      case SEND_INIT:
        if (!ps->pendingData.empty()) {
          sendHeader(x);
        } else if (finishRequests.find(x.first) != finishRequests.end()) {
          sendFinishHeader(x);
        }
        break;

      case SEND_LENGTH_POSTED:
        // Check if the header send is complete
        if (cudaStreamQuery(stream_) == cudaSuccess) {
          // Post the actual data send
          auto& r = ps->pendingData.front();
          ncclSend(r->buffer, r->length, ncclChar, r->target, comm_, stream_);
          ps->status = SEND_POSTED;
          ps->currentSend = r;
          ps->pendingData.pop();
        }
        break;

      case SEND_POSTED:
        // Check if the send is complete
        if (cudaStreamQuery(stream_) == cudaSuccess) {
          if (!ps->pendingData.empty()) {
            sendHeader(x);
            send_comp_fn->sendComplete(ps->currentSend);
            ps->currentSend = {};
          } else {
            send_comp_fn->sendComplete(ps->currentSend);
            ps->currentSend = {};
            
            if (finishRequests.find(x.first) != finishRequests.end()) {
              sendFinishHeader(x);
            } else {
              ps->status = SEND_INIT;
            }
          }
        }
        break;

      case SEND_FINISH:
        // Check if the finish header send is complete
        if (cudaStreamQuery(stream_) == cudaSuccess) {
          send_comp_fn->sendFinishComplete(finishRequests[x.first]);
          ps->status = SEND_DONE;
        }
        break;

      case SEND_DONE:
        // Do nothing, we're done with this send
        break;

      default:
        LOG(FATAL) << "At an unexpected state " << ps->status;
        break;
    }
  }
}

void NCCLChannel::progressReceives() {
  for (auto& x : pendingReceives) {
    PendingReceive* pr = x.second;
    switch (pr->status) {
      case RECEIVE_LENGTH_POSTED:
        // NCCL doesn't have non-blocking receives, so we'll use cudaMemcpyAsync
        cudaMemcpyAsync(pr->headerBuf, pr->d_headerBuf, 
                        CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                        cudaMemcpyDeviceToHost, stream_);
        
        // Check if the transfer is complete
        if (cudaStreamQuery(stream_) == cudaSuccess) {
          int length = pr->headerBuf[0];
          int finFlag = pr->headerBuf[1];
          
          if (finFlag != CYLON_MSG_FIN) {
            // Allocate buffer and post receive
            Status stat = allocator->Allocate(length, &pr->data);
            if (!stat.is_ok()) {
              LOG(FATAL) << "Failed to allocate buffer with length " << length;
            }
            pr->length = length;
            ncclRecv(pr->data->GetByteBuffer(), length, ncclChar, 
                     pr->receiveId, comm_, stream_);
            pr->status = RECEIVE_POSTED;
            
            // Notify the receiver about the header
            int* header = nullptr;
            if (CYLON_CHANNEL_HEADER_SIZE > 2) {
              header = new int[CYLON_CHANNEL_HEADER_SIZE - 2];
              std::memcpy(header, &(pr->headerBuf[2]), (CYLON_CHANNEL_HEADER_SIZE - 2) * sizeof(int));
            }
            rcv_fn->receivedHeader(x.first, finFlag, header, CYLON_CHANNEL_HEADER_SIZE - 2);
          } else {
            pr->status = RECEIVED_FIN;
            rcv_fn->receivedHeader(x.first, finFlag, nullptr, 0);
          }
        }
        break;

      case RECEIVE_POSTED:
        // Check if the receive is complete
        if (cudaStreamQuery(stream_) == cudaSuccess) {
          rcv_fn->receivedData(x.first, pr->data, pr->length);
          pr->status = RECEIVE_LENGTH_POSTED;
          
          // Post the next header receive
          ncclRecv(pr->d_headerBuf, CYLON_CHANNEL_HEADER_SIZE, ncclInt, 
                   pr->receiveId, comm_, stream_);
        }
        break;

      case RECEIVED_FIN:
        // Do nothing, we're done with this receive
        break;

      default:
        LOG(FATAL) << "At an unexpected state " << pr->status;
        break;
    }
  }
}

void sendHeader(const std::pair<const int, PendingSend*>& x) {
  const auto& r = x.second->pendingData.front();
  x.second->headerBuf[0] = r->length;
  x.second->headerBuf[1] = 0;
  if (r->headerLength > 0) {
    memcpy(&(x.second->headerBuf[2]), &(r->header[0]), r->headerLength * sizeof(int));
  }
  cudaMemcpyAsync(x.second->d_headerBuf, x.second->headerBuf,
                  (2 + r->headerLength) * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);
  ncclSend(x.second->d_headerBuf, 2 + r->headerLength, ncclInt,
            x.first, comm_, stream_);
  x.second->status = SEND_LENGTH_POSTED;
}

void sendFinishHeader(const std::pair<const int, PendingSend*>& x) {
    x.second->headerBuf[0] = 0;
    x.second->headerBuf[1] = CYLON_MSG_FIN;
    cudaMemcpyAsync(x.second->d_headerBuf, x.second->headerBuf,
                    2 * sizeof(int), cudaMemcpyHostToDevice, stream_);
    ncclSend(x.second->d_headerBuf, 2, ncclInt, x.first, comm_, stream_);
    x.second->status = SEND_FINISH;
  }
}

void NCCLChannel::close() {
  for (auto &pendingReceive : pendingReceives) {
    //MPI_Cancel(&pendingReceive.second->request); // what is the nccl equivalent of mpi cancel
    std::cout << "Cancelling pendingReceive - No nccl version of cancel" << std:endl;
    delete (pendingReceive.second);
  }
  pendingReceives.clear();

  for (auto &s : sends) {
    // MPI_Cancel(&s.second->request);
    std::cout << "Cancelling sends - No nccl version of cancel" << std:endl;
    delete (s.second);
  }
  sends.clear();
}
}  // namespace cylon







