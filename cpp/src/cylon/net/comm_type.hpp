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


#ifndef CYLON_SRC_CYLON_NET_COMM_TYPE_HPP_
#define CYLON_SRC_CYLON_NET_COMM_TYPE_HPP_

namespace cylon {
namespace net {
enum CommType {
  LOCAL = 0,
  MPI = 1,
  TCP = 2,
  UCX = 3,
  GLOO = 4,
  UCC = 5,
  NCCL = 6
};
}  // namespace net
}  // namespace cylon
#endif //CYLON_SRC_CYLON_NET_COMM_TYPE_HPP_
