// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \brief Helper interface to the GPU device, meant to be compatible with manual allocation/streams and GPUReconstruction ones.
/// \author matteo.concas@cern.ch

#ifndef DCAFITTER_GPU_INTERFACE
#define DCAFITTER_GPU_INTERFACE

#include <thread>
#include <vector>
#include <atomic>

namespace o2
{
namespace vertexing
{
namespace device
{

#if !defined(__HIPCC__) && !defined(__CUDACC__)
typedef struct _dummyStream {
} stream;
#else
#ifdef __HIPCC__
typedef hipStream_t stream;
#else
typedef cudaStream_t stream;
#endif
#endif

class GPUInterface
{
 public:
  GPUInterface(GPUInterface& other) = delete;
  void operator=(const GPUInterface&) = delete;

  static GPUInterface* Instance();

  // APIs
  void register(void*, size_t);
  void allocAsync(void**, size_t, unsigned short streamId = -1);

 protected:
  GPUInterface(size_t N)
  {
    resize(N);
  }

  void resize(size_t);
  unsigned short getNextCursor();

  static GPUInterface* sGPUInterface = nullptr;
  std::atomic<unsigned short> mCursor{0};
  std::vector<std::thread> mPool{};
  std::vector<stream> mStreams{};
};

inline void GPUInterface::resize(size_t N)
{
  mPool.resize(N);
  mStreams.resize(N);
}

inline unsigned short GPUInterface::getNextCursor()
{
  auto index = mCursor++;

  auto id = index % mPool.size();

  auto oldValue = mCursor;
  auto newValue = oldValue % mPool.size();
  while (!mCursor.compare_exchange_weak(oldValue, newValue, std::memory_order_relaxed)) {
    newValue = oldValue % mPool.size();
  }
  return id;
}
} // namespace device
} // namespace vertexing
} // namespace o2
#endif
