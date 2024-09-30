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

namespace o2
{
namespace vertexing
{
namespace device
{

#if !defined(__HIPCC__) && !defined(__CUDACC__)
typedef struct _dummyStream {
} Stream;
#else
#ifdef __HIPCC__
typedef hipStream_t Stream;
#else
typedef cudaStream_t Stream;
#endif
#endif

class GPUInterface
{
 public:
  GPUInterface(GPUInterface& other) = delete;
  void operator=(const GPUInterface&) = delete;

  static GPUInterface* Instance();

  // APIs
  void registerBuffer(void*, size_t);
  void unregisterBuffer(void* addr);
  void allocDevice(void**, size_t);
  void freeDevice(void*);
  Stream& getStream(unsigned short N = 0);
  Stream& getNextStream();

 protected:
  GPUInterface(size_t N = 1);
  ~GPUInterface();

  void resize(size_t);

  std::atomic<unsigned short> mLastUsedStream{0};
  static GPUInterface* sGPUInterface;
  std::vector<std::thread> mPool{};
  std::vector<Stream> mStreams{};
};

} // namespace device
} // namespace vertexing
} // namespace o2
#endif
