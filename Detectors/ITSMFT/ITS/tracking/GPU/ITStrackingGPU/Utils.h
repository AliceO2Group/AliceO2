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
///
/// \file Utils.h
/// \brief
///

#ifndef ITSTRACKINGGPU_UTILS_H_
#define ITSTRACKINGGPU_UTILS_H_

#include <vector>
#include <string>
#include <tuple>

#include "ITStracking/MathUtils.h"
#include "ITStracking/ExternalAllocator.h"

#include "GPUCommonDef.h"
#include "GPUCommonHelpers.h"
#include "GPUCommonLogger.h"
#include "GPUCommonDefAPI.h"

#ifdef GPUCA_GPUCODE
#include <thrust/device_ptr.h>
#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif
#endif

#ifdef ITS_GPU_LOG
#define GPULog(...)                      \
  do {                                   \
    LOGP(info, __VA_ARGS__);             \
    GPUChkErrS(cudaDeviceSynchronize()); \
  } while (0)
#else
#define GPULog(...)
#endif

namespace o2::its
{
// FWD declarations
template <int>
class IndexTableUtils;
class Tracklet;

template <typename T1, typename T2>
using gpuPair = std::pair<T1, T2>;

namespace gpu
{

template <typename T>
struct gpuSpan {
  using value_type = T;
  using ptr = T*;
  using ref = T&;

  GPUd() gpuSpan() : _data(nullptr), _size(0) {}
  GPUd() gpuSpan(ptr data, unsigned int dim) : _data(data), _size(dim) {}
  GPUd() ref operator[](unsigned int idx) const { return _data[idx]; }
  GPUd() unsigned int size() const { return _size; }
  GPUd() bool empty() const { return _size == 0; }
  GPUd() ref front() const { return _data[0]; }
  GPUd() ref back() const { return _data[_size - 1]; }
  GPUd() ptr begin() const { return _data; }
  GPUd() ptr end() const { return _data + _size; }

 protected:
  ptr _data;
  unsigned int _size;
};

template <typename T>
struct gpuSpan<const T> {
  using value_type = T;
  using ptr = const T*;
  using ref = const T&;

  GPUd() gpuSpan() : _data(nullptr), _size(0) {}
  GPUd() gpuSpan(ptr data, unsigned int dim) : _data(data), _size(dim) {}
  GPUd() gpuSpan(const gpuSpan<T>& other) : _data(other._data), _size(other._size) {}
  GPUd() ref operator[](unsigned int idx) const { return _data[idx]; }
  GPUd() unsigned int size() const { return _size; }
  GPUd() bool empty() const { return _size == 0; }
  GPUd() ref front() const { return _data[0]; }
  GPUd() ref back() const { return _data[_size - 1]; }
  GPUd() ptr begin() const { return _data; }
  GPUd() ptr end() const { return _data + _size; }

 protected:
  ptr _data;
  unsigned int _size;
};

// Abstract stream class
class Stream
{
 public:
#if defined(__HIPCC__)
  using Handle = hipStream_t;
  static constexpr Handle DefaultStream = 0;
  static constexpr unsigned int DefaultFlag = hipStreamNonBlocking;
  using Event = hipEvent_t;
#elif defined(__CUDACC__)
  using Handle = cudaStream_t;
  static constexpr Handle DefaultStream = 0;
  static constexpr unsigned int DefaultFlag = cudaStreamNonBlocking;
  using Event = cudaEvent_t;
#else
  using Handle = void*;
  static constexpr Handle DefaultStream = nullptr;
  static constexpr unsigned int DefaultFlag = 0;
  using Event = void*;
#endif

  Stream(unsigned int flags = DefaultFlag)
  {
#if defined(__HIPCC__)
    GPUChkErrS(hipStreamCreateWithFlags(&mHandle, flags));
    GPUChkErrS(hipEventCreateWithFlags(&mEvent, hipEventDisableTiming));
#elif defined(__CUDACC__)
    GPUChkErrS(cudaStreamCreateWithFlags(&mHandle, flags));
    GPUChkErrS(cudaEventCreateWithFlags(&mEvent, cudaEventDisableTiming));
#endif
  }

  Stream(Handle h) : mHandle(h) {}
  ~Stream()
  {
    if (mHandle != DefaultStream) {
#if defined(__HIPCC__)
      GPUChkErrS(hipStreamDestroy(mHandle));
      GPUChkErrS(hipEventDestroy(mEvent));
#elif defined(__CUDACC__)
      GPUChkErrS(cudaStreamDestroy(mHandle));
      GPUChkErrS(cudaEventDestroy(mEvent));
#endif
    }
  }

  operator bool() const { return mHandle != DefaultStream; }
  const Handle& get() { return mHandle; }
  const Handle& getStream() { return mHandle; }
  const Event& getEvent() { return mEvent; }
  void sync() const
  {
#if defined(__HIPCC__)
    GPUChkErrS(hipStreamSynchronize(mHandle));
#elif defined(__CUDACC__)
    GPUChkErrS(cudaStreamSynchronize(mHandle));
#endif
  }
  void record()
  {
#if defined(__HIPCC__)
    GPUChkErrS(hipEventRecord(mEvent, mHandle));
#elif defined(__CUDACC__)
    GPUChkErrS(cudaEventRecord(mEvent, mHandle));
#endif
  }

 private:
  Handle mHandle{DefaultStream};
  Event mEvent{nullptr};
};

// Abstract vector for streams.
class Streams
{
 public:
  size_t size() const noexcept { return mStreams.size(); }
  void resize(size_t n) { mStreams.resize(n); }
  void clear() { mStreams.clear(); }
  auto& operator[](size_t i) { return mStreams[i]; }
  void push_back(const Stream& stream) { mStreams.push_back(stream); }
  void sync(bool device = true)
  {
    if (device) {
#if defined(__HIPCC__)
      GPUChkErrS(hipDeviceSynchronize());
#elif defined(__CUDACC__)
      GPUChkErrS(cudaDeviceSynchronize());
#endif
    } else {
      for (auto& s : mStreams) {
        s.sync();
      }
    }
  }
  void waitEvent(size_t iStream, size_t iEvent)
  {
#if defined(__HIPCC__)
    GPUChkErrS(hipStreamWaitEvent(mStreams[iStream].get(), mStreams[iEvent].getEvent()));
#elif defined(__CUDACC__)
    GPUChkErrS(cudaStreamWaitEvent(mStreams[iStream].get(), mStreams[iEvent].getEvent()));
#endif
  }

 private:
  std::vector<Stream> mStreams;
};

#ifdef ITS_MEASURE_GPU_TIME
class GPUTimer
{
 public:
  GPUTimer(const std::string& name)
    : mName(name)
  {
    mStreams.emplace_back(Stream::DefaultStream);
    startTimers();
  }
  GPUTimer(Streams& streams, const std::string& name)
    : mName(name)
  {
    for (size_t i{0}; i < streams.size(); ++i) {
      mStreams.push_back(streams[i].get());
    }
    startTimers();
  }
  GPUTimer(Streams& streams, const std::string& name, size_t end, size_t start = 0)
    : mName(name)
  {
    for (size_t sta{start}; sta < end; ++sta) {
      mStreams.push_back(streams[sta].get());
    }
    startTimers();
  }
  GPUTimer(Stream& stream, const std::string& name, const int id = 0)
    : mName(name)
  {
    mStreams.push_back(stream.get());
    mName += ":id" + std::to_string(id);
    startTimers();
  }
  ~GPUTimer()
  {
    for (size_t i{0}; i < mStreams.size(); ++i) {
      float ms = 0.0f;
#if defined(__HIPCC__)
      GPUChkErrS(hipEventRecord(mStops[i], mStreams[i]));
      GPUChkErrS(hipEventSynchronize(mStops[i]));
      GPUChkErrS(hipEventElapsedTime(&ms, mStarts[i], mStops[i]));
      GPUChkErrS(hipEventDestroy(mStarts[i]));
      GPUChkErrS(hipEventDestroy(mStops[i]));
#elif defined(__CUDACC__)
      GPUChkErrS(cudaEventRecord(mStops[i], mStreams[i]));
      GPUChkErrS(cudaEventSynchronize(mStops[i]));
      GPUChkErrS(cudaEventElapsedTime(&ms, mStarts[i], mStops[i]));
      GPUChkErrS(cudaEventDestroy(mStarts[i]));
      GPUChkErrS(cudaEventDestroy(mStops[i]));
#endif
      LOGP(info, "Elapsed time for {}:{} {} ms", mName, i, ms);
    }
  }

  void startTimers()
  {
    mStarts.resize(mStreams.size());
    mStops.resize(mStreams.size());
    for (size_t i{0}; i < mStreams.size(); ++i) {
#if defined(__HIPCC__)
      GPUChkErrS(hipEventCreate(&mStarts[i]));
      GPUChkErrS(hipEventCreate(&mStops[i]));
      GPUChkErrS(hipEventRecord(mStarts[i], mStreams[i]));
#elif defined(__CUDACC__)
      GPUChkErrS(cudaEventCreate(&mStarts[i]));
      GPUChkErrS(cudaEventCreate(&mStops[i]));
      GPUChkErrS(cudaEventRecord(mStarts[i], mStreams[i]));
#endif
    }
  }

 private:
  std::string mName;
  std::vector<Stream::Event> mStarts, mStops;
  std::vector<Stream::Handle> mStreams;
};
#else // ITS_MEASURE_GPU_TIME not defined
class GPUTimer
{
 public:
  template <typename... Args>
  GPUTimer(Args&&...)
  {
  }
};
#endif

#ifdef GPUCA_GPUCODE
template <typename T>
struct TypedAllocator {
  using value_type = T;
  using pointer = thrust::device_ptr<T>;
  using const_pointer = thrust::device_ptr<const T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  TypedAllocator() noexcept : mInternalAllocator(nullptr) {}
  explicit TypedAllocator(ExternalAllocator* a) noexcept : mInternalAllocator(a) {}

  template <typename U>
  TypedAllocator(const TypedAllocator<U>& o) noexcept : mInternalAllocator(o.mInternalAllocator)
  {
  }

  pointer allocate(size_type n)
  {
    void* raw = mInternalAllocator->allocateStack(n * sizeof(T));
    return thrust::device_pointer_cast(static_cast<T*>(raw));
  }

  void deallocate(pointer p, size_type n) noexcept
  {
    if (!p) {
      return;
    }
    void* raw = thrust::raw_pointer_cast(p);
    mInternalAllocator->deallocate(static_cast<char*>(raw), n * sizeof(T));
  }

  bool operator==(TypedAllocator const& o) const noexcept
  {
    return mInternalAllocator == o.mInternalAllocator;
  }
  bool operator!=(TypedAllocator const& o) const noexcept
  {
    return !(*this == o);
  }

 private:
  ExternalAllocator* mInternalAllocator;
};

template <int nLayers>
GPUdii() const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                const o2::its::IndexTableUtils<nLayers>* utils,
                                const float z1, const float z2, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = o2::gpu::CAMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = (maxdeltaphi > o2::constants::math::PI) ? 0.f : currentCluster.phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::CAMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = (maxdeltaphi > o2::constants::math::PI) ? o2::constants::math::TwoPI : currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -utils->getLayerZ(layerIndex) ||
      zRangeMin > utils->getLayerZ(layerIndex) || zRangeMin > zRangeMax) {
    return {};
  }

  return int4{o2::gpu::CAMath::Max(0, utils->getZBinIndex(layerIndex, zRangeMin)),
              utils->getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::CAMath::Min(utils->getNzBins() - 1, utils->getZBinIndex(layerIndex, zRangeMax)),
              utils->getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

GPUdii() gpuSpan<const Vertex> getPrimaryVertices(const int rof,
                                                  const int* roframesPV,
                                                  const int nROF,
                                                  const uint8_t* mask,
                                                  const Vertex* vertices)
{
  const int start_pv_id = roframesPV[rof];
  const int stop_rof = rof >= nROF - 1 ? nROF : rof + 1;
  size_t delta = mask[rof] ? roframesPV[stop_rof] - start_pv_id : 0; // return empty span if ROF is excluded
  return gpuSpan<const Vertex>(&vertices[start_pv_id], delta);
};

GPUdii() gpuSpan<const Vertex> getPrimaryVertices(const int romin,
                                                  const int romax,
                                                  const int* roframesPV,
                                                  const int nROF,
                                                  const Vertex* vertices)
{
  const int start_pv_id = roframesPV[romin];
  const int stop_rof = romax >= nROF - 1 ? nROF : romax + 1;
  return gpuSpan<const Vertex>(&vertices[start_pv_id], roframesPV[stop_rof] - roframesPV[romin]);
};

GPUdii() gpuSpan<const Cluster> getClustersOnLayer(const int rof,
                                                   const int totROFs,
                                                   const int layer,
                                                   const int** roframesClus,
                                                   const Cluster** clusters)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<const Cluster>();
  }
  const int start_clus_id{roframesClus[layer][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[layer][stop_rof] - start_clus_id;
  return gpuSpan<const Cluster>(&(clusters[layer][start_clus_id]), delta);
}

GPUdii() gpuSpan<const Tracklet> getTrackletsPerCluster(const int rof,
                                                        const int totROFs,
                                                        const int mode,
                                                        const int** roframesClus,
                                                        const Tracklet** tracklets)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<const Tracklet>();
  }
  const int start_clus_id{roframesClus[1][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[1][stop_rof] - start_clus_id;
  return gpuSpan<const Tracklet>(&(tracklets[mode][start_clus_id]), delta);
}

GPUdii() gpuSpan<int> getNTrackletsPerCluster(const int rof,
                                              const int totROFs,
                                              const int mode,
                                              const int** roframesClus,
                                              int** ntracklets)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<int>();
  }
  const int start_clus_id{roframesClus[1][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[1][stop_rof] - start_clus_id;
  return gpuSpan<int>(&(ntracklets[mode][start_clus_id]), delta);
}

GPUdii() gpuSpan<const int> getNTrackletsPerCluster(const int rof,
                                                    const int totROFs,
                                                    const int mode,
                                                    const int** roframesClus,
                                                    const int** ntracklets)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<const int>();
  }
  const int start_clus_id{roframesClus[1][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[1][stop_rof] - start_clus_id;
  return gpuSpan<const int>(&(ntracklets[mode][start_clus_id]), delta);
}

GPUdii() gpuSpan<int> getNLinesPerCluster(const int rof,
                                          const int totROFs,
                                          const int** roframesClus,
                                          int* nlines)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<int>();
  }
  const int start_clus_id{roframesClus[1][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[1][stop_rof] - start_clus_id;
  return gpuSpan<int>(&(nlines[start_clus_id]), delta);
}

GPUdii() gpuSpan<const int> getNLinesPerCluster(const int rof,
                                                const int totROFs,
                                                const int** roframesClus,
                                                const int* nlines)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<const int>();
  }
  const int start_clus_id{roframesClus[1][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[1][stop_rof] - start_clus_id;
  return gpuSpan<const int>(&(nlines[start_clus_id]), delta);
}
#endif
} // namespace gpu
} // namespace o2::its

#endif
