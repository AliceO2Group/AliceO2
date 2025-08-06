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

#include "GPUCommonDef.h"
#include "GPUCommonHelpers.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

namespace o2::its
{

template <typename T1, typename T2>
struct gpuPair {
  T1 first;
  T2 second;
};

namespace gpu
{

// Poor man implementation of a span-like struct. It is very limited.
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

enum class Task {
  Tracker = 0,
  Vertexer = 1
};

// Abstract stream class
class Stream
{
 public:
#if defined(__HIPCC__)
  using Handle = hipStream_t;
  static constexpr Handle DefaultStream = 0;
  // static constexpr unsigned int DefaultFlag = hipStreamNonBlocking; TODO replace once ready
  static constexpr unsigned int DefaultFlag = 0;
#elif defined(__CUDACC__)
  using Handle = cudaStream_t;
  static constexpr Handle DefaultStream = 0;
  // static constexpr unsigned int DefaultFlag = cudaStreamNonBlocking; TODO replace once ready
  static constexpr unsigned int DefaultFlag = 0;
#else
  using Handle = void*;
  static constexpr Handle DefaultStream = nullptr;
  static constexpr unsigned int DefaultFlag = 0;
#endif

  Stream(unsigned int flags = DefaultFlag)
  {
#if defined(__HIPCC__)
    GPUChkErrS(hipStreamCreateWithFlags(&mHandle, flags));
#elif defined(__CUDACC__)
    GPUChkErrS(cudaStreamCreateWithFlags(&mHandle, flags));
#endif
  }

  Stream(Handle h) : mHandle(h) {}
  ~Stream()
  {
    if (mHandle != DefaultStream) {
#if defined(__HIPCC__)
      GPUChkErrS(hipStreamDestroy(mHandle));
#elif defined(__CUDACC__)
      GPUChkErrS(cudaStreamDestroy(mHandle));
#endif
    }
  }

  operator bool() const { return mHandle != DefaultStream; }
  const Handle& get() { return mHandle; }
  void sync() const
  {
#if defined(__HIPCC__)
    GPUChkErrS(hipStreamSynchronize(mHandle));
#elif defined(__CUDACC__)
    GPUChkErrS(cudaStreamSynchronize(mHandle));
#endif
  }

 private:
  Handle mHandle{DefaultStream};
};
static_assert(sizeof(Stream) == sizeof(void*), "Stream type must match pointer type!");

// Abstract vector for streams.
// Handles specifically wrap around.
class Streams
{
 public:
  size_t size() const noexcept { return mStreams.size(); }
  void resize(size_t n) { mStreams.resize(n); }
  void clear() { mStreams.clear(); }
  auto& operator[](size_t i) { return mStreams[i % mStreams.size()]; }
  void push_back(const Stream& stream) { mStreams.push_back(stream); }
  void sync()
  {
    for (auto& s : mStreams) {
      s.sync();
    }
  }

 private:
  std::vector<Stream> mStreams;
};

} // namespace gpu
} // namespace o2::its

#endif
