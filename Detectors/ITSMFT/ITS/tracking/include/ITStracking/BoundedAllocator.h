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
/// \file BoundedAllocator.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_BOUNDEDALLOCATOR_H_
#define TRACKINGITSU_INCLUDE_BOUNDEDALLOCATOR_H_

#include <limits>
#include <memory_resource>
#include <atomic>
#include <new>
#include <vector>

#include "ITStracking/ExternalAllocator.h"

#include "GPUCommonLogger.h"

namespace o2::its
{

class BoundedMemoryResource final : public std::pmr::memory_resource
{
 public:
  class MemoryLimitExceeded final : public std::bad_alloc
  {
   public:
    MemoryLimitExceeded(size_t attempted, size_t used, size_t max)
      : mAttempted(attempted), mUsed(used), mMax(max) {}
    const char* what() const noexcept final
    {
      static thread_local char msg[256];
      if (mAttempted != 0) {
        snprintf(msg, sizeof(msg),
                 "Reached set memory limit (attempted: %zu, used: %zu, max: %zu)",
                 mAttempted, mUsed, mMax);
      } else {
        snprintf(msg, sizeof(msg),
                 "New set maximum below current used (newMax: %zu, used: %zu)",
                 mMax, mUsed);
      }
      return msg;
    }

   private:
    size_t mAttempted{0}, mUsed{0}, mMax{0};
  };

  BoundedMemoryResource(size_t maxBytes = std::numeric_limits<size_t>::max(), std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
    : mMaxMemory(maxBytes), mUpstream(upstream) {}
  BoundedMemoryResource(ExternalAllocator* alloc) : mAdaptor(std::make_unique<ExternalAllocatorAdaptor>(alloc)), mUpstream(mAdaptor.get()) {}

  void* do_allocate(size_t bytes, size_t alignment) final
  {
    size_t new_used{0}, current_used{mUsedMemory.load(std::memory_order_relaxed)};
    do {
      new_used = current_used + bytes;
      if (new_used > mMaxMemory) {
        ++mCountThrow;
        throw MemoryLimitExceeded(new_used, current_used, mMaxMemory);
      }
    } while (!mUsedMemory.compare_exchange_weak(current_used, new_used,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed));
    void* p{nullptr};
    try {
      p = mUpstream->allocate(bytes, alignment);
    } catch (...) {
      mUsedMemory.fetch_sub(bytes, std::memory_order_relaxed);
      throw;
    }
    return p;
  }

  void do_deallocate(void* p, size_t bytes, size_t alignment) final
  {
    mUpstream->deallocate(p, bytes, alignment);
    mUsedMemory.fetch_sub(bytes, std::memory_order_relaxed);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final
  {
    return this == &other;
  }

  size_t getUsedMemory() const noexcept { return mUsedMemory.load(); }
  size_t getMaxMemory() const noexcept { return mMaxMemory; }
  void setMaxMemory(size_t max)
  {
    if (max == mMaxMemory) {
      return;
    }
    size_t used = mUsedMemory.load(std::memory_order_acquire);
    if (used > max) {
      ++mCountThrow;
      throw MemoryLimitExceeded(0, used, max);
    }
    mMaxMemory.store(max, std::memory_order_release);
  }

  void print() const
  {
#if !defined(GPUCA_GPUCODE_DEVICE)
    constexpr double GB{1024 * 1024 * 1024};
    auto throw_ = mCountThrow.load(std::memory_order_relaxed);
    auto used = static_cast<double>(mUsedMemory.load(std::memory_order_relaxed));
    LOGP(info, "maxthrow={} maxmem={:.2f} GB used={:.2f} ({:.2f}%)",
         throw_, (double)mMaxMemory / GB, used / GB, 100. * used / (double)mMaxMemory);
#endif
  }

 private:
  std::atomic<size_t> mMaxMemory{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> mCountThrow{0};
  std::atomic<size_t> mUsedMemory{0};
  std::unique_ptr<ExternalAllocatorAdaptor> mAdaptor{nullptr};
  std::pmr::memory_resource* mUpstream{nullptr};
};

template <typename T>
using bounded_vector = std::pmr::vector<T>;

template <typename T>
inline void deepVectorClear(std::vector<T>& vec)
{
  std::vector<T>().swap(vec);
}

template <typename T>
inline void deepVectorClear(bounded_vector<T>& vec, std::pmr::memory_resource* mr = nullptr)
{
  std::pmr::memory_resource* tmr = (mr != nullptr) ? mr : vec.get_allocator().resource();
  vec.~bounded_vector<T>();
  new (&vec) bounded_vector<T>(std::pmr::polymorphic_allocator<T>{tmr});
}

template <typename T>
inline void deepVectorClear(std::vector<bounded_vector<T>>& vec, std::pmr::memory_resource* mr = nullptr)
{
  for (auto& v : vec) {
    deepVectorClear(v, mr);
  }
}

template <typename T, size_t S>
inline void deepVectorClear(std::array<bounded_vector<T>, S>& arr, std::pmr::memory_resource* mr = nullptr)
{
  for (size_t i{0}; i < S; ++i) {
    deepVectorClear(arr[i], mr);
  }
}

template <typename T>
inline void clearResizeBoundedVector(bounded_vector<T>& vec, size_t sz, std::pmr::memory_resource* mr = nullptr, T def = T())
{
  std::pmr::memory_resource* tmr = (mr != nullptr) ? mr : vec.get_allocator().resource();
  vec.~bounded_vector<T>();
  new (&vec) bounded_vector<T>(sz, def, std::pmr::polymorphic_allocator<T>{tmr});
}

template <typename T>
inline void clearResizeBoundedVector(std::vector<bounded_vector<T>>& vec, size_t size, std::pmr::memory_resource* mr)
{
  vec.clear();
  vec.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    vec.emplace_back(std::pmr::polymorphic_allocator<bounded_vector<T>>{mr});
  }
}

template <typename T, size_t S>
inline void clearResizeBoundedArray(std::array<bounded_vector<T>, S>& arr, size_t size, std::pmr::memory_resource* mr = nullptr, T def = T())
{
  for (size_t i{0}; i < S; ++i) {
    clearResizeBoundedVector(arr[i], size, mr, def);
  }
}

template <typename T>
inline std::vector<T> toSTDVector(const bounded_vector<T>& b)
{
  std::vector<T> t(b.size());
  std::copy(b.cbegin(), b.cend(), t.begin());
  return t;
}

} // namespace o2::its

#endif
