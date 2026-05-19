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

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <format>
#include <string>
#include "GPUCommonLogger.h"
#endif
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/Constants.h"

namespace o2::its
{

// #define BOUNDED_MR_STATS
class BoundedMemoryResource final : public std::pmr::memory_resource
{
 public:
  class MemoryLimitExceeded final : public std::bad_alloc
  {
   public:
    MemoryLimitExceeded(size_t attempted, size_t used, size_t max)
    {
      char buf[256];
      if (attempted != 0) {
        (void)snprintf(buf, sizeof(buf), "Reached set memory limit (attempted: %zu, used: %zu, max: %zu)", attempted, used, max);
      } else {
        (void)snprintf(buf, sizeof(buf), "New set maximum below current used (newMax: %zu, used: %zu)", max, used);
      }
      mMsg = buf;
    }
    const char* what() const noexcept final { return mMsg.c_str(); }

   private:
    std::string mMsg;
  };

  BoundedMemoryResource(size_t maxBytes = std::numeric_limits<size_t>::max(),
                        std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
    : mMaxMemory(maxBytes), mUpstream(upstream) {}

  BoundedMemoryResource(ExternalAllocator* alloc,
                        size_t maxBytes = std::numeric_limits<size_t>::max())
    : mMaxMemory(maxBytes),
      mAdaptor(std::make_unique<ExternalAllocatorAdaptor>(alloc)),
      mUpstream(mAdaptor.get()) {}

  void* do_allocate(size_t bytes, size_t alignment) final
  {
    size_t new_used{0};
    size_t current_used{mUsedMemory.load(std::memory_order_relaxed)};
    do {
      new_used = current_used + bytes;
      if (new_used > mMaxMemory.load(std::memory_order_relaxed)) {
        mCountThrow.fetch_add(1, std::memory_order_relaxed);
        throw MemoryLimitExceeded(new_used, current_used,
                                  mMaxMemory.load(std::memory_order_relaxed));
      }
    } while (!mUsedMemory.compare_exchange_weak(current_used, new_used,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed));

    void* p{nullptr};
    try {
      p = mUpstream->allocate(bytes, alignment);
    } catch (...) {
      mUsedMemory.fetch_sub(bytes, std::memory_order_relaxed);
#ifdef BOUNDED_MR_STATS
      mStats.upstreamFailures.fetch_add(1, std::memory_order_relaxed);
#endif
      throw;
    }

#ifdef BOUNDED_MR_STATS
    size_t peak = mStats.peak.load(std::memory_order_relaxed);
    while (new_used > peak &&
           !mStats.peak.compare_exchange_weak(peak, new_used,
                                              std::memory_order_relaxed)) {
    }
    mStats.live.fetch_add(1, std::memory_order_relaxed);
    mStats.nAlloc.fetch_add(1, std::memory_order_relaxed);
    mStats.totalAlloc.fetch_add(bytes, std::memory_order_relaxed);

    size_t ma = mStats.maxAlign.load(std::memory_order_relaxed);
    while (alignment > ma && !mStats.maxAlign.compare_exchange_weak(ma, alignment, std::memory_order_relaxed)) {
    }
#endif
    return p;
  }

  void do_deallocate(void* p, size_t bytes, size_t alignment) final
  {
    mUpstream->deallocate(p, bytes, alignment);
    mUsedMemory.fetch_sub(bytes, std::memory_order_relaxed);
#ifdef BOUNDED_MR_STATS
    mStats.live.fetch_sub(1, std::memory_order_relaxed);
    mStats.nFree.fetch_add(1, std::memory_order_relaxed);
    mStats.totalFreed.fetch_add(bytes, std::memory_order_relaxed);
#endif
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final
  {
    return this == &other;
  }

  [[nodiscard]] size_t getUsedMemory() const noexcept
  {
    return mUsedMemory.load(std::memory_order_relaxed);
  }
  [[nodiscard]] size_t getMaxMemory() const noexcept
  {
    return mMaxMemory.load(std::memory_order_relaxed);
  }
  [[nodiscard]] size_t getThrowCount() const noexcept
  {
    return mCountThrow.load(std::memory_order_relaxed);
  }

  void setMaxMemory(size_t max)
  {
    size_t current = mMaxMemory.load(std::memory_order_relaxed);
    if (max == current) {
      return;
    }
    for (;;) {
      size_t used = mUsedMemory.load(std::memory_order_acquire);
      if (used > max) {
        mCountThrow.fetch_add(1, std::memory_order_relaxed);
        throw MemoryLimitExceeded(0, used, max);
      }
      if (mMaxMemory.compare_exchange_weak(current, max,
                                           std::memory_order_release,
                                           std::memory_order_relaxed)) {
        return;
      }
      if (current == max) {
        return;
      }
    }
  }

#if !defined(__HIPCC__) && !defined(__CUDACC__)
  std::string asString() const
  {
    const auto throw_ = mCountThrow.load(std::memory_order_relaxed);
    const auto used = static_cast<double>(mUsedMemory.load(std::memory_order_relaxed));
    const auto maxm = mMaxMemory.load(std::memory_order_relaxed);
    std::string ret;
    if (maxm == std::numeric_limits<size_t>::max()) {
      ret += std::format("maxthrow={} maxmem=unbounded used={:.2f} GB", throw_, used / constants::GB);
    } else {
      ret += std::format("maxthrow={} maxmem={:.2f} GB used={:.2f} GB ({:.2f}%)", throw_, (double)maxm / constants::GB, used / constants::GB, 100.0 * used / (double)maxm);
    }
#ifdef BOUNDED_MR_STATS
    ret += std::format("  peak={:.2f} GB live={} nAlloc={} nFree={} totalAlloc={:.2f} GB totalFreed={:.2f} GB maxAlign={} upstreamFail={}",
                       (float)mStats.peak.load(std::memory_order_relaxed) / constants::GB,
                       mStats.live.load(std::memory_order_relaxed),
                       mStats.nAlloc.load(std::memory_order_relaxed),
                       mStats.nFree.load(std::memory_order_relaxed),
                       (float)mStats.totalAlloc.load(std::memory_order_relaxed) / constants::GB,
                       (float)mStats.totalFreed.load(std::memory_order_relaxed) / constants::GB,
                       mStats.maxAlign.load(std::memory_order_relaxed),
                       mStats.upstreamFailures.load(std::memory_order_relaxed));
#endif
    return ret;
  }

  void print() const
  {
    LOGP(info, "{}", asString());
  }
#endif

 private:
  std::atomic<size_t> mMaxMemory{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> mCountThrow{0};
  std::atomic<size_t> mUsedMemory{0};
  std::unique_ptr<ExternalAllocatorAdaptor> mAdaptor{nullptr};
  std::pmr::memory_resource* mUpstream{nullptr};

#ifdef BOUNDED_MR_STATS
  struct Stats {
    std::atomic<size_t> peak{0};
    std::atomic<size_t> live{0};
    std::atomic<size_t> nAlloc{0};
    std::atomic<size_t> nFree{0};
    std::atomic<size_t> totalAlloc{0};
    std::atomic<size_t> totalFreed{0};
    std::atomic<size_t> maxAlign{0};
    std::atomic<size_t> upstreamFailures{0};
  };
  Stats mStats{};
#endif
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
