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
/// \brief Shared budgeted host allocation and bounded-vector utilities.
///
/// Reconciles GTRK's overflow-safe allocator, profiling counters and vector
/// rebuilds with the owned-resource, mutable-budget and stage-peak contracts
/// used by the upstream legacy ITS tracker and vertexer.
///

#ifndef ALICEO2_GENERICTRACKING_BOUNDEDALLOCATOR_H_
#define ALICEO2_GENERICTRACKING_BOUNDEDALLOCATOR_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace o2::gtrk
{

// #define BOUNDED_MR_STATS
class BoundedMemoryResource final : public std::pmr::memory_resource
{
 public:
  class MemoryLimitExceeded final : public std::bad_alloc
  {
   public:
    MemoryLimitExceeded(size_t attempted, size_t used, size_t max);
    const char* what() const noexcept final;

   private:
    std::string mMsg;
  };

  static std::pmr::memory_resource* cachingUpstream();

  BoundedMemoryResource(size_t maxBytes = std::numeric_limits<size_t>::max(),
                        std::pmr::memory_resource* upstream = nullptr);

  BoundedMemoryResource(std::unique_ptr<std::pmr::memory_resource> upstream,
                        size_t maxBytes = std::numeric_limits<size_t>::max());

  [[nodiscard]] size_t getUsedMemory() const noexcept;
  [[nodiscard]] size_t getMaxMemory() const noexcept;
  [[nodiscard]] size_t getThrowCount() const noexcept;
  [[nodiscard]] size_t getPeakMemory() const noexcept;
  [[nodiscard]] size_t getPeakMemoryDelta() const noexcept;

  // Reset the stage peak/baseline, returning the previous peak for nested profiling.
  size_t resetPeakMemory() noexcept;
  void raisePeakMemory(size_t peak) noexcept;
  [[nodiscard]] size_t getAllocationCount() const noexcept { return mAllocations.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t getAllocatedBytes() const noexcept { return mAllocatedBytes.load(std::memory_order_relaxed); }
  void setMaxMemory(size_t max);

#if !defined(__HIPCC__) && !defined(__CUDACC__)
  std::string asString() const;
  void print() const;
#endif

 private:
  void* do_allocate(size_t bytes, size_t alignment) final;
  void do_deallocate(void* p, size_t bytes, size_t alignment) final;
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final;

  std::atomic<size_t> mMaxMemory{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> mCountThrow{0};
  std::atomic<size_t> mUsedMemory{0};
  std::atomic<size_t> mAllocations{0};
  std::atomic<size_t> mAllocatedBytes{0};
  std::atomic<size_t> mPeakUsedMemory{0};
  std::atomic<size_t> mPeakBaselineMemory{0};
  std::unique_ptr<std::pmr::memory_resource> mOwnedUpstream;
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
using BoundedVector = std::pmr::vector<T>;

template <typename T>
using bounded_vector = BoundedVector<T>;

template <typename T>
inline void deepVectorClear(std::vector<T>& vec)
{
  std::vector<T>().swap(vec);
}

template <typename T>
inline void deepVectorClear(bounded_vector<T>& vec, std::pmr::memory_resource* mr = nullptr)
{
  std::pmr::memory_resource* tmr = (mr != nullptr) ? mr : vec.get_allocator().resource();
  std::destroy_at(&vec);
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
  // Rebuild a valid empty vector before allocation or element construction can
  // throw. The old allocation is released first, as required by bounded pools.
  deepVectorClear(vec, mr);
  vec.assign(sz, def);
}

// GTRK's capacity-reusing resize: retained storage is intended for arrays
// rewritten by the caller. A non-null different resource requires a rebuild.
template <typename T>
inline void resizeBoundedVector(BoundedVector<T>& vec, size_t sz, std::pmr::memory_resource* mr = nullptr)
{
  if (sz > vec.capacity() || (mr != nullptr && mr != vec.get_allocator().resource())) {
    clearResizeBoundedVector(vec, sz, mr);
    return;
  }
  vec.resize(sz);
}

// Release several bounded vectors onto one resource.
template <typename... T>
inline void deepVectorClear(std::pmr::memory_resource* mr, BoundedVector<T>&... vecs)
{
  (deepVectorClear(vecs, mr), ...);
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

} // namespace o2::gtrk

#endif /* ALICEO2_GENERICTRACKING_BOUNDEDALLOCATOR_H_ */
