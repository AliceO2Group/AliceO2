// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
/// \file SlabBumpAllocator.h
/// \brief Lock-free slot allocator and single-pass sink.
///

#ifndef TRACKINGITSU_INCLUDE_SLABBUMPALLOCATOR_H_
#define TRACKINGITSU_INCLUDE_SLABBUMPALLOCATOR_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

#include "ITStracking/BoundedAllocator.h"

namespace o2::its
{

class SlabBumpAllocator
{
 public:
  struct Range {
    size_t base{0};
    size_t n{0};
    bool valid() const noexcept { return n != 0; }
  };

  SlabBumpAllocator(size_t capacity, size_t slab) noexcept
    : mCapacity{capacity}, mSlab{slab ? slab : size_t{1}} {}

  Range grab() noexcept
  {
    if (mExhausted.load(std::memory_order_relaxed)) {
      return {};
    }
    const size_t base = mCursor.fetch_add(mSlab, std::memory_order_relaxed);
    if (base >= mCapacity) {
      mExhausted.store(true, std::memory_order_relaxed);
      return {};
    }
    return {.base = base, .n = std::min(mSlab, mCapacity - base)};
  }

  [[nodiscard]] size_t capacity() const noexcept { return mCapacity; }
  [[nodiscard]] size_t slab() const noexcept { return mSlab; }
  [[nodiscard]] size_t watermark() const noexcept
  {
    return std::min(mCursor.load(std::memory_order_relaxed), mCapacity);
  }

  static size_t suggestSlab(size_t capacity, int nThreads, size_t minSlab = 256, size_t maxSlab = 4096) noexcept
  {
    const size_t t = static_cast<size_t>(std::max(1, nThreads));
    const size_t fairShare = std::max<size_t>(1, capacity / t);
    return std::clamp(std::max<size_t>(1, capacity / (8 * t)),
                      std::min(minSlab, fairShare),
                      std::min(maxSlab, fairShare));
  }

  void resetCapacity(size_t capacity) noexcept
  {
    assert(mCursor.load(std::memory_order_relaxed) == 0);
    mCapacity = capacity;
    mExhausted.store(capacity == 0, std::memory_order_relaxed);
  }

 private:
  std::atomic<size_t> mCursor{0};
  std::atomic<bool> mExhausted{false};
  size_t mCapacity;
  size_t mSlab;
};

enum class SlabMode : uint8_t {
  Unordered,
  GroupedByProducer
};

struct SlabSinkStats {
  size_t requested{0}; ///< slots the caller predicted it would need
  size_t capacity{0};  ///< slots the memory pool actually granted
  size_t emitted{0};
  size_t spilled{0};
  bool overflowed{false};    ///< something did not fit into the staging area
  bool memoryLimited{false}; ///< the pool granted less than was requested
};

template <typename T, SlabMode Mode>
class SlabSink
{
  static constexpr int32_t NoProducer = -1;

 public:
  struct Config {
    size_t capacity{0};      ///< predicted number of slots
    int nThreads{1};         ///< workers that will feed this sink
    int nConcurrentSinks{1}; ///< sinks that may be alive on the same pool at the same time
    size_t slabOverride{0};  ///< 0: derive the slab size from the granted capacity
  };

  static constexpr size_t BytesPerSlot = Mode == SlabMode::GroupedByProducer ? (2 * sizeof(T)) + sizeof(int32_t) : sizeof(T);

  struct Run {
    size_t begin{0};
    size_t end{0};
  };

  class Handle
  {
   public:
    explicit Handle(SlabSink* sink)
      : mSink{sink}, mRuns{sink->memoryResource()}, mSpill{sink->memoryResource()}, mSpillProducer{sink->memoryResource()} {}

    void beginProducer(int32_t p) noexcept { mProducer = p; }

    template <typename... Args>
    void emplace(Args&&... args)
    {
      if constexpr (Mode == SlabMode::GroupedByProducer) {
        assert(mProducer != NoProducer);
      }
      ++mEmitted;
      if (mSlot == mSlotEnd && !refill()) {
        mSpill.emplace_back(std::forward<Args>(args)...);
        if constexpr (Mode == SlabMode::GroupedByProducer) {
          mSpillProducer.push_back(mProducer);
        }
        return;
      }
      mSink->store(mSlot++, mProducer, std::forward<Args>(args)...);
    }

    [[nodiscard]] size_t emitted() const noexcept { return mEmitted; }
    [[nodiscard]] size_t spilled() const noexcept { return mSpill.size(); }

   private:
    friend class SlabSink;

    bool refill()
    {
      if (mDrained) { // the arena is gone, do not touch the shared cursor again
        return false;
      }
      closeRun();
      const auto r = mSink->mAlloc.grab();
      if (!r.valid()) {
        mDrained = true;
        return false;
      }
      mRunBegin = r.base;
      mSlot = r.base;
      mSlotEnd = r.base + r.n;
      return true;
    }

    void closeRun()
    {
      if constexpr (Mode == SlabMode::Unordered) {
        if (mSlot > mRunBegin) {
          mRuns.push_back(Run{.begin = mRunBegin, .end = mSlot});
          mRunBegin = mSlot; // only advanced once push_back succeeded, so a throw can be retried
        }
      }
    }

    SlabSink* mSink{nullptr};
    size_t mSlot{0};
    size_t mSlotEnd{0};
    size_t mRunBegin{0};
    int32_t mProducer{NoProducer};
    bool mDrained{false};
    size_t mEmitted{0};
    bounded_vector<Run> mRuns;
    bounded_vector<T> mSpill;
    bounded_vector<int32_t> mSpillProducer;
  };

  SlabSink(const Config& cfg, std::pmr::memory_resource* mr)
    : SlabSink{cfg, grantedCapacity(cfg.capacity, cfg.nConcurrentSinks, mr), mr} {}

  SlabSink(SlabSink&&) = delete;
  SlabSink(const SlabSink&) = delete;
  SlabSink& operator=(SlabSink&&) = delete;
  SlabSink& operator=(const SlabSink&) = delete;
  ~SlabSink() = default;

  Handle& local() { return mHandles.local(); }

  [[nodiscard]] std::pmr::memory_resource* memoryResource() const noexcept { return mMR; }

  [[nodiscard]] SlabSinkStats stats() const
  {
    SlabSinkStats s;
    s.requested = mRequested;
    s.capacity = mAlloc.capacity();
    s.memoryLimited = s.capacity < s.requested;
    for (const auto& h : mHandles) {
      s.emitted += h.emitted();
      s.spilled += h.spilled();
    }
    s.overflowed = s.spilled != 0;
    return s;
  }

  void finalizeUnordered(bounded_vector<T>& dest)
  {
    static_assert(Mode == SlabMode::Unordered);
    assert(!mFinalized);
    assert(dest.get_allocator().resource()->is_equal(*mMR));
    mFinalized = true;

    bounded_vector<Run> runs{mMR};
    size_t nRuns{0};
    for (auto& h : mHandles) {
      h.closeRun();
      nRuns += h.mRuns.size();
    }
    runs.reserve(nRuns);
    for (const auto& h : mHandles) {
      runs.insert(runs.end(), h.mRuns.begin(), h.mRuns.end());
    }
    std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.begin < b.begin; });

    // Runs are disjoint and now ordered, so the compaction target never runs ahead of the source.
    size_t outputSize{0};
    for (const auto& run : runs) {
      for (size_t slot{run.begin}; slot < run.end; ++slot) {
        if (outputSize != slot) {
          mStaging[outputSize] = std::move(mStaging[slot]);
        }
        ++outputSize;
      }
    }
    deepVectorClear(runs, mMR);
    mStaging.resize(outputSize);
    dest.swap(mStaging);

    for (auto& h : mHandles) {
      dest.insert(dest.end(), std::make_move_iterator(h.mSpill.begin()), std::make_move_iterator(h.mSpill.end()));
      deepVectorClear(h.mSpill, mMR);
    }
    shrinkIfWasteful(dest);
    deepVectorClear(mStaging, mMR);
  }

  void finalizeGrouped(size_t nProducers, bounded_vector<int>& lut, bounded_vector<T>& dest)
  {
    static_assert(Mode == SlabMode::GroupedByProducer);
    assert(!mFinalized);
    mFinalized = true;
    const size_t wm = mAlloc.watermark();

    lut.assign(nProducers + 1, 0);

    for (size_t s = 0; s < wm; ++s) {
      const int32_t p = mProducerOf[s];
      if (p != NoProducer) {
        ++lut[p + 1];
      }
    }
    for (const auto& h : mHandles) {
      for (const int32_t p : h.mSpillProducer) {
        ++lut[p + 1];
      }
    }
    std::inclusive_scan(lut.begin(), lut.end(), lut.begin());

    bounded_vector<int> cursor(lut.begin(), lut.begin() + static_cast<ptrdiff_t>(nProducers), mMR);
    for (size_t s = 0; s < wm; ++s) {
      const int32_t p = mProducerOf[s];
      mProducerOf[s] = (p != NoProducer) ? cursor[p]++ : -1;
    }

    const auto total = static_cast<size_t>(lut.back());
    dest.resize(total);
    for (auto& h : mHandles) {
      for (size_t i = 0; i < h.mSpill.size(); ++i) {
        dest[cursor[h.mSpillProducer[i]]++] = std::move(h.mSpill[i]);
      }
      deepVectorClear(h.mSpill, mMR);
      deepVectorClear(h.mSpillProducer, mMR);
    }
    deepVectorClear(cursor, mMR);

    T* const staging = mStaging.data();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, wm, 4096), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t s = r.begin(); s != r.end(); ++s) {
        const int d = mProducerOf[s];
        if (d < 0) {
          continue;
        }
        dest[d] = std::move(staging[s]);
      }
    });

    deepVectorClear(mStaging, mMR);
    deepVectorClear(mProducerOf, mMR);
  }

 private:
  SlabSink(const Config& cfg, size_t granted, std::pmr::memory_resource* mr)
    : mMR{mr},
      mRequested{cfg.capacity},
      mAlloc{granted, cfg.slabOverride ? cfg.slabOverride : SlabBumpAllocator::suggestSlab(granted, cfg.nThreads)},
      mStaging{mr},
      mProducerOf{mr},
      mHandles{[this]() { return Handle{this}; }}
  {
    try {
      mStaging.resize(granted);
      if constexpr (Mode == SlabMode::GroupedByProducer) {
        mProducerOf.assign(granted, NoProducer);
      }
    } catch (const std::bad_alloc&) {
      discardPreallocation();
    } catch (const std::length_error&) {
      discardPreallocation();
    }
  }

  static size_t grantedCapacity(size_t requested, int nConcurrentSinks, const std::pmr::memory_resource* mr) noexcept
  {
    const auto* bounded = dynamic_cast<const BoundedMemoryResource*>(mr);
    if (bounded == nullptr) {
      return requested;
    }
    const size_t used = bounded->getUsedMemory();
    const size_t limit = bounded->getMaxMemory();
    const size_t remaining = used < limit ? limit - used : 0;
    // Keep half of what is left for the spill vectors and whatever else is still live, then
    // split the rest between the sinks that may be running on this pool at the same time.
    const size_t budget = (remaining / 2) / static_cast<size_t>(std::max(1, nConcurrentSinks));
    return std::min(requested, budget / BytesPerSlot);
  }

  static void shrinkIfWasteful(bounded_vector<T>& v)
  {
    if (v.capacity() > v.size() + (v.size() / 4)) {
      v.shrink_to_fit();
    }
  }

  void discardPreallocation()
  {
    // Capacity prediction is only an optimization; spilling preserves the output.
    deepVectorClear(mStaging, mMR);
    deepVectorClear(mProducerOf, mMR);
    mAlloc.resetCapacity(0);
  }

  template <typename... Args>
  void store(size_t slot, [[maybe_unused]] int32_t producer, Args&&... args)
  {
    mStaging[slot] = T(std::forward<Args>(args)...);
    if constexpr (Mode == SlabMode::GroupedByProducer) {
      mProducerOf[slot] = producer;
    }
  }

  std::pmr::memory_resource* mMR{nullptr};
  size_t mRequested{0};
  SlabBumpAllocator mAlloc;
  bounded_vector<T> mStaging;
  bounded_vector<int32_t> mProducerOf;
  tbb::enumerable_thread_specific<Handle> mHandles;
  bool mFinalized{false};
};

template <typename T>
using UnorderedSlabSink = SlabSink<T, SlabMode::Unordered>;

template <typename T>
using GroupedSlabSink = SlabSink<T, SlabMode::GroupedByProducer>;

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_SLABBUMPALLOCATOR_H_ */
