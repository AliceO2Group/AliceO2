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

#include "ITSMFTTracking/SlabBumpAllocator.h"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

namespace o2::itsmft::tracking
{

namespace detail
{

struct ThreadLocalStorage::Impl {
  Impl(void* context_, Factory factory_, Deleter deleter_)
    : context{context_}, factory{factory_}, deleter{deleter_}, values{[this] { return factory(context); }}
  {
  }

  ~Impl()
  {
    for (void* value : values) {
      deleter(value);
    }
  }

  void* context;
  Factory factory;
  Deleter deleter;
  tbb::enumerable_thread_specific<void*> values;
};

ThreadLocalStorage::ThreadLocalStorage(void* context, Factory factory, Deleter deleter)
  : mImpl{std::make_unique<Impl>(context, factory, deleter)}
{
}

ThreadLocalStorage::~ThreadLocalStorage() = default;

void* ThreadLocalStorage::local()
{
  return mImpl->values.local();
}

std::vector<void*> ThreadLocalStorage::values() const
{
  return {mImpl->values.begin(), mImpl->values.end()};
}

void parallelFor(size_t begin, size_t end, size_t grainSize, void* context, ParallelForBody body)
{
  tbb::parallel_for(tbb::blocked_range<size_t>{begin, end, grainSize}, [context, body](const tbb::blocked_range<size_t>& range) {
    body(context, range.begin(), range.end());
  });
}

} // namespace detail

SlabBumpAllocator::SlabBumpAllocator(size_t capacity, size_t slab) noexcept
  : mCapacity{capacity}, mSlab{slab ? slab : size_t{1}}
{
}

SlabBumpAllocator::Range SlabBumpAllocator::grab() noexcept
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

size_t SlabBumpAllocator::watermark() const noexcept
{
  return std::min(mCursor.load(std::memory_order_relaxed), mCapacity);
}

size_t SlabBumpAllocator::suggestSlab(size_t capacity, int nThreads, size_t minSlab, size_t maxSlab) noexcept
{
  const size_t threads = static_cast<size_t>(std::max(1, nThreads));
  const size_t fairShare = std::max<size_t>(1, capacity / threads);
  return std::clamp(std::max<size_t>(1, capacity / (8 * threads)),
                    std::min(minSlab, fairShare),
                    std::min(maxSlab, fairShare));
}

void SlabBumpAllocator::resetCapacity(size_t capacity) noexcept
{
  assert(mCursor.load(std::memory_order_relaxed) == 0);
  mCapacity = capacity;
  mExhausted.store(capacity == 0, std::memory_order_relaxed);
}

} // namespace o2::itsmft::tracking
