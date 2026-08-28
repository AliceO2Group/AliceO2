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

#include "ITSMFTTracking/BoundedAllocator.h"

#include <cstdio>
#include <format>

#include "GPUCommonLogger.h"
#include "ITSMFTTracking/Constants.h"

namespace o2::itsmft::tracking
{

BoundedMemoryResource::MemoryLimitExceeded::MemoryLimitExceeded(size_t attempted, size_t used, size_t max)
{
  char buf[256];
  if (attempted != 0) {
    (void)snprintf(buf, sizeof(buf), "Reached set memory limit (attempted: %zu, used: %zu, max: %zu)", attempted, used, max);
  } else {
    (void)snprintf(buf, sizeof(buf), "New set maximum below current used (newMax: %zu, used: %zu)", max, used);
  }
  mMsg = buf;
}

const char* BoundedMemoryResource::MemoryLimitExceeded::what() const noexcept
{
  return mMsg.c_str();
}

std::pmr::memory_resource* BoundedMemoryResource::cachingUpstream()
{
  static std::pmr::synchronized_pool_resource pool{std::pmr::get_default_resource()};
  return &pool;
}

BoundedMemoryResource::BoundedMemoryResource(size_t maxBytes, std::pmr::memory_resource* upstream)
  : mMaxMemory(maxBytes), mUpstream(upstream != nullptr ? upstream : cachingUpstream())
{
}

BoundedMemoryResource::BoundedMemoryResource(std::unique_ptr<std::pmr::memory_resource> upstream, size_t maxBytes)
  : mMaxMemory(maxBytes), mOwnedUpstream(std::move(upstream)), mUpstream(mOwnedUpstream.get())
{
}

void* BoundedMemoryResource::do_allocate(size_t bytes, size_t alignment)
{
  size_t newUsed{0};
  size_t currentUsed{mUsedMemory.load(std::memory_order_relaxed)};
  do {
    newUsed = currentUsed + bytes;
    if (newUsed > mMaxMemory.load(std::memory_order_relaxed)) {
      mCountThrow.fetch_add(1, std::memory_order_relaxed);
      throw MemoryLimitExceeded(newUsed, currentUsed, mMaxMemory.load(std::memory_order_relaxed));
    }
  } while (!mUsedMemory.compare_exchange_weak(currentUsed, newUsed, std::memory_order_acq_rel, std::memory_order_relaxed));

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

  size_t peak = mPeakUsedMemory.load(std::memory_order_relaxed);
  while (newUsed > peak && !mPeakUsedMemory.compare_exchange_weak(peak, newUsed, std::memory_order_relaxed)) {
  }

#ifdef BOUNDED_MR_STATS
  size_t statsPeak = mStats.peak.load(std::memory_order_relaxed);
  while (newUsed > statsPeak && !mStats.peak.compare_exchange_weak(statsPeak, newUsed, std::memory_order_relaxed)) {
  }
  mStats.live.fetch_add(1, std::memory_order_relaxed);
  mStats.nAlloc.fetch_add(1, std::memory_order_relaxed);
  mStats.totalAlloc.fetch_add(bytes, std::memory_order_relaxed);

  size_t maxAlignment = mStats.maxAlign.load(std::memory_order_relaxed);
  while (alignment > maxAlignment && !mStats.maxAlign.compare_exchange_weak(maxAlignment, alignment, std::memory_order_relaxed)) {
  }
#endif
  return p;
}

void BoundedMemoryResource::do_deallocate(void* p, size_t bytes, size_t alignment)
{
  mUpstream->deallocate(p, bytes, alignment);
  mUsedMemory.fetch_sub(bytes, std::memory_order_relaxed);
#ifdef BOUNDED_MR_STATS
  mStats.live.fetch_sub(1, std::memory_order_relaxed);
  mStats.nFree.fetch_add(1, std::memory_order_relaxed);
  mStats.totalFreed.fetch_add(bytes, std::memory_order_relaxed);
#endif
}

bool BoundedMemoryResource::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
  return this == &other;
}

size_t BoundedMemoryResource::getUsedMemory() const noexcept
{
  return mUsedMemory.load(std::memory_order_relaxed);
}

size_t BoundedMemoryResource::getMaxMemory() const noexcept
{
  return mMaxMemory.load(std::memory_order_relaxed);
}

size_t BoundedMemoryResource::getThrowCount() const noexcept
{
  return mCountThrow.load(std::memory_order_relaxed);
}

size_t BoundedMemoryResource::getPeakMemory() const noexcept
{
  return mPeakUsedMemory.load(std::memory_order_relaxed);
}

size_t BoundedMemoryResource::getPeakMemoryDelta() const noexcept
{
  const size_t peak = mPeakUsedMemory.load(std::memory_order_relaxed);
  const size_t baseline = mPeakBaselineMemory.load(std::memory_order_relaxed);
  return peak > baseline ? peak - baseline : 0;
}

void BoundedMemoryResource::resetPeakMemory() noexcept
{
  const size_t used = mUsedMemory.load(std::memory_order_acquire);
  mPeakBaselineMemory.store(used, std::memory_order_release);
  mPeakUsedMemory.store(used, std::memory_order_release);
}

void BoundedMemoryResource::setMaxMemory(size_t max)
{
  size_t current = mMaxMemory.load(std::memory_order_relaxed);
  if (max == current) {
    return;
  }
  for (;;) {
    const size_t used = mUsedMemory.load(std::memory_order_acquire);
    if (used > max) {
      mCountThrow.fetch_add(1, std::memory_order_relaxed);
      throw MemoryLimitExceeded(0, used, max);
    }
    if (mMaxMemory.compare_exchange_weak(current, max, std::memory_order_release, std::memory_order_relaxed)) {
      return;
    }
    if (current == max) {
      return;
    }
  }
}

std::string BoundedMemoryResource::asString() const
{
  const auto throwCount = mCountThrow.load(std::memory_order_relaxed);
  const auto used = static_cast<double>(mUsedMemory.load(std::memory_order_relaxed));
  const auto peak = static_cast<double>(mPeakUsedMemory.load(std::memory_order_relaxed));
  const auto peakDelta = static_cast<double>(getPeakMemoryDelta());
  const auto maxMemory = mMaxMemory.load(std::memory_order_relaxed);
  std::string result;
  if (maxMemory == std::numeric_limits<size_t>::max()) {
    result += std::format("maxthrow={} maxmem=unbounded used={:.2f} GB stagepeak={:.2f} GB stagealloc={:.2f} GB", throwCount, used / o2::its::constants::GB, peak / o2::its::constants::GB, peakDelta / o2::its::constants::GB);
  } else {
    result += std::format("maxthrow={} maxmem={:.2f} GB used={:.2f} GB ({:.2f}%) stagepeak={:.2f} GB stagealloc={:.2f} GB", throwCount, static_cast<double>(maxMemory) / o2::its::constants::GB, used / o2::its::constants::GB, 100.0 * used / static_cast<double>(maxMemory), peak / o2::its::constants::GB, peakDelta / o2::its::constants::GB);
  }
#ifdef BOUNDED_MR_STATS
  result += std::format("  peak={:.2f} GB live={} nAlloc={} nFree={} totalAlloc={:.2f} GB totalFreed={:.2f} GB maxAlign={} upstreamFail={}",
                        static_cast<float>(mStats.peak.load(std::memory_order_relaxed)) / o2::its::constants::GB,
                        mStats.live.load(std::memory_order_relaxed),
                        mStats.nAlloc.load(std::memory_order_relaxed),
                        mStats.nFree.load(std::memory_order_relaxed),
                        static_cast<float>(mStats.totalAlloc.load(std::memory_order_relaxed)) / o2::its::constants::GB,
                        static_cast<float>(mStats.totalFreed.load(std::memory_order_relaxed)) / o2::its::constants::GB,
                        mStats.maxAlign.load(std::memory_order_relaxed),
                        mStats.upstreamFailures.load(std::memory_order_relaxed));
#endif
  return result;
}

void BoundedMemoryResource::print() const
{
  LOGP(info, "{}", asString());
}

} // namespace o2::itsmft::tracking
