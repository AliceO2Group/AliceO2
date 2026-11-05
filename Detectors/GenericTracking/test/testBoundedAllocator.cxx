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

#define BOOST_TEST_MODULE GenericTrackingBoundedAllocator
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "GenericTracking/BoundedAllocator.h"

#include <stdexcept>

using namespace o2::gtrk;

namespace
{
class ThrowingResource final : public std::pmr::memory_resource
{
  void* do_allocate(size_t, size_t) final { throw std::bad_alloc{}; }
  void do_deallocate(void*, size_t, size_t) final {}
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final { return this == &other; }
};

struct ThrowingCopy {
  static inline int copiesBeforeFailure = -1;
  ThrowingCopy() = default;
  ThrowingCopy(const ThrowingCopy&)
  {
    if (copiesBeforeFailure == 0) {
      throw std::runtime_error{"copy"};
    }
    if (copiesBeforeFailure > 0) {
      --copiesBeforeFailure;
    }
  }
  ThrowingCopy& operator=(const ThrowingCopy&) = default;
};
} // namespace

BOOST_AUTO_TEST_CASE(huge_request_cannot_wrap_past_budget)
{
  BoundedMemoryResource pool;
  void* held = pool.allocate(8);
  BOOST_CHECK_THROW(static_cast<void>(pool.allocate(std::numeric_limits<size_t>::max())), BoundedMemoryResource::MemoryLimitExceeded);
  BOOST_CHECK_EQUAL(pool.getUsedMemory(), 8u);
  BOOST_CHECK_EQUAL(pool.getThrowCount(), 1u);
  pool.deallocate(held, 8);
  BOOST_CHECK_EQUAL(pool.getUsedMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(upstream_failure_rolls_back_reserved_bytes)
{
  ThrowingResource upstream;
  BoundedMemoryResource pool{1024, &upstream};
  BOOST_CHECK_THROW(static_cast<void>(pool.allocate(16)), std::bad_alloc);
  BOOST_CHECK_EQUAL(pool.getUsedMemory(), 0u);
  BOOST_CHECK_EQUAL(pool.getAllocationCount(), 0u);
  BOOST_CHECK_EQUAL(pool.getAllocatedBytes(), 0u);
}

BOOST_AUTO_TEST_CASE(legacy_mutable_budget_and_stage_peak_are_preserved)
{
  BoundedMemoryResource pool{64};
  void* first = pool.allocate(16);
  BOOST_CHECK_THROW(pool.setMaxMemory(8), BoundedMemoryResource::MemoryLimitExceeded);
  BOOST_CHECK_EQUAL(pool.getMaxMemory(), 64u);
  const size_t previousPeak = pool.resetPeakMemory();
  BOOST_CHECK_EQUAL(previousPeak, 16u);
  BOOST_CHECK_EQUAL(pool.getPeakMemoryDelta(), 0u);
  pool.setMaxMemory(24);
  void* second = pool.allocate(8);
  BOOST_CHECK_EQUAL(pool.getPeakMemoryDelta(), 8u);
  BOOST_CHECK_THROW(static_cast<void>(pool.allocate(1)), BoundedMemoryResource::MemoryLimitExceeded);
  pool.deallocate(second, 8);
  pool.deallocate(first, 16);
  BOOST_CHECK_EQUAL(pool.getAllocationCount(), 2u);
  BOOST_CHECK_EQUAL(pool.getAllocatedBytes(), 24u);
  BOOST_CHECK_EQUAL(pool.resetPeakMemory(), 24u);
  pool.raisePeakMemory(24);
  BOOST_CHECK_EQUAL(pool.getPeakMemory(), 24u);
}

BOOST_AUTO_TEST_CASE(failed_rebuild_leaves_valid_vector_on_requested_resource)
{
  BoundedMemoryResource pool{64};
  BoundedVector<int> values{&pool};
  values.assign(8, 7);
  BOOST_CHECK_THROW(clearResizeBoundedVector(values, 32, &pool, 3), BoundedMemoryResource::MemoryLimitExceeded);
  BOOST_CHECK(values.empty());
  BOOST_CHECK_EQUAL(pool.getUsedMemory(), 0u);
  BOOST_CHECK(values.get_allocator().resource() == &pool);
  values.push_back(9);
  BOOST_CHECK_EQUAL(values.front(), 9);
}

BOOST_AUTO_TEST_CASE(element_constructor_failure_leaves_vector_destructible)
{
  BoundedMemoryResource pool;
  BoundedVector<ThrowingCopy> values{&pool};
  values.resize(2);
  ThrowingCopy sample;
  ThrowingCopy::copiesBeforeFailure = 1;
  // The by-value default argument succeeds; assign throws after rebuilding.
  BOOST_CHECK_THROW(clearResizeBoundedVector(values, 4, &pool, sample), std::runtime_error);
  ThrowingCopy::copiesBeforeFailure = -1;
  BOOST_CHECK(values.empty());
  deepVectorClear(values);
  BOOST_CHECK_EQUAL(pool.getUsedMemory(), 0u);
  values.resize(1);
}

BOOST_AUTO_TEST_CASE(resize_reuses_capacity_and_honors_resource_changes)
{
  BoundedMemoryResource first, second;
  BoundedVector<int> values{&first};
  values.reserve(16);
  const auto allocations = first.getAllocationCount();
  auto* storage = values.data();
  resizeBoundedVector(values, 8);
  BOOST_CHECK(values.data() == storage);
  BOOST_CHECK_EQUAL(first.getAllocationCount(), allocations);
  resizeBoundedVector(values, 4, &second);
  BOOST_CHECK(values.get_allocator().resource() == &second);
  BOOST_CHECK_EQUAL(first.getUsedMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(nested_vectors_and_arrays_retain_legacy_clear_semantics)
{
  BoundedMemoryResource first, second;
  std::vector<BoundedVector<int>> nested;
  clearResizeBoundedVector(nested, 3, &first);
  for (auto& values : nested) {
    values.assign(4, 5);
  }
  deepVectorClear(nested, &second);
  BOOST_CHECK_EQUAL(nested.size(), 3u);
  BOOST_CHECK_EQUAL(first.getUsedMemory(), 0u);
  for (const auto& values : nested) {
    BOOST_CHECK(values.empty());
    BOOST_CHECK(values.get_allocator().resource() == &second);
  }
  std::array<BoundedVector<int>, 2> array;
  clearResizeBoundedArray(array, 4, &second, 7);
  BOOST_CHECK_EQUAL(array[0][0], 7);
  deepVectorClear(array, &first);
  BOOST_CHECK_EQUAL(second.getUsedMemory(), 0u);
  BOOST_CHECK(array[0].get_allocator().resource() == &first);
}
