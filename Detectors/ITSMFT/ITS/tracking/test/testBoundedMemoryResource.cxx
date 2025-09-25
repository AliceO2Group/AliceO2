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

#define BOOST_TEST_MODULE Test Flags
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <random>
#include "ITStracking/BoundedAllocator.h"

using namespace o2::its;
using Vec = bounded_vector<int>;
auto getRandomInt(int min = -100, int max = 100)
{
  static std::mt19937 gen(std::random_device{}()); // static generator, seeded once
  std::uniform_int_distribution<> dist(min, max);
  return [&, dist]() mutable {
    return dist(gen);
  };
}

// -------- Throwing upstream resource for testing rollback --------
class ThrowingResource final : public std::pmr::memory_resource
{
 protected:
  void* do_allocate(size_t, size_t) final
  {
    throw std::bad_alloc(); // always fail
  }
  void do_deallocate(void*, size_t, size_t) noexcept final
  {
    // nothing
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final
  {
    return this == &other;
  }
};

// -------- Upstream resource with empty deallocate --------
class NoDeallocateResource final : public std::pmr::memory_resource
{
 public:
  NoDeallocateResource(std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
    : mUpstream(upstream) {}

 protected:
  void* do_allocate(size_t bytes, size_t alignment) final
  {
    return mUpstream->allocate(bytes, alignment);
  }
  void do_deallocate(void*, size_t, size_t) noexcept final
  {
    // nothing
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final
  {
    return this == &other;
  }

 private:
  std::pmr::memory_resource* mUpstream;
};

// -------- Tests --------
BOOST_AUTO_TEST_CASE(allocation_and_clear_updates_used_memory)
{
  BoundedMemoryResource bmr(10 * 1024 * 1024); // 10 MB cap

  Vec v(std::pmr::polymorphic_allocator<int>{&bmr});
  BOOST_CHECK_EQUAL(bmr.getUsedMemory(), 0u);

  const size_t count = 128;
  v.reserve(count);
  const size_t expected = count * sizeof(int);
  BOOST_CHECK_GE(bmr.getUsedMemory(), expected);
  BOOST_CHECK_LE(bmr.getUsedMemory(), expected + 64);

  deepVectorClear(v, &bmr);
  BOOST_CHECK_EQUAL(bmr.getUsedMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(clearResizeBoundedVector_resizes_and_tracks_memory)
{
  BoundedMemoryResource bmr(1024 * 1024); // 1 MB cap

  Vec v(std::pmr::polymorphic_allocator<int>{&bmr});
  v.reserve(200);
  const size_t used_before = bmr.getUsedMemory();
  BOOST_CHECK_GT(used_before, 0u);

  clearResizeBoundedVector(v, 50, &bmr, 7);
  const size_t used_after = bmr.getUsedMemory();
  BOOST_CHECK_GE(used_after, 50 * sizeof(int));
  BOOST_CHECK_LT(used_after, used_before);

  clearResizeBoundedVector(v, 300, &bmr, 3);
  BOOST_CHECK_GE(bmr.getUsedMemory(), 300 * sizeof(int));
}

BOOST_AUTO_TEST_CASE(upstream_throw_rolls_back_reservation)
{
  ThrowingResource upstream;
  BoundedMemoryResource bmr(std::numeric_limits<size_t>::max(), &upstream);
  const size_t bytes = 1024;
  bool threw = false;
  void* p{nullptr};
  try {
    p = bmr.allocate(bytes, alignof(std::max_align_t));
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  BOOST_CHECK(threw);
  BOOST_CHECK_EQUAL(p, nullptr);
  BOOST_CHECK_EQUAL(bmr.getUsedMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(vector_of_bounded_vectors_deep_clear_releases_all)
{
  BoundedMemoryResource bmr(10 * 1024 * 1024); // 10 MB
  std::vector<Vec> outer;
  outer.reserve(5);
  for (int i = 0; i < 5; ++i) {
    outer.emplace_back(std::pmr::polymorphic_allocator<int>{&bmr});
    outer.back().reserve(100);
  }
  BOOST_CHECK_GT(bmr.getUsedMemory(), 0u);
  deepVectorClear(outer, &bmr); // deep clear outer
  BOOST_CHECK_EQUAL(bmr.getUsedMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(array_of_bounded_vectors_clear_resize_works)
{
  BoundedMemoryResource bmr(10 * 1024 * 1024);
  std::array<Vec, 3> arr{{Vec(std::pmr::polymorphic_allocator<int>{&bmr}),
                          Vec(std::pmr::polymorphic_allocator<int>{&bmr}),
                          Vec(std::pmr::polymorphic_allocator<int>{&bmr})}};
  clearResizeBoundedVector(arr[0], 10, &bmr, 1);
  clearResizeBoundedVector(arr[1], 20, &bmr, 2);
  clearResizeBoundedVector(arr[2], 30, &bmr, 3);
  BOOST_CHECK_GT(bmr.getUsedMemory(), 0u);
  deepVectorClear(arr, &bmr); // now clear all recursively
  BOOST_CHECK_EQUAL(bmr.getUsedMemory(), 0u);
}

BOOST_AUTO_TEST_CASE(deepVectorClear_releases_and_reuses_resource)
{
  // Use a small bounded memory resource
  BoundedMemoryResource bmr(1024);
  bounded_vector<int> vec{std::pmr::polymorphic_allocator<int>{&bmr}};
  vec.resize(100, 42);
  BOOST_TEST(bmr.getUsedMemory() > 0);
  deepVectorClear(vec, &bmr);
  BOOST_TEST(vec.empty());
  BOOST_TEST(vec.get_allocator().resource() == &bmr);
  auto usedAfter = bmr.getUsedMemory();
  BOOST_CHECK_EQUAL(bmr.getUsedMemory(), 0);
  vec.push_back(7);
  BOOST_TEST(vec.size() == 1);
  BOOST_TEST(vec[0] == 7);
  BOOST_TEST(vec.get_allocator().resource() == &bmr);
}

BOOST_AUTO_TEST_CASE(clear_with_memory_resource_without_deallocator)
{
  NoDeallocateResource dmr;
  Vec v(std::pmr::polymorphic_allocator<int>{&dmr});

  for (int shift{0}; shift < 12; ++shift) {
    const int c{1 << shift};
    v.resize(100);
    std::generate(v.begin(), v.end(), getRandomInt());
    // allocate different sizes, which is actually a no-op now
    clearResizeBoundedVector(v, c / 2, &dmr, 999);
    for (size_t i{0}; i < c / 2; ++i) { // now only the first c/2 elements should be set
      BOOST_CHECK_EQUAL(v[i], 999);
    }
    // try to deepclear
    deepVectorClear(v);
  }
}
