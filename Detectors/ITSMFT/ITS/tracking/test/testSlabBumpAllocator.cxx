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

#define BOOST_TEST_MODULE Test SlabBumpAllocator
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <new>
#include <random>
#include <vector>

#include <oneapi/tbb/task_arena.h>

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/CapacityEstimator.h"
#include "ITStracking/SlabBumpAllocator.h"

using namespace o2::its;

namespace
{

struct Rec {
  int a{-1};
  int b{-1};
  float payload{0.f};
  Rec() = default;
  Rec(int aa, int bb, float p) : a{aa}, b{bb}, payload{p} {}
  bool operator<(const Rec& o) const
  {
    if ((a < 0) != (o.a < 0)) {
      return o.a < 0;
    }
    return a != o.a ? a < o.a : b < o.b;
  }
  bool operator==(const Rec& o) const { return a == o.a && b == o.b; }
};

std::ostream& operator<<(std::ostream& os, const Rec& r)
{
  return os << "Rec{" << r.a << ',' << r.b << ',' << r.payload << '}';
}

class StingyResource final : public std::pmr::memory_resource
{
 public:
  explicit StingyResource(size_t maxBytes) : mMax{maxBytes} {}

 private:
  void* do_allocate(size_t bytes, size_t alignment) final
  {
    if (bytes > mMax) {
      throw std::bad_alloc{};
    }
    return std::pmr::new_delete_resource()->allocate(bytes, alignment);
  }
  void do_deallocate(void* p, size_t bytes, size_t alignment) final
  {
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept final { return this == &other; }

  size_t mMax;
};

template <typename F>
void runConcurrently(F&& f)
{
  tbb::task_arena arena{4};
  arena.execute(std::forward<F>(f));
}

template <typename Emit>
void produce(int i, uint32_t seed, Emit&& emit)
{
  std::mt19937 rng(seed + (uint32_t(i) * 2654435761u));
  const int n = int(rng() % 12);
  for (int k = 0; k < n; ++k) {
    emit(i, k, float((i * 100) + k));
  }
}

std::vector<std::vector<Rec>> reference(int nProducers, uint32_t seed)
{
  std::vector<std::vector<Rec>> out(nProducers);
  for (int i = 0; i < nProducers; ++i) {
    produce(i, seed, [&](int a, int b, float p) { out[i].emplace_back(a, b, p); });
  }
  return out;
}

void checkGrouped(int nProducers, size_t capacity, size_t slab, size_t maxMemory = std::numeric_limits<size_t>::max())
{
  constexpr uint32_t seed = 7u;
  BoundedMemoryResource mr{maxMemory};

  const auto ref = reference(nProducers, seed);
  std::vector<Rec> flat;
  std::vector<int> refLut(nProducers + 1, 0);
  for (int i = 0; i < nProducers; ++i) {
    refLut[i + 1] = refLut[i] + int(ref[i].size());
    flat.insert(flat.end(), ref[i].begin(), ref[i].end());
  }

  GroupedSlabSink<Rec> sink{{.capacity = capacity, .nThreads = 4, .slabOverride = slab}, &mr};
  runConcurrently([&] {
    tbb::parallel_for(0, nProducers, [&](int i) {
      auto& h = sink.local();
      h.beginProducer(i);
      produce(i, seed, [&](int a, int b, float p) { h.emplace(a, b, p); });
    });
  });

  const auto st = sink.stats();
  BOOST_TEST(st.emitted == flat.size());

  bounded_vector<int> lut{&mr};
  bounded_vector<Rec> dest{&mr};
  sink.finalizeGrouped(size_t(nProducers), lut, dest);

  BOOST_REQUIRE(lut.size() == size_t(nProducers) + 1);
  BOOST_TEST(std::equal(lut.begin(), lut.end(), refLut.begin()));
  BOOST_REQUIRE(dest.size() == flat.size());
  for (size_t i = 0; i < flat.size(); ++i) {
    BOOST_TEST(dest[i] == flat[i]);
    BOOST_TEST(dest[i].payload == flat[i].payload);
  }
}

void checkUnordered(int nProducers, size_t capacity, size_t slab, size_t maxMemory = std::numeric_limits<size_t>::max())
{
  constexpr uint32_t seed = 11u;
  BoundedMemoryResource mr{maxMemory};

  const auto ref = reference(nProducers, seed);
  std::vector<Rec> flat;
  for (const auto& v : ref) {
    flat.insert(flat.end(), v.begin(), v.end());
  }
  std::sort(flat.begin(), flat.end());
  flat.erase(std::unique(flat.begin(), flat.end()), flat.end());

  UnorderedSlabSink<Rec> sink{{.capacity = capacity, .nThreads = 4, .slabOverride = slab}, &mr};
  runConcurrently([&] {
    tbb::parallel_for(0, nProducers, [&](int i) {
      auto& h = sink.local();
      produce(i, seed, [&](int a, int b, float p) { h.emplace(a, b, p); });
    });
  });

  const auto st = sink.stats();
  BOOST_TEST(st.emitted == flat.size());

  bounded_vector<Rec> dest{&mr};
  sink.finalizeUnordered(dest);

  std::sort(dest.begin(), dest.end());

  BOOST_REQUIRE(dest.size() == flat.size());
  for (size_t i = 0; i < flat.size(); ++i) {
    BOOST_TEST(dest[i] == flat[i]);
    BOOST_TEST(dest[i].payload == flat[i].payload);
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(slab_hands_out_disjoint_ranges)
{
  SlabBumpAllocator alloc{1000, 256};
  std::vector<char> seen(1000, 0);
  size_t got{0};
  while (true) {
    const auto r = alloc.grab();
    if (!r.valid()) {
      break;
    }
    BOOST_REQUIRE(r.base + r.n <= 1000);
    for (size_t s = r.base; s < r.base + r.n; ++s) {
      BOOST_REQUIRE(seen[s] == 0);
      seen[s] = 1;
    }
    got += r.n;
  }
  BOOST_TEST(got == 1000u);
  BOOST_TEST(alloc.watermark() <= 1000u);
}

BOOST_AUTO_TEST_CASE(slab_never_exceeds_a_threads_fair_share)
{
  BOOST_TEST(SlabBumpAllocator::suggestSlab(64, 8) <= 8u);
  BOOST_TEST(SlabBumpAllocator::suggestSlab(0, 8) >= 1u);
  BOOST_TEST(SlabBumpAllocator::suggestSlab(1u << 20, 8) == 4096u);
}

BOOST_AUTO_TEST_CASE(grouped_reproduces_two_pass_layout)
{
  checkGrouped(2000, 40000, 512);
  checkGrouped(300, 20000, 4096);
}

BOOST_AUTO_TEST_CASE(grouped_survives_capacity_underestimate)
{
  checkGrouped(2000, 3000, 256);
  checkGrouped(500, 0, 1, 1u << 20);
}

BOOST_AUTO_TEST_CASE(grouped_survives_capacity_overestimate)
{
  checkGrouped(20, 1u << 20, 256, 1u << 16);
}

BOOST_AUTO_TEST_CASE(grouped_keeps_order_across_slab_and_spill_boundaries)
{
  BoundedMemoryResource mr;
  const std::vector<int> counts{3, 5, 6, 0, 2};
  GroupedSlabSink<Rec> sink{{.capacity = 10, .nThreads = 1, .slabOverride = 4}, &mr};

  auto& h = sink.local();
  for (size_t p = 0; p < counts.size(); ++p) {
    h.beginProducer(int(p));
    for (int k = 0; k < counts[p]; ++k) {
      h.emplace(int(p), k, float(k));
    }
  }
  const auto st = sink.stats();
  BOOST_TEST(st.emitted == 16u);
  BOOST_TEST(st.spilled == 6u); // capacity 10 of 16
  BOOST_TEST(st.overflowed);

  bounded_vector<int> lut{&mr};
  bounded_vector<Rec> dest{&mr};
  sink.finalizeGrouped(counts.size(), lut, dest);

  BOOST_REQUIRE(lut.size() == counts.size() + 1);
  BOOST_REQUIRE(dest.size() == 16u);
  int expected{0};
  for (size_t p = 0; p < counts.size(); ++p) {
    BOOST_TEST(lut[p] == expected);
    for (int k = 0; k < counts[p]; ++k) {
      BOOST_TEST(dest[expected + k] == Rec(int(p), k, 0.f));
    }
    expected += counts[p];
  }
  BOOST_TEST(lut.back() == expected);
}

BOOST_AUTO_TEST_CASE(unordered_reproduces_emitted_records)
{
  checkUnordered(2000, 40000, 512);
  checkUnordered(300, 20000, 4096);
}

BOOST_AUTO_TEST_CASE(unordered_survives_capacity_underestimate)
{
  checkUnordered(2000, 3000, 256);
  checkUnordered(500, 0, 1, 1u << 20);
}

BOOST_AUTO_TEST_CASE(unordered_keeps_records_across_slab_and_spill_boundaries)
{
  BoundedMemoryResource mr;
  UnorderedSlabSink<Rec> sink{{.capacity = 10, .nThreads = 1, .slabOverride = 4}, &mr};

  auto& h = sink.local();
  for (int i = 0; i < 14; ++i) {
    h.emplace(i, i + 1, float(i));
  }
  const auto st = sink.stats();
  BOOST_TEST(st.emitted == 14u);
  BOOST_TEST(st.spilled == 4u);

  bounded_vector<Rec> dest{&mr};
  sink.finalizeUnordered(dest);

  BOOST_REQUIRE(dest.size() == 14u);
  for (int i = 0; i < 14; ++i) {
    BOOST_TEST(dest[i] == Rec(i, i + 1, float(i)));
  }
}

BOOST_AUTO_TEST_CASE(unordered_removes_unused_slots)
{
  BoundedMemoryResource mr;
  UnorderedSlabSink<Rec> sink{{.capacity = 10, .nThreads = 1, .slabOverride = 4}, &mr};
  sink.local().emplace(1, 2, 3.f);
  sink.local().emplace();

  bounded_vector<Rec> dest{&mr};
  sink.finalizeUnordered(dest);

  BOOST_REQUIRE(dest.size() == 2u);
  BOOST_TEST(dest.front() == Rec(1, 2, 3.f));
  BOOST_TEST(dest.front().payload == 3.f);
  BOOST_TEST(dest.back() == Rec{});
}

BOOST_AUTO_TEST_CASE(unordered_does_not_hand_back_an_oversized_buffer)
{
  BoundedMemoryResource mr;
  UnorderedSlabSink<Rec> sink{{.capacity = 100000, .nThreads = 1, .slabOverride = 256}, &mr};

  auto& h = sink.local();
  for (int i = 0; i < 100; ++i) {
    h.emplace(i, i + 1, float(i));
  }
  bounded_vector<Rec> dest{&mr};
  sink.finalizeUnordered(dest);

  BOOST_REQUIRE(dest.size() == 100u);
  BOOST_TEST(dest.capacity() < 1000u);
}

BOOST_AUTO_TEST_CASE(capacity_is_clamped_to_what_the_pool_can_spare)
{
  constexpr size_t maxMemory = 1u << 16;
  BoundedMemoryResource mr{maxMemory};
  UnorderedSlabSink<Rec> sink{{.capacity = 1u << 20, .nThreads = 4}, &mr};

  const auto st = sink.stats();
  BOOST_TEST(st.requested == size_t{1u << 20});
  BOOST_TEST(st.capacity > 0u);
  BOOST_TEST(st.capacity < st.requested);
  BOOST_TEST(st.memoryLimited);
  BOOST_TEST(st.capacity * sizeof(Rec) <= maxMemory / 2);
}

BOOST_AUTO_TEST_CASE(capacity_is_split_between_concurrent_sinks)
{
  size_t alone{0}, shared{0};
  {
    BoundedMemoryResource mr{1u << 16};
    UnorderedSlabSink<Rec> sink{{.capacity = 1u << 20, .nThreads = 4, .nConcurrentSinks = 1}, &mr};
    alone = sink.stats().capacity;
  }
  {
    BoundedMemoryResource mr{1u << 16};
    UnorderedSlabSink<Rec> sink{{.capacity = 1u << 20, .nThreads = 4, .nConcurrentSinks = 4}, &mr};
    shared = sink.stats().capacity;
  }
  BOOST_TEST(shared > 0u);
  BOOST_TEST(shared < alone);
  BOOST_TEST(shared * 4 <= alone + 8); // integer division slack
}

BOOST_AUTO_TEST_CASE(unordered_survives_a_failed_preallocation)
{
  StingyResource mr{1u << 12};
  UnorderedSlabSink<Rec> sink{{.capacity = 1u << 20, .nThreads = 1}, &mr};

  const auto st = sink.stats();
  BOOST_TEST(st.capacity == 0u);
  BOOST_TEST(st.memoryLimited);

  auto& handle = sink.local();
  for (int i = 0; i < 10; ++i) {
    handle.emplace(i, i + 1, float(i));
  }

  bounded_vector<Rec> dest{&mr};
  sink.finalizeUnordered(dest);
  BOOST_REQUIRE(dest.size() == 10u);
  for (int i = 0; i < 10; ++i) {
    BOOST_TEST(dest[i] == Rec(i, i + 1, float(i)));
  }
}

BOOST_AUTO_TEST_CASE(estimator_cold_start_has_capacity)
{
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Cells, 0, 0, 3);
  BOOST_TEST(est.capacity(key, 1000.) == 1024u);

  est.update(key, 0., 0, 0, false, false);
  BOOST_TEST(est.capacity(key, 0.) == 0u);
  BOOST_TEST(est.capacity(key, 1000.) == 1024u);

  est.update(key, 1000., 0, 1024, false, false);
  BOOST_TEST(est.capacity(key, 1000.) == 1024u);
}

BOOST_AUTO_TEST_CASE(estimator_converges_and_reacts_to_overflow)
{
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, 1, 0, 0);
  constexpr double scale = 1000.;
  constexpr double rate = 5.;

  for (int tf = 0; tf < 12; ++tf) {
    const size_t cap = est.capacity(key, scale);
    const auto emitted = size_t(scale * rate);
    est.update(key, scale, emitted, cap != 0 ? cap : emitted, cap != 0 && emitted > cap, false);
  }

  const size_t cap = est.capacity(key, scale);
  BOOST_TEST(cap >= size_t(scale * rate));
  BOOST_TEST(cap <= size_t(scale * rate * 1.35));

  const size_t bigger = est.capacity(key, 2. * scale);
  BOOST_TEST(bigger > size_t(2. * scale * rate));
  BOOST_TEST(bigger <= size_t(2. * scale * rate * 1.35));

  est.update(key, scale, size_t(scale * rate * 4.), size_t(scale * rate), true, false);
  BOOST_TEST(est.capacity(key, scale) > cap);
}

BOOST_AUTO_TEST_CASE(estimator_does_not_extrapolate_a_low_statistics_ratio)
{
  // A first sample taken on a handful of inputs sets the ratio outright, so without a ceiling the
  // next timeframe would ask for a slab orders of magnitude past anything the site ever emitted.
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 2, CapacityEstimator::makeVariant(3, 3), 5);
  constexpr size_t emitted = 100000;

  est.update(key, 2., emitted, est.capacity(key, 2.), true, false); // ratio of 50000, from two inputs

  const size_t asked = est.capacity(key, 500000.);
  BOOST_TEST(asked <= emitted * 4u); // bounded by what this site has ever actually produced
  BOOST_TEST(asked >= emitted);      // but still enough headroom not to force a pointless retry
}

BOOST_AUTO_TEST_CASE(estimator_reports_a_scale_independent_peak)
{
  // Sizing a buffer that has to serve several differently sized runs cannot use capacity(), which
  // needs the scale of one particular run.
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 0, CapacityEstimator::makeVariant(5, 3), 2);
  BOOST_TEST(est.peakCapacity(key) == 1024u); // cold start falls back to the floor

  est.update(key, 1000., 50000, 60000, false, false);
  BOOST_TEST(est.peakCapacity(key) >= 50000u);

  est.update(key, 10., 700, 1024, false, false); // a much smaller run must not shrink the peak
  BOOST_TEST(est.peakCapacity(key) >= 50000u);
  BOOST_TEST(est.peakCapacity(key) <= 50000u * 4u);
}

BOOST_AUTO_TEST_CASE(estimator_expected_tracks_the_current_input)
{
  // Chaining sites whose input is the previous one's output needs a margin-free prediction that
  // follows this timeframe, not the largest one ever seen.
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 0, CapacityEstimator::makeVariant(5, 4), 4);
  BOOST_TEST(est.expected(key, 1000.) == 0.); // nothing learned yet

  est.update(key, 1000., 2000, 2600, false, false); // ratio of 2
  BOOST_TEST(est.expected(key, 1000.) == 2000.);
  BOOST_TEST(est.expected(key, 250.) == 500.); // a smaller timeframe predicts proportionally less
  BOOST_TEST(est.expected(key, 0.) == 0.);

  // ... while the all-time peak stays where it was, which is why it cannot size a shared buffer.
  BOOST_TEST(est.peakCapacity(key) >= 2000u);
}

BOOST_AUTO_TEST_CASE(estimator_ceiling_follows_real_growth)
{
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Cells, 0, 0, 1);
  constexpr double scale = 1000.;
  size_t need = 10000;

  for (int tf = 0; tf < 6; ++tf) {
    const size_t cap = est.capacity(key, scale);
    est.update(key, scale, need, cap, need > cap, false);
    need *= 2;
  }
  // Each timeframe doubled the output; the ceiling has to have followed, or every one of them
  // would have paid for a retry.
  BOOST_TEST(est.capacity(key, scale) >= need / 2);
}

BOOST_AUTO_TEST_CASE(estimator_backs_off_when_the_pool_refuses)
{
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 2, 0, 0);
  constexpr double scale = 1000.;
  constexpr double rate = 5.;
  const auto emitted = size_t(scale * rate);

  for (int tf = 0; tf < 12; ++tf) {
    const size_t cap = est.capacity(key, scale);
    est.update(key, scale, emitted, cap, emitted > cap, false);
  }
  const size_t settled = est.capacity(key, scale);

  for (int tf = 0; tf < 12; ++tf) {
    est.update(key, scale, emitted, 100, true, true);
  }
  BOOST_TEST(est.capacity(key, scale) < settled);
}

BOOST_AUTO_TEST_CASE(estimator_grows_in_proportion_to_the_miss)
{
  CapacityEstimator est;
  constexpr double scale = 1000.;
  const auto nearMiss = CapacityEstimator::makeKey(SlabSite::Cells, 3, 0, 0);
  const auto wayOff = CapacityEstimator::makeKey(SlabSite::Cells, 3, 0, 1);

  for (const auto key : {nearMiss, wayOff}) {
    est.update(key, scale, 2000, 2000, false, false);
  }
  const size_t settled = est.capacity(nearMiss, scale);

  est.update(nearMiss, scale, 2000, 1900, true, false); // overran by 5%
  est.update(wayOff, scale, 2000, 500, true, false);    // overran by 4x

  const size_t afterNearMiss = est.capacity(nearMiss, scale);
  const size_t afterWayOff = est.capacity(wayOff, scale);
  BOOST_TEST(afterNearMiss > settled);
  BOOST_TEST(afterNearMiss < afterWayOff);
  BOOST_TEST(afterNearMiss < size_t(1.25 * double(settled)));
  BOOST_TEST(afterWayOff > size_t(1.4 * double(settled)));
}

BOOST_AUTO_TEST_CASE(estimator_recovers_from_a_single_overflow)
{
  CapacityEstimator::Config cfg;
  cfg.decayAfter = 1;
  CapacityEstimator est{cfg};
  const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, 4, 0, 0);
  constexpr double scale = 1000.;

  est.update(key, scale, 2000, 2000, false, false);
  est.update(key, scale, 2000, 500, true, false);
  const size_t inflated = est.capacity(key, scale);

  for (int tf = 0; tf < 30; ++tf) {
    est.update(key, scale, 2000, 20000, false, false); // 10% utilisation
  }
  const size_t recovered = est.capacity(key, scale);
  BOOST_TEST(recovered < inflated);
  BOOST_TEST(recovered <= size_t(2. * scale * double(cfg.marginMin)) + 2);
}

BOOST_AUTO_TEST_CASE(estimator_decay_survives_interleaved_busy_timeframes)
{
  CapacityEstimator::Config cfg;
  cfg.decayAfter = 4;
  CapacityEstimator est{cfg};
  const auto key = CapacityEstimator::makeKey(SlabSite::Roads, 5, 0, 0);
  constexpr double scale = 1000.;

  est.update(key, scale, 2000, 2000, false, false);
  est.update(key, scale, 2000, 500, true, false);
  const size_t inflated = est.capacity(key, scale);

  for (int tf = 0; tf < 80; ++tf) {
    const bool quiet = (tf % 4) != 3;
    est.update(key, scale, 2000, quiet ? 20000 : 2000, false, false);
  }
  BOOST_TEST(est.capacity(key, scale) < inflated);
}

BOOST_AUTO_TEST_CASE(estimator_reset_forgets_inflated_margins)
{
  CapacityEstimator est;
  const auto key = CapacityEstimator::makeKey(SlabSite::Cells, 0, 0, 0);
  constexpr double scale = 1000.;

  for (int tf = 0; tf < 6; ++tf) {
    est.update(key, scale, size_t(scale * 5.), 10, true, false);
  }
  BOOST_TEST(est.capacity(key, scale) > 5000u);

  est.reset();
  BOOST_TEST(est.capacity(key, scale) == 1024u);
}

BOOST_AUTO_TEST_CASE(estimator_keys_separate_the_road_walk_steps)
{
  const auto a = CapacityEstimator::makeKey(SlabSite::Roads, 0, CapacityEstimator::makeVariant(6, 4), 1);
  const auto b = CapacityEstimator::makeKey(SlabSite::Roads, 0, CapacityEstimator::makeVariant(5, 4), 1);
  const auto c = CapacityEstimator::makeKey(SlabSite::Roads, 0, CapacityEstimator::makeVariant(6, 4), 2);
  BOOST_TEST(a != b);
  BOOST_TEST(a != c);
  BOOST_TEST(b != c);
}
