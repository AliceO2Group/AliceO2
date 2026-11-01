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

#include "AODNative/ReadPlan.h"
#include <catch_amalgamated.hpp>
#include <cstring>
#include <thread>

using namespace o2::aod::native;

namespace
{
// File contents where byte i is i % 251, recording every read.
struct RecordingSource : ByteSource {
  explicit RecordingSource(size_t n) : bytes(n)
  {
    for (size_t i = 0; i < n; ++i) {
      bytes[i] = i % 251;
    }
  }
  uint64_t size() const override { return bytes.size(); }
  void readInto(uint64_t off, size_t len, void* dst) override
  {
    reads.emplace_back(off, len);
    std::memcpy(dst, bytes.data() + off, len);
  }
  void readvInto(std::span<ReadRequest> requests) override
  {
    for (auto& r : requests) {
      readInto(r.offset, r.length, r.dst);
    }
  }
  uint64_t bytesRead() const override { return 0; }
  uint64_t readCalls() const override { return reads.size(); }
  std::string url() const override { return "memory"; }

  std::vector<uint8_t> bytes;
  std::vector<std::pair<uint64_t, size_t>> reads;
};

bool holds(std::span<const uint8_t> span, uint64_t offset)
{
  for (size_t i = 0; i < span.size(); ++i) {
    if (span[i] != (offset + i) % 251) {
      return false;
    }
  }
  return true;
}

Options withGap(size_t gap)
{
  Options options;
  options.coalesceGap = gap;
  return options;
}
} // namespace

TEST_CASE("ranges closer than the gap are fetched in one read, on first use")
{
  auto source = std::make_shared<RecordingSource>(1 << 20);
  ReadPlan plan{source, withGap(100)};
  plan.add(5000, 200);
  plan.add(1000, 100);
  plan.add(1150, 50); // 50 bytes after the first: merged
  plan.add(1300, 10); // 100 bytes after the second: merged
  REQUIRE(source->reads.empty());

  auto bytes = plan.find(1150, 50);
  REQUIRE(holds(bytes, 1150));
  REQUIRE(source->reads == std::vector<std::pair<uint64_t, size_t>>{{1000, 310}, {5000, 200}});
  REQUIRE(plan.bytesHeld() == 510);

  REQUIRE(holds(plan.find(5100, 100), 5100));
  REQUIRE(holds(plan.find(1000, 310), 1000));
  REQUIRE(source->reads.size() == 2);
}

TEST_CASE("ranges which were never registered are not found")
{
  auto source = std::make_shared<RecordingSource>(1 << 20);
  ReadPlan plan{source, withGap(0)};
  plan.add(1000, 100);
  REQUIRE(plan.find(900, 50).empty());
  REQUIRE(plan.find(1050, 100).empty()); // runs past the fetched range
  REQUIRE(plan.find(5000, 1).empty());
  REQUIRE(source->reads.size() == 1);
}

TEST_CASE("ranges added after a fetch are fetched on their first use")
{
  auto source = std::make_shared<RecordingSource>(1 << 20);
  ReadPlan plan{source, withGap(1000)};
  plan.add(1000, 100);
  REQUIRE(holds(plan.find(1000, 100), 1000));
  auto first = plan.find(1000, 100);

  // Within the gap of the fetched chunk on both sides, but not merged across it.
  plan.add(500, 100);
  plan.add(1500, 100);
  plan.add(1000, 100); // already in memory: not read again
  REQUIRE(holds(plan.find(1500, 100), 1500));
  REQUIRE(holds(plan.find(500, 100), 500));
  REQUIRE(source->reads == std::vector<std::pair<uint64_t, size_t>>{{1000, 100}, {500, 100}, {1500, 100}});
  // Spans handed out before stay valid.
  REQUIRE(plan.find(1000, 100).data() == first.data());
  REQUIRE(holds(first, 1000));
}

TEST_CASE("ranges past the end of the file are left to the reader")
{
  auto source = std::make_shared<RecordingSource>(1000);
  ReadPlan plan{source, withGap(0)};
  plan.add(900, 200);
  plan.add(2000, 1);
  REQUIRE(plan.find(900, 100).empty());
  REQUIRE(source->reads.empty());
}

TEST_CASE("a planned source serves planned ranges from memory and the rest from the source")
{
  auto source = std::make_shared<RecordingSource>(1 << 20);
  auto plan = std::make_shared<ReadPlan>(source, withGap(0));
  plan->add(1000, 100);
  PlannedByteSource planned{source, plan};

  REQUIRE(holds(planned.tryMap(1010, 20), 1010));
  std::vector<uint8_t> out(30);
  planned.readInto(1020, 30, out.data());
  REQUIRE(holds(out, 1020));
  REQUIRE(planned.tryMap(3000, 10).empty());
  planned.readInto(3000, 30, out.data());
  REQUIRE(holds(out, 3000));
  REQUIRE(source->reads == std::vector<std::pair<uint64_t, size_t>>{{1000, 100}, {3000, 30}});
  REQUIRE(planned.readCalls() == 2);
}

TEST_CASE("concurrent lookups fetch the pending ranges once")
{
  auto source = std::make_shared<RecordingSource>(1 << 20);
  ReadPlan plan{source, withGap(0)};
  for (int i = 0; i < 64; ++i) {
    plan.add(i * 10000, 1000);
  }
  std::vector<std::thread> threads;
  std::atomic<int> good = 0;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = t; i < 64; i += 8) {
        good += holds(plan.find(i * 10000 + 10, 900), i * 10000 + 10);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  REQUIRE(good == 64);
  REQUIRE(source->reads.size() == 64);
}

TEST_CASE("buffers of finished plans are reused by the next one")
{
  auto source = std::make_shared<RecordingSource>(1 << 20);
  auto pool = std::make_shared<ReadBufferPool>();
  uint8_t const* first = nullptr;
  {
    ReadPlan plan{source, withGap(0), pool};
    plan.add(1000, 1000);
    first = plan.find(1000, 1000).data();
  }
  ReadPlan next{source, withGap(0), pool};
  next.add(50000, 900); // fits in the first buffer
  auto bytes = next.find(50000, 900);
  REQUIRE(bytes.data() == first);
  REQUIRE(holds(bytes, 50000));

  auto big = pool->get(1 << 16);
  REQUIRE(big.capacity >= 1 << 16);
}
