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

#include "Framework/BigEndian.h"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace o2::framework;

static void BM_BigEndianCopyUInt16(benchmark::State& state)
{
  auto const bytes = static_cast<size_t>(state.range(0));
  int const count = bytes / sizeof(uint16_t);
  std::vector<uint16_t> src(count, 0xCAFE);
  auto* dest = static_cast<uint16_t*>(std::aligned_alloc(64, bytes));
  for (auto _ : state) {
    bigEndianCopy(dest, src.data(), count, sizeof(uint16_t));
    benchmark::DoNotOptimize(dest);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
  std::free(dest);
}
BENCHMARK(BM_BigEndianCopyUInt16)->RangeMultiplier(2)->Range(32000, 512000);

static void BM_BigEndianCopyUInt32(benchmark::State& state)
{
  auto const bytes = static_cast<size_t>(state.range(0));
  int const count = bytes / sizeof(uint32_t);
  std::vector<uint32_t> src(count, 0xDEADBEEF);
  auto* dest = static_cast<uint32_t*>(std::aligned_alloc(64, bytes));
  for (auto _ : state) {
    bigEndianCopy(dest, src.data(), count, sizeof(uint32_t));
    benchmark::DoNotOptimize(dest);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
  std::free(dest);
}
BENCHMARK(BM_BigEndianCopyUInt32)->RangeMultiplier(2)->Range(32000, 512000);

static void BM_BigEndianCopyUInt64(benchmark::State& state)
{
  auto const bytes = static_cast<size_t>(state.range(0));
  int const count = bytes / sizeof(uint64_t);
  std::vector<uint64_t> src(count, 0x0123456789ABCDEFULL);
  auto* dest = static_cast<uint64_t*>(std::aligned_alloc(64, bytes));
  for (auto _ : state) {
    bigEndianCopy(dest, src.data(), count, sizeof(uint64_t));
    benchmark::DoNotOptimize(dest);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
  std::free(dest);
}
BENCHMARK(BM_BigEndianCopyUInt64)->RangeMultiplier(2)->Range(32000, 512000);

BENCHMARK_MAIN();
