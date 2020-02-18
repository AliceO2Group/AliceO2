// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <benchmark/benchmark.h>
#include <bitset>
#include <array>
#include "BitSet.h"
#include <boost/multiprecision/cpp_int.hpp>

constexpr uint64_t one{1};

using namespace o2::mch::raw;

static void BM_BitSetGet(benchmark::State& state)
{
  BitSet bs;
  bs.grow(10024);
  for (auto _ : state) {
    for (int i = 0; i < 100; i++) {
      bs.get(i);
    }
  }
}

static void BM_BitSetSet(benchmark::State& state)
{
  BitSet bs;
  bs.grow(10024);
  int a{0};
  for (auto _ : state) {
    for (int i = 0; i < 100; i++) {
      bs.set(i, false);
    }
  }
}

static void BM_BitSetSetFast(benchmark::State& state)
{
  BitSet bs;
  bs.grow(10024);
  for (auto _ : state) {
    for (int i = 0; i < 100; i++) {
      bs.setFast(i, false);
    }
  }
}

using namespace boost::multiprecision;
static void BM_Set(benchmark::State& state)
{
  typedef number<cpp_int_backend<16384, 16384, unsigned_magnitude, unchecked, void>> bigint;
  bigint bs;
  for (auto _ : state) {
    for (int i = 0; i < 100; i++) {
      bit_unset(bs, i);
    }
  }
}

using namespace boost::multiprecision;
static void BM_Get(benchmark::State& state)
{
  typedef number<cpp_int_backend<16384, 16384, unsigned_magnitude, unchecked, void>> bigint;
  bigint bs;
  bool a;
  for (auto _ : state) {
    for (int i = 0; i < 100; i++) {
      benchmark::DoNotOptimize(bit_test(bs, i));
    }
  }
}

// Register the function as a benchmark
BENCHMARK(BM_BitSetSet);
BENCHMARK(BM_BitSetGet);
BENCHMARK(BM_BitSetSetFast);
BENCHMARK(BM_Set);
BENCHMARK(BM_Get);

// Run the benchmark
BENCHMARK_MAIN();
