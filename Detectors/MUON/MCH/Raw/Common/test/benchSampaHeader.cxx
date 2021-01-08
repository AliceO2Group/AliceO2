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
#include "MCHRawCommon/SampaHeader.h"

using namespace o2::mch::raw;

static void BM_ComputeHamming1(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHammingCode1(0x3722e80103208));
  }
}

static void BM_ComputeHamming2(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHammingCode2(0x3722e80103208));
  }
}

static void BM_ComputeHamming3(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHammingCode3(0x3722e80103208));
  }
}
static void BM_ComputeHamming4(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHammingCode4(0x3722e80103208));
  }
}

static void BM_ComputeHeaderParity1(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHeaderParity1(0x3722e80103208));
  }
}

static void BM_ComputeHeaderParity2(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHeaderParity2(0x3722e80103208));
  }
}

static void BM_ComputeHeaderParity3(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHeaderParity3(0x3722e80103208));
  }
}

static void BM_ComputeHeaderParity4(benchmark::State& state)
{
  for (auto _ : state) {
    benchmark::DoNotOptimize(computeHeaderParity4(0x3722e80103208));
  }
}

// Register the function as a benchmark
BENCHMARK(BM_ComputeHamming1);
BENCHMARK(BM_ComputeHamming2);
BENCHMARK(BM_ComputeHamming3);
BENCHMARK(BM_ComputeHamming4);

BENCHMARK(BM_ComputeHeaderParity1);
BENCHMARK(BM_ComputeHeaderParity2);
BENCHMARK(BM_ComputeHeaderParity3);
BENCHMARK(BM_ComputeHeaderParity4);

// Run the benchmark
BENCHMARK_MAIN();
