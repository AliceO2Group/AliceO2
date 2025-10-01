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

#include "Framework/ASoAHelpers.h"
#include "Framework/GroupedCombinations.h"
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <list>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

// Validation of new event mixing in detail

#ifdef __APPLE__
constexpr unsigned int maxColPairsRange = 20;
#else
constexpr unsigned int maxColPairsRange = 20;
#endif
constexpr int numEventsToMix = 5;

using namespace o2::framework;
using namespace o2::soa;

static void BM_EventMixingTableCreation(benchmark::State& state)
{
  for (auto _ : state) {
    // Seed with a real random value, if available
    std::default_random_engine e1(1234567891);
    std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
    std::uniform_real_distribution<float> uniform_dist_x(-0.065f, 0.073f);
    std::uniform_real_distribution<float> uniform_dist_y(-0.320f, 0.360f);
    std::uniform_int_distribution<int> uniform_dist_int(0, 5);

    TableBuilder colBuilder;
    auto rowWriterCol = colBuilder.cursor<o2::aod::Collisions>();
    for (auto i = 0; i < state.range(0); ++i) {
      float x = uniform_dist_x(e1);
      float y = uniform_dist_y(e1);
      rowWriterCol(0, uniform_dist_int(e1),
                   x, y, uniform_dist(e1),
                   uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                   uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                   uniform_dist_int(e1), uniform_dist(e1),
                   uniform_dist_int(e1),
                   uniform_dist(e1), uniform_dist(e1));
    }
    auto tableCol = colBuilder.finalize();
    o2::aod::Collisions collisions{tableCol};
  }
  state.SetBytesProcessed(state.iterations() * (12 * sizeof(float) + sizeof(int64_t) + 3 * sizeof(int)) * state.range(0));
}

BENCHMARK(BM_EventMixingTableCreation)->RangeMultiplier(2)->Range(4UL, 2UL << maxColPairsRange);

static void BM_EventMixingBinningCreation(benchmark::State& state)
{
  std::vector<double> xBins{VARIABLE_WIDTH, -0.064, -0.062, -0.060, 0.066, 0.068, 0.070, 0.072};
  std::vector<double> yBins{VARIABLE_WIDTH, -0.320, -0.301, -0.300, 0.330, 0.340, 0.350, 0.360};

  for (auto _ : state) {
    using BinningType = ColumnBinningPolicy<o2::aod::collision::PosX, o2::aod::collision::PosY>;
    BinningType binningOnPositions{{xBins, yBins}, true}; // true is for 'ignore overflows' (true by default)
  }
  state.SetBytesProcessed(state.iterations() * sizeof(float));
}

BENCHMARK(BM_EventMixingBinningCreation)->RangeMultiplier(2)->Range(4UL, 2UL << maxColPairsRange);

static void BM_EventMixingPolicyCreation(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
  std::uniform_real_distribution<float> uniform_dist_x(-0.065f, 0.073f);
  std::uniform_real_distribution<float> uniform_dist_y(-0.320f, 0.360f);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder colBuilder;
  auto rowWriterCol = colBuilder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    float x = uniform_dist_x(e1);
    float y = uniform_dist_y(e1);
    rowWriterCol(0, uniform_dist_int(e1),
                 x, y, uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist_int(e1), uniform_dist(e1),
                 uniform_dist_int(e1),
                 uniform_dist(e1), uniform_dist(e1));
  }
  auto tableCol = colBuilder.finalize();
  o2::aod::Collisions collisions{tableCol};

  std::vector<double> xBins{VARIABLE_WIDTH, -0.064, -0.062, -0.060, 0.066, 0.068, 0.070, 0.072};
  std::vector<double> yBins{VARIABLE_WIDTH, -0.320, -0.301, -0.300, 0.330, 0.340, 0.350, 0.360};
  using BinningType = ColumnBinningPolicy<o2::aod::collision::PosX, o2::aod::collision::PosY>;
  BinningType binningOnPositions{{xBins, yBins}, true}; // true is for 'ignore overflows' (true by default)

  for (auto _ : state) {
    auto combPolicy = CombinationsBlockUpperSameIndexPolicy(binningOnPositions, numEventsToMix - 1, -1, collisions, collisions);
  }
  state.SetBytesProcessed(state.iterations() * sizeof(float) * state.range(0));
}

BENCHMARK(BM_EventMixingPolicyCreation)->RangeMultiplier(2)->Range(4UL, 2UL << maxColPairsRange);

static void BM_EventMixingCombinationsCreation(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
  std::uniform_real_distribution<float> uniform_dist_x(-0.065f, 0.073f);
  std::uniform_real_distribution<float> uniform_dist_y(-0.320f, 0.360f);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder colBuilder;
  auto rowWriterCol = colBuilder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    float x = uniform_dist_x(e1);
    float y = uniform_dist_y(e1);
    rowWriterCol(0, uniform_dist_int(e1),
                 x, y, uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist_int(e1), uniform_dist(e1),
                 uniform_dist_int(e1),
                 uniform_dist(e1), uniform_dist(e1));
  }
  auto tableCol = colBuilder.finalize();
  o2::aod::Collisions collisions{tableCol};

  std::vector<double> xBins{VARIABLE_WIDTH, -0.064, -0.062, -0.060, 0.066, 0.068, 0.070, 0.072};
  std::vector<double> yBins{VARIABLE_WIDTH, -0.320, -0.301, -0.300, 0.330, 0.340, 0.350, 0.360};
  using BinningType = ColumnBinningPolicy<o2::aod::collision::PosX, o2::aod::collision::PosY>;
  BinningType binningOnPositions{{xBins, yBins}, true}; // true is for 'ignore overflows' (true by default)

  auto combPolicy = CombinationsBlockUpperSameIndexPolicy(binningOnPositions, numEventsToMix - 1, -1, collisions, collisions);

  for (auto _ : state) {
    auto comb = combinations(combPolicy);
  }
  state.SetBytesProcessed(state.iterations() * sizeof(float));
}

BENCHMARK(BM_EventMixingCombinationsCreation)->RangeMultiplier(2)->Range(4UL, 2UL << maxColPairsRange);

static void BM_EventMixingCombinations(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
  std::uniform_real_distribution<float> uniform_dist_x(-0.065f, 0.073f);
  std::uniform_real_distribution<float> uniform_dist_y(-0.320f, 0.360f);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder colBuilder;
  auto rowWriterCol = colBuilder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    float x = uniform_dist_x(e1);
    float y = uniform_dist_y(e1);
    rowWriterCol(0, uniform_dist_int(e1),
                 x, y, uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist_int(e1), uniform_dist(e1),
                 uniform_dist_int(e1),
                 uniform_dist(e1), uniform_dist(e1));
  }
  auto tableCol = colBuilder.finalize();
  o2::aod::Collisions collisions{tableCol};

  std::vector<double> xBins{VARIABLE_WIDTH, -0.064, -0.062, -0.060, 0.066, 0.068, 0.070, 0.072};
  std::vector<double> yBins{VARIABLE_WIDTH, -0.320, -0.301, -0.300, 0.330, 0.340, 0.350, 0.360};
  using BinningType = ColumnBinningPolicy<o2::aod::collision::PosX, o2::aod::collision::PosY>;
  BinningType binningOnPositions{{xBins, yBins}, true}; // true is for 'ignore overflows' (true by default)

  auto combPolicy = CombinationsBlockUpperSameIndexPolicy(binningOnPositions, numEventsToMix - 1, -1, collisions, collisions);

  auto comb = combinations(combPolicy);

  int64_t colCount = 0;

  for (auto _ : state) {
    colCount = 0;
    for (auto& combT : comb) {
      colCount++;
    }
    benchmark::DoNotOptimize(colCount);
  }
  state.counters["Mixed collision pairs"] = colCount;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * colCount);
}

BENCHMARK(BM_EventMixingCombinations)->RangeMultiplier(2)->Range(4UL, 2UL << maxColPairsRange);

BENCHMARK_MAIN();
