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
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

/// FIXME: do not use data model tables

namespace test
{
DECLARE_SOA_COLUMN_FULL(X, x, float, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, float, "y");
DECLARE_SOA_COLUMN_FULL(Z, z, float, "z");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](float x, float y) { return x + y; });
} // namespace test

#ifdef __APPLE__
constexpr unsigned int maxPairsRange = 10;
constexpr unsigned int maxFivesRange = 3;
#else
constexpr unsigned int maxPairsRange = 12;
constexpr unsigned int maxFivesRange = 3;
#endif

static void BM_ASoAHelpersEmptySimplePairs(benchmark::State& state)
{
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    int n = state.range(0);
    for (int i = 0; i < n - 1; i++) {
      for (int j = i + 1; j < n; j++) {
        count++;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersEmptySimplePairs)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersEmptySimpleFives(benchmark::State& state)
{
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    int n = state.range(0);
    for (int i = 0; i < n - 4; i++) {
      for (int j = i + 1; j < n - 3; j++) {
        for (int k = j + 1; k < n - 2; k++) {
          for (int l = k + 1; l < n - 1; l++) {
            for (int m = l + 1; m < n; m++) {
              count++;
            }
          }
        }
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersEmptySimpleFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersNaiveSimplePairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 + 1 != tests.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != tests.end(); ++t1) {
        auto comb = std::make_tuple(t0, t1);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersNaiveSimplePairs)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersNaiveSimpleFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 + 4 != tests.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != tests.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != tests.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != tests.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != tests.end(); ++t4) {
              auto comb = std::make_tuple(t0, t1, t2, t3, t4);
              count++;
              benchmark::DoNotOptimize(comb);
            }
          }
        }
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersNaiveSimpleFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersNaiveCalosPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Calos>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Calos Calos{table};
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = Calos.begin(); t0 + 1 != Calos.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != Calos.end(); ++t1) {
        auto comb = std::make_tuple(t0, t1);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersNaiveCalosPairs)->Range(8, 8 << (maxPairsRange - 3));

static void BM_ASoAHelpersNaiveCalosFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Calos>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Calos calos{table};
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = calos.begin(); t0 + 4 != calos.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != calos.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != calos.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != calos.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != calos.end(); ++t4) {
              auto comb = std::make_tuple(t0, t1, t2, t3, t4);
              count++;
              benchmark::DoNotOptimize(comb);
            }
          }
        }
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersNaiveCalosFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersNaiveTracksPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::StoredTracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), 0,
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::StoredTracks tracks{table};
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tracks.begin(); t0 + 1 != tracks.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != tracks.end(); ++t1) {
        auto comb = std::make_tuple(t0, t1);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersNaiveTracksPairs)->Range(8, 8 << (maxPairsRange - 3));

static void BM_ASoAHelpersNaiveTracksFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::StoredTracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), 0,
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::StoredTracks tracks{table};
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tracks.begin(); t0 + 4 != tracks.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != tracks.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != tracks.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != tracks.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != tracks.end(); ++t4) {
              auto comb = std::make_tuple(t0, t1, t2, t3, t4);
              count++;
              benchmark::DoNotOptimize(comb);
            }
          }
        }
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersNaiveTracksFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersCombGenSimplePairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(tests, tests)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimplePairs)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersCombGenSimpleFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(tests, tests, tests, tests, tests)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimpleFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersCombGenCalosPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Calos>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Calos calos{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(calos, calos)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCalosPairs)->Range(8, 8 << (maxPairsRange - 3));

static void BM_ASoAHelpersCombGenCalosFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Calos>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Calos calos{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(calos, calos, calos, calos, calos)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCalosFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersCombGenTracksPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::StoredTracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), 0,
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::StoredTracks tracks{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(tracks, tracks)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksPairs)->Range(8, 8 << (maxPairsRange - 3));

static void BM_ASoAHelpersCombGenTracksFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::StoredTracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), 0,
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::StoredTracks tracks{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(tracks, tracks, tracks, tracks, tracks)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersCombGenSimpleFivesMultipleChunks(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builderA;
  auto rowWriterA = builderA.persist<float, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterA(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto tableA = builderA.finalize();

  TableBuilder builderB;
  auto rowWriterB = builderB.persist<int32_t>({"x"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterB(0, uniform_dist(e1));
  }
  auto tableB = builderB.finalize();

  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  using TestB = o2::soa::Table<o2::soa::Index<>, test::X>;
  using ConcatTest = Concat<TestA, TestB>;

  ConcatTest tests{tableA, tableB};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(tests, tests, tests, tests, tests)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimpleFivesMultipleChunks)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange - 1));

static void BM_ASoAHelpersCombGenCalosFivesMultipleChunks(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builderA;
  auto rowWriterA = builderA.cursor<o2::aod::Calos>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterA(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto tableA = builderA.finalize();

  TableBuilder builderB;
  auto rowWriterB = builderB.cursor<o2::aod::Calos>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterB(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto tableB = builderB.finalize();

  using ConcatTest = Concat<o2::aod::Calos, o2::aod::Calos>;

  ConcatTest calos{tableA, tableB};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(calos, calos, calos, calos, calos)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCalosFivesMultipleChunks)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange - 1));

static void BM_ASoAHelpersCombGenSimplePairsSameCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};
  NoBinningPolicy<test::X> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperSameIndexPolicy(noBinning, 2, -1, tests, tests))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimplePairsSameCategories)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersCombGenSimpleFivesSameCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};
  NoBinningPolicy<test::X> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperSameIndexPolicy(noBinning, 5, -1, tests, tests, tests, tests, tests))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimpleFivesSameCategories)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange + 1));

static void BM_ASoAHelpersCombGenSimplePairsCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};
  NoBinningPolicy<test::X> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperIndexPolicy(noBinning, 2, -1, tests, tests))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimplePairsCategories)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersCombGenSimpleFivesCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder builder;
  auto rowWriter = builder.persist<int, float, float>({"x", "y", "z"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;
  Test tests{table};
  NoBinningPolicy<test::X> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperIndexPolicy(noBinning, 2, -1, tests, tests, tests, tests, tests))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenSimpleFivesCategories)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange + 1));

static void BM_ASoAHelpersCombGenCollisionsPairsSameCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist_int(e1), uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};
  NoBinningPolicy<o2::aod::collision::NumContrib> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperSameIndexPolicy(noBinning, 2, -1, collisions, collisions))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCollisionsPairsSameCategories)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersCombGenCollisionsFivesSameCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist_int(e1), uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};
  NoBinningPolicy<o2::aod::collision::NumContrib> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperSameIndexPolicy(noBinning, 5, -1, collisions, collisions, collisions, collisions, collisions))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCollisionsFivesSameCategories)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange + 1));

static void BM_ASoAHelpersCombGenCollisionsPairsCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 10);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist_int(e1), uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};
  NoBinningPolicy<o2::aod::collision::NumContrib> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperIndexPolicy(noBinning, 2, -1, collisions, collisions))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCollisionsPairsCategories)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersCombGenCollisionsFivesCategories(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist_int(e1), uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};
  NoBinningPolicy<o2::aod::collision::NumContrib> noBinning;

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations(CombinationsBlockUpperIndexPolicy(noBinning, 5, -1, collisions, collisions, collisions, collisions, collisions))) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCollisionsFivesCategories)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange + 1));

BENCHMARK_MAIN();
