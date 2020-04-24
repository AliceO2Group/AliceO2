// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

static void BM_ASoAHelpersNaiveTracksPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
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

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
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

static void BM_ASoAHelpersCombGenTracksPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

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

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

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

static void BM_ASoAHelpersCombGenTracksFivesMultipleChunks(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builderA;
  auto rowWriterA = builderA.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterA(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto tableA = builderA.finalize();

  TableBuilder builderB;
  auto rowWriterB = builderB.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterB(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto tableB = builderB.finalize();

  using ConcatTest = Concat<o2::aod::Tracks, o2::aod::Tracks>;

  ConcatTest tracks{tableA, tableB};

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

BENCHMARK(BM_ASoAHelpersCombGenTracksFivesMultipleChunks)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange - 1));

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

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : selfCombinations("x", tests, tests)) {
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

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : selfCombinations("x", tests, tests, tests, tests, tests)) {
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

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations("x", tests, tests)) {
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

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations("x", tests, tests, tests, tests, tests)) {
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
              uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist_int(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : selfCombinations("fNumContrib", collisions, collisions)) {
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
              uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist_int(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : selfCombinations("fNumContrib", collisions, collisions, collisions, collisions, collisions)) {
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
              uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist_int(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations("fNumContrib", collisions, collisions)) {
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
              uniform_dist(e1),
              uniform_dist_int(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist_int(e1));
  }
  auto table = builder.finalize();

  o2::aod::Collisions collisions{table};

  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& comb : combinations("fNumContrib", collisions, collisions, collisions, collisions, collisions)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenCollisionsFivesCategories)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange + 1));

static void BM_ASoAHelpersCombGenSimplePairsFilters(benchmark::State& state)
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
  expressions::Filter filter = test::x > (test::x * (-1.0f) + 1.0f);
  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1] : selfCombinations(filter, tests, tests)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  int64_t filteringCount = state.range(0) * state.range(0);
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenSimplePairsFilters)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersCombGenSimpleFivesFilters(benchmark::State& state)
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
  expressions::Filter filter = (test::x > 0.3f) && (test::x > (test::x * (-1.0f) + 1.0f)) && (test::x > (test::x * (-1.0f) + 1.0f));
  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1, t2, t3, t4] : selfCombinations(filter, tests, tests, tests, tests, tests)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  int64_t filteringCount = state.range(0) * state.range(0) * state.range(0) * state.range(0) * state.range(0);
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenSimpleFivesFilters)->RangeMultiplier(2)->Range(8, 16);

static void BM_ASoAHelpersCombGenTracksPairsFilters(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int64_t count = 0;
  expressions::Filter filter = o2::aod::track::x > (o2::aod::track::x * (-1.0f) + 1.0f);
  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1] : selfCombinations(filter, tracks, tracks)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  int64_t filteringCount = state.range(0) * state.range(0);
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksPairsFilters)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersCombGenTracksFivesFilters(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int64_t count = 0;
  expressions::Filter filter = (o2::aod::track::x > 0.3f) && (o2::aod::track::x > (o2::aod::track::x * (-1.0f) + 1.0f)) && (o2::aod::track::x > (o2::aod::track::x * (-1.0f) + 1.0f));
  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1, t2, t3, t4] : selfCombinations(filter, tracks, tracks, tracks, tracks, tracks)) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  int64_t filteringCount = state.range(0) * state.range(0) * state.range(0) * state.range(0) * state.range(0);
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksFivesFilters)->RangeMultiplier(2)->Range(8, 16);

static void BM_ASoAHelpersCombGenSimplePairsIfs(benchmark::State& state)
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
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1] : combinations(tests, tests)) {
      filteringCount++;
      if (t0.x() > (t1.x() * (-1.0f) + 1.0f)) {
        auto comb = std::make_tuple(t0, t1);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenSimplePairsIfs)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersCombGenSimpleFivesIfs(benchmark::State& state)
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
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1, t2, t3, t4] : combinations(tests, tests, tests, tests, tests)) {
      filteringCount++;
      if ((t0.x() > 0.3f) && (t2.x() > (t1.x() * (-1.0f) + 1.0f)) && (t4.x() > (t3.x() * (-1.0f) + 1.0f))) {
        auto comb = std::make_tuple(t0, t1, t2, t3, t4);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenSimpleFivesIfs)->RangeMultiplier(2)->Range(8, 16);

static void BM_ASoAHelpersCombGenTracksPairsIfs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int64_t count = 0;
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1] : combinations(tracks, tracks)) {
      filteringCount++;
      if (t1.x() > (t0.x() * (-1.0f) + 1.0f)) {
        auto comb = std::make_tuple(t0, t1);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksPairsIfs)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersCombGenTracksFivesIfs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int64_t count = 0;
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto& [t0, t1, t2, t3, t4] : combinations(tracks, tracks, tracks, tracks, tracks)) {
      filteringCount++;
      if ((t0.x() > 0.3f) && (t2.x() > (t1.x() * (-1.0f) + 1.0f)) && (t4.x() > (t3.x() * (-1.0f) + 1.0f))) {
        auto comb = std::make_tuple(t0, t1, t2, t3, t4);
        count++;
        benchmark::DoNotOptimize(comb);
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksFivesIfs)->RangeMultiplier(2)->Range(8, 16);

static void BM_ASoAHelpersNaiveSimplePairsIfs(benchmark::State& state)
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
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 + 1 != tests.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != tests.end(); ++t1) {
        filteringCount++;
        if ((*t1).x() > ((*t0).x() * (-1.0f) + 1.0f)) {
          auto comb = std::make_tuple(t0, t1);
          count++;
          benchmark::DoNotOptimize(comb);
        }
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveSimplePairsIfs)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersNaiveSimpleFivesIfs(benchmark::State& state)
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
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 + 4 != tests.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != tests.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != tests.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != tests.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != tests.end(); ++t4) {
              filteringCount++;
              if (((*t0).x() > 0.3f) && ((*t2).x() > ((*t1).x() * (-1.0f) + 1.0f)) && ((*t4).x() > ((*t3).x() * (-1.0f) + 1.0f))) {
                auto comb = std::make_tuple(t0, t1, t2, t3, t4);
                count++;
                benchmark::DoNotOptimize(comb);
              }
            }
          }
        }
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveSimpleFivesIfs)->RangeMultiplier(2)->Range(8, 16);

static void BM_ASoAHelpersNaiveTracksPairsIfs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int64_t count = 0;
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tracks.begin(); t0 + 1 != tracks.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != tracks.end(); ++t1) {
        filteringCount++;
        if ((*t1).x() > ((*t0).x() * (-1.0f) + 1.0f)) {
          auto comb = std::make_tuple(t0, t1);
          count++;
          benchmark::DoNotOptimize(comb);
        }
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveTracksPairsIfs)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersNaiveTracksFivesIfs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int64_t count = 0;
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tracks.begin(); t0 + 4 != tracks.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != tracks.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != tracks.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != tracks.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != tracks.end(); ++t4) {
              filteringCount++;
              if (((*t0).x() > 0.3f) && ((*t2).x() > ((*t1).x() * (-1.0f) + 1.0f)) && ((*t4).x() > ((*t3).x() * (-1.0f) + 1.0f))) {
                auto comb = std::make_tuple(t0, t1, t2, t3, t4);
                count++;
                benchmark::DoNotOptimize(comb);
              }
            }
          }
        }
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveTracksFivesIfs)->RangeMultiplier(2)->Range(8, 16);

static void BM_ASoAHelpersNaiveSimplePairsFilters(benchmark::State& state)
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
    o2::framework::expressions::Filter filter = test::x > (test::x * (-1.0f) + 1.0f);
    o2::framework::expressions::Operations operations = createOperations(filter);
    int i = 0;
    for (; i < operations.size() && operations[i].left.datum.index() != 3; i++)
      ;

    for (int j = 0; j < tests.size(); j++) {
      setColumnValue(tests, "x", operations[i].left, j);
      o2::framework::expressions::Selection selection = o2::framework::expressions::createSelection(tests.asArrowTable(), createFilter(tests.asArrowTable()->schema(), operations));
      // Find first index bigger than the index of 1st iterator
      int beginSelectionIndex = 0;
      for (; beginSelectionIndex < selection->GetNumSlots() &&
             selection->GetIndex(beginSelectionIndex) <= j;
           beginSelectionIndex++) {
        ;
      }
      if (beginSelectionIndex != selection->GetNumSlots()) {
        Filtered<Test> filtered{{tests.asArrowTable()}, selection};
        for (auto t0 = filtered.begin() + beginSelectionIndex; t0 != filtered.end(); ++t0) {
          count++;
        }
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  int64_t filteringCount = state.range(0) * state.range(0);
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveSimplePairsFilters)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersNaiveFullSimplePairsIfs(benchmark::State& state)
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
  int64_t filteringCount = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 != tests.end(); ++t0) {
      for (auto t1 = tests.begin(); t1 != tests.end(); ++t1) {
        filteringCount++;
        if ((*t1).x() > ((*t0).x() * (-1.0f) + 1.0f)) {
          auto comb = std::make_tuple(t0, t1);
          count++;
          benchmark::DoNotOptimize(comb);
        }
      }
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(filteringCount);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveFullSimplePairsIfs)->RangeMultiplier(2)->Range(8, 8 << 6);

static void BM_ASoAHelpersNaiveFullSimplePairsFilters(benchmark::State& state)
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
    o2::framework::expressions::Filter filter = test::x > (test::x * (-1.0f) + 1.0f);
    o2::framework::expressions::Operations operations = createOperations(filter);
    int i = 0;
    for (; i < operations.size() && operations[i].left.datum.index() != 3; i++)
      ;
    for (int j = 0; j < tests.size(); j++) {
      setColumnValue(tests, "x", operations[i].left, j);
      o2::framework::expressions::Selection selection = o2::framework::expressions::createSelection(tests.asArrowTable(), createFilter(tests.asArrowTable()->schema(), operations));
      Filtered<Test> filtered{{tests.asArrowTable()}, selection};
      for (auto t0 = filtered.begin(); t0 != filtered.end(); ++t0) {
        count++;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  int64_t filteringCount = state.range(0) * state.range(0);
  state.SetBytesProcessed(state.iterations() * sizeof(float) * filteringCount);
}

BENCHMARK(BM_ASoAHelpersNaiveFullSimplePairsFilters)->RangeMultiplier(2)->Range(8, 8 << 6);

BENCHMARK_MAIN();
