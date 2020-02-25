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
DECLARE_SOA_COLUMN(X, x, float, "x");
DECLARE_SOA_COLUMN(Y, y, float, "y");
DECLARE_SOA_COLUMN(Z, z, float, "z");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](float x, float y) { return x + y; });
} // namespace test

#ifdef __APPLE__
constexpr unsigned int maxPairsRange = 10;
constexpr unsigned int maxFivesRange = 3;
#else
constexpr unsigned int maxPairsRange = 12;
constexpr unsigned int maxFivesRange = 3;
#endif

// Helper to reset the iterators for each benchmark loop
template <typename T, size_t K>
void resetCombination(std::array<T, K>& array, const T& maxOffset, bool& isEnd,
                      const size_t n, const std::function<bool(const std::array<T, K>&)>& condition)
{
  for (int i = 0; i < K; i++) {
    array[i].setCursor(i);
  }
  isEnd = n == K;
  while (!isEnd && !condition(array)) {
    addOne(array, maxOffset, isEnd);
  }
}

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

BENCHMARK(BM_ASoAHelpersEmptySimpleFives)->Range(8, 8 << maxFivesRange);

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
  std::function<bool(const std::array<Test::iterator, 2>&)> condition = [](const auto& testCombination) { return true; };
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 + 1 != tests.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != tests.end(); ++t1) {
        std::array<Test::iterator, 2> comb{t0, t1};
        if (condition(comb)) {
          count++;
        }
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
  std::function<bool(const std::array<Test::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tests.begin(); t0 + 4 != tests.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != tests.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != tests.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != tests.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != tests.end(); ++t4) {
              std::array<Test::iterator, 5> comb{t0, t1, t2, t3, t4};
              if (condition(comb)) {
                count++;
              }
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
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
  std::function<bool(const std::array<o2::aod::Tracks::iterator, 2>&)> condition = [](const auto& testCombination) { return true; };
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tracks.begin(); t0 + 1 != tracks.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 != tracks.end(); ++t1) {
        std::array<o2::aod::Tracks::iterator, 2> comb{t0, t1};
        if (condition(comb)) {
          count++;
        }
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
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
  std::function<bool(const std::array<o2::aod::Tracks::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto t0 = tracks.begin(); t0 + 4 != tracks.end(); ++t0) {
      for (auto t1 = t0 + 1; t1 + 3 != tracks.end(); ++t1) {
        for (auto t2 = t1 + 1; t2 + 2 != tracks.end(); ++t2) {
          for (auto t3 = t2 + 1; t3 + 1 != tracks.end(); ++t3) {
            for (auto t4 = t3 + 1; t4 != tracks.end(); ++t4) {
              std::array<o2::aod::Tracks::iterator, 5> comb{t0, t1, t2, t3, t4};
              if (condition(comb)) {
                count++;
              }
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

static void BM_ASoAHelpersAddOneSimplePairs(benchmark::State& state)
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

  std::array<Test::iterator, 2> comb2{tests.begin(), tests.begin() + 1};
  bool isEnd = false;
  int64_t count = 0;
  Test::iterator maxOffset = tests.begin() + tests.size() - 2 + 1;
  std::function<bool(const std::array<Test::iterator, 2>&)> condition = [](const auto& testCombination) { return true; };

  for (auto _ : state) {
    count = 0;
    resetCombination(comb2, maxOffset, isEnd, tests.size(), condition);
    count++;
    while (!isEnd) {
      addOne(comb2, maxOffset, isEnd);
      while (!isEnd && !condition(comb2)) {
        addOne(comb2, maxOffset, isEnd);
      }
      if (!isEnd) {
        count++;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersAddOneSimplePairs)->Range(8, 8 << maxPairsRange);

static void BM_ASoAHelpersAddOneSimpleFives(benchmark::State& state)
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

  int k = 5;
  std::array<Test::iterator, 5> comb5{tests.begin(), tests.begin() + 1, tests.begin() + 2, tests.begin() + 3, tests.begin() + 4};
  bool isEnd = false;
  int64_t count = 0;
  auto maxOffset = tests.begin() + tests.size() - 5 + 1;
  std::function<bool(const std::array<Test::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };

  for (auto _ : state) {
    count = 0;
    resetCombination(comb5, maxOffset, isEnd, tests.size(), condition);
    count++;
    while (!isEnd) {
      addOne(comb5, maxOffset, isEnd);
      while (!isEnd && !condition(comb5)) {
        addOne(comb5, maxOffset, isEnd);
      }
      if (!isEnd) {
        count++;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersAddOneSimpleFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

static void BM_ASoAHelpersAddOneTracksPairs(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int k = 2;
  std::array<o2::aod::Tracks::iterator, 2> comb2{tracks.begin(), tracks.begin() + 1};
  bool isEnd = false;
  int64_t count = 0;
  auto maxOffset = tracks.begin() + tracks.size() - 2 + 1;
  std::function<bool(const std::array<o2::aod::Tracks::iterator, 2>&)> condition = [](const auto& testCombination) { return true; };

  for (auto _ : state) {
    count = 0;
    resetCombination(comb2, maxOffset, isEnd, tracks.size(), condition);
    count++;
    while (!isEnd) {
      addOne(comb2, maxOffset, isEnd);
      while (!isEnd && !condition(comb2)) {
        addOne(comb2, maxOffset, isEnd);
      }
      if (!isEnd) {
        count++;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersAddOneTracksPairs)->Range(8, 8 << (maxPairsRange - 3));

static void BM_ASoAHelpersAddOneTracksFives(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};

  int k = 5;
  std::array<o2::aod::Tracks::iterator, 5> comb5{tracks.begin(), tracks.begin() + 1, tracks.begin() + 2, tracks.begin() + 3, tracks.begin() + 4};
  bool isEnd = false;
  int64_t count = 0;
  auto maxOffset = tracks.begin() + tracks.size() - 5 + 1;
  std::function<bool(const std::array<o2::aod::Tracks::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };

  for (auto _ : state) {
    count = 0;
    resetCombination(comb5, maxOffset, isEnd, tracks.size(), condition);
    count++;
    while (!isEnd) {
      addOne(comb5, maxOffset, isEnd);
      while (!isEnd && !condition(comb5)) {
        addOne(comb5, maxOffset, isEnd);
      }
      if (!isEnd) {
        count++;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersAddOneTracksFives)->RangeMultiplier(2)->Range(8, 8 << maxFivesRange);

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
  std::function<bool(const std::array<Test::iterator, 2>&)> condition = [](const auto& testCombination) { return true; };

  auto comb2 = CombinationsGenerator<Test, 2>(tests, condition);
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto comb : comb2) {
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
  std::function<bool(const std::array<Test::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };

  auto comb5 = CombinationsGenerator<Test, 5>(tests, condition);
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto comb : comb5) {
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
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
  std::function<bool(const std::array<o2::aod::Tracks::iterator, 2>&)> condition = [](const auto& testCombination) { return true; };

  auto comb2 = CombinationsGenerator<o2::aod::Tracks, 2>(tracks, condition);
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto comb : comb2) {
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
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
  std::function<bool(const std::array<o2::aod::Tracks::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };

  auto comb5 = CombinationsGenerator<o2::aod::Tracks, 5>(tracks, condition);
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto comb : comb5) {
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
  std::function<bool(const std::array<ConcatTest::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };

  auto comb5 = CombinationsGenerator<ConcatTest, 5>(tests, condition);
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto comb : comb5) {
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
               uniform_dist(e1), uniform_dist(e1));
  }
  auto tableA = builderA.finalize();

  TableBuilder builderB;
  auto rowWriterB = builderB.cursor<o2::aod::Tracks>();
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriterB(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
               uniform_dist(e1), uniform_dist(e1));
  }
  auto tableB = builderB.finalize();

  using ConcatTest = Concat<o2::aod::Tracks, o2::aod::Tracks>;

  ConcatTest tracks{tableA, tableB};
  std::function<bool(const std::array<ConcatTest::iterator, 5>&)> condition = [](const auto& testCombination) { return true; };

  auto comb5 = CombinationsGenerator<ConcatTest, 5>(tracks, condition);
  int64_t count = 0;

  for (auto _ : state) {
    count = 0;
    for (auto comb : comb5) {
      count++;
    }
    benchmark::DoNotOptimize(count);
  }
  state.counters["Combinations"] = count;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_ASoAHelpersCombGenTracksFivesMultipleChunks)->RangeMultiplier(2)->Range(8, 8 << (maxFivesRange - 1));

BENCHMARK_MAIN();
