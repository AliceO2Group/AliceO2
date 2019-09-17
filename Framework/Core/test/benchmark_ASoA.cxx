// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ASoA.h"
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

static void BM_SimpleForLoop(benchmark::State& state)
{
  struct XYZ {
    float x;
    float y;
    float z;
  };
  std::vector<XYZ> foo;
  foo.resize(state.range(0));

  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  for (size_t i = 0; i < state.range(0); ++i) {
    foo[i] = XYZ{uniform_dist(e1), uniform_dist(e1), uniform_dist(e1)};
  }

  for (auto _ : state) {
    float sum = 0;
    for (auto& xyz : foo) {
      benchmark::DoNotOptimize(sum++);
    }
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_SimpleForLoop)->Range(8, 8 << 17);

static void BM_TrackForLoop(benchmark::State& state)
{
  struct TestTrack {
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;
  };
  std::vector<TestTrack> foo;
  foo.resize(state.range(0));

  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  for (size_t i = 0; i < state.range(0); ++i) {
    foo[i] = TestTrack{
      uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
      uniform_dist(e1), uniform_dist(e1), uniform_dist(e1)};
  }

  for (auto _ : state) {
    float sum = 0;
    for (auto& xyz : foo) {
      sum += xyz.a + xyz.d;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_TrackForLoop)->Range(8, 8 << 17);

static void BM_TrackForPhi(benchmark::State& state)
{
  struct TestTrack {
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;
  };
  std::vector<TestTrack> foo;
  foo.resize(state.range(0));

  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  for (size_t i = 0; i < state.range(0); ++i) {
    foo[i] = TestTrack{
      uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
      uniform_dist(e1), uniform_dist(e1), uniform_dist(e1)};
  }

  for (auto _ : state) {
    size_t i = 0;
    std::vector<float> result;
    result.resize(state.range(0));
    for (auto& track : foo) {
      result[i++] = asin(track.a) + track.d + M_PI;
    }
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_TrackForPhi)->Range(8, 8 << 17);

static void BM_SimpleForLoopWithOp(benchmark::State& state)
{
  struct XYZ {
    float x;
    float y;
    float z;
  };
  std::vector<XYZ> foo;
  foo.resize(state.range(0));

  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  for (size_t i = 0; i < state.range(0); ++i) {
    foo[i] = XYZ{uniform_dist(e1), uniform_dist(e1), uniform_dist(e1)};
  }

  for (auto _ : state) {
    float sum = 0;
    for (auto& xyz : foo) {
      sum += xyz.x + xyz.y;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_SimpleForLoopWithOp)->Range(8, 8 << 17);

static void BM_ASoASimpleForLoop(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (size_t i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X>;

  for (auto _ : state) {
    float sum = 0;
    Test tests{table};
    for (auto& test : tests) {
      sum++;
    }
    benchmark::DoNotOptimize(sum++);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_ASoASimpleForLoop)->Range(8, 8 << 17);

static void BM_ASoASimpleForLoopWithOp(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (size_t i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X, test::Y>;

  for (auto _ : state) {
    Test tests{table};
    float sum = 0;
    for (auto& test : tests) {
      sum += test.x() + test.y();
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_ASoASimpleForLoopWithOp)->Range(8, 8 << 17);

static void BM_ASoADynamicColumnPresent(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (size_t i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X, test::Y, test::Z, test::Sum<test::X, test::Y>>;

  for (auto _ : state) {
    Test tests{table};
    float sum = 0;
    for (auto& test : tests) {
      sum += test.x() + test.y();
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_ASoADynamicColumnPresent)->Range(8, 8 << 17);

static void BM_ASoADynamicColumnCall(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.persist<float, float, float>({"x", "y", "z"});
  for (size_t i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  using Test = o2::soa::Table<test::X, test::Y, test::Sum<test::X, test::Y>>;

  Test tests{table};
  for (auto _ : state) {
    float sum = 0;
    for (auto& test : tests) {
      sum += test.sum();
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_ASoADynamicColumnCall)->Range(8, 8 << 17);

static void BM_ASoAGettersPhi(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (size_t i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
  for (auto _ : state) {
    int i = 0;
    state.PauseTiming();
    std::vector<float> out;
    out.resize(state.range(0));
    float* result = out.data();
    state.ResumeTiming();
    for (auto& track : tracks) {
      *result++ = asin(track.snp()) + track.alpha() + M_PI;
    }
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}

BENCHMARK(BM_ASoAGettersPhi)->Range(8, 8 << 17);

static void BM_ASoADynamicColumnPhi(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0, 1);

  TableBuilder builder;
  auto rowWriter = builder.cursor<o2::aod::Tracks>();
  for (size_t i = 0; i < state.range(0); ++i) {
    rowWriter(0, uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
              uniform_dist(e1), uniform_dist(e1));
  }
  auto table = builder.finalize();

  o2::aod::Tracks tracks{table};
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<float> out;
    out.resize(state.range(0));
    float* result = out.data();
    state.ResumeTiming();
    for (auto& track : tracks) {
      *result++ = track.phi();
    }
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(float) * 2);
}
BENCHMARK(BM_ASoADynamicColumnPhi)->Range(8, 8 << 17);

BENCHMARK_MAIN()
