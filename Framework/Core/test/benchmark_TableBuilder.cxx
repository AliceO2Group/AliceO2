// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/TableBuilder.h"
#include "Framework/TableConsumer.h"

#include <benchmark/benchmark.h>

using namespace o2::framework;

static void BM_TableBuilderOverhead(benchmark::State& state)
{
  using namespace o2::framework;

  for (auto _ : state) {
    TableBuilder builder;
    auto rowWriter = builder.persist<float, float, float>({ "x", "y", "z" });
    auto table = builder.finalize();
  }
}

BENCHMARK(BM_TableBuilderOverhead);

static void BM_TableBuilderScalar(benchmark::State& state)
{
  using namespace o2::framework;
  for (auto _ : state) {
    TableBuilder builder;
    auto rowWriter = builder.persist<float>({ "x" });
    for (size_t i = 0; i < state.range(0); ++i) {
      rowWriter(0, 0.f);
    }
    auto table = builder.finalize();
  }
}

BENCHMARK(BM_TableBuilderScalar)->Arg(1 << 20);
BENCHMARK(BM_TableBuilderScalar)->Range(8, 8 << 16);

static void BM_TableBuilderScalarPresized(benchmark::State& state)
{
  using namespace o2::framework;
  for (auto _ : state) {
    TableBuilder builder;
    auto rowWriter = builder.preallocatedPersist<float>({ "x" }, state.range(0));
    for (size_t i = 0; i < state.range(0); ++i) {
      rowWriter(0, 0.f);
    }
    auto table = builder.finalize();
  }
}

BENCHMARK(BM_TableBuilderScalarPresized)->Arg(1 << 20);
BENCHMARK(BM_TableBuilderScalarPresized)->Range(8, 8 << 16);

static void BM_TableBuilderSimple(benchmark::State& state)
{
  using namespace o2::framework;
  for (auto _ : state) {
    TableBuilder builder;
    auto rowWriter = builder.persist<float, float, float>({ "x", "y", "z" });
    for (size_t i = 0; i < state.range(0); ++i) {
      rowWriter(0, 0.f, 0.f, 0.f);
    }
    auto table = builder.finalize();
  }
}

BENCHMARK(BM_TableBuilderSimple)->Arg(1 << 20);

static void BM_TableBuilderSimple2(benchmark::State& state)
{
  using namespace o2::framework;
  for (auto _ : state) {
    TableBuilder builder;
    auto rowWriter = builder.persist<float, float, float>({ "x", "y", "z" });
    for (size_t i = 0; i < state.range(0); ++i) {
      rowWriter(0, 0.f, 0.f, 0.f);
    }
    auto table = builder.finalize();
  }
}

BENCHMARK(BM_TableBuilderSimple2)->Range(8, 8 << 16);

static void BM_TableBuilderComplex(benchmark::State& state)
{
  using namespace o2::framework;
  for (auto _ : state) {
    TableBuilder builder;
    auto rowWriter = builder.persist<int, float, std::string, bool>({ "x", "y", "s", "b" });
    for (size_t i = 0; i < state.range(0); ++i) {
      rowWriter(0, 0, 0., "foo", true);
    }
    auto table = builder.finalize();
  }
}

BENCHMARK(BM_TableBuilderComplex)->Range(8, 8 << 16);

BENCHMARK_MAIN()
