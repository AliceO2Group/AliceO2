// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Expressions.h"
#include "../src/ExpressionHelpers.h"

#include "Framework/HistogramRegistry.h"
#include "Framework/Logger.h"
#include "../src/ExpressionHelpers.h"

#include <benchmark/benchmark.h>
#include <random>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

namespace test
{
DECLARE_SOA_COLUMN(X, x, float, "x");
DECLARE_SOA_COLUMN(Y, y, float, "y");
DECLARE_SOA_COLUMN(Z, z, float, "z");
DECLARE_SOA_COLUMN(cD, cd, float, "cd");
DECLARE_SOA_DYNAMIC_COLUMN(D, d, [](float x, float y, float z) { return std::sqrt(x * x + y * y + z * z); });
} // namespace test

using TT = o2::soa::Table<test::X, test::Y, test::Z, test::D<test::X, test::Y, test::Z>>;
using RT = o2::soa::Table<test::X, test::Y, test::Z, test::cD, test::D<test::X, test::Y, test::Z>>;

const static size_t maxrows = 100;

static std::default_random_engine e(1234567890);
static std::normal_distribution<float> G;

auto createTable = [](size_t nrows) {
  static TableBuilder builder;
  auto static rowWriter = builder.persist<float, float, float>({"x", "y", "z"});

  for (auto i = 0u; i < nrows; ++i) {
    rowWriter(0, G(e), G(e), G(e));
  }
  auto table = builder.finalize();

  return TT{table};
};

static void BM_DirectCalculation(benchmark::State& state)
{
  state.PauseTiming();
  auto tt = createTable(state.range(0));
  state.ResumeTiming();
  benchmark::DoNotOptimize(tt);
}

static void BM_GandivaExpression(benchmark::State& state)
{
  state.PauseTiming();
  auto tt = createTable(state.range(0));
  state.ResumeTiming();
  benchmark::DoNotOptimize(tt);
}

BENCHMARK(BM_DirectCalculation)->Arg(maxrows);
BENCHMARK(BM_GandivaExpression)->Arg(maxrows);

BENCHMARK_MAIN();
