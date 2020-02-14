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
#include "Framework/CommonDataProcessors.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include <TFile.h>

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
constexpr unsigned int maxrange = 15;
#else
constexpr unsigned int maxrange = 20;
#endif

static void BM_Table2Tree(benchmark::State& state)
{

  // initialize a random generator
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<double> rd(0, 1);
  std::normal_distribution<float> rf(5., 2.);
  std::discrete_distribution<ULong64_t> rl({10, 20, 30, 30, 5, 5});
  std::discrete_distribution<int> ri({10, 20, 30, 30, 5, 5});

  // create a table and fill the columns with random numbers
  TableBuilder builder;
  auto rowWriter =
    builder.persist<double, float, ULong64_t, int>({"a", "b", "c", "d"});
  for (auto i = 0; i < state.range(0); ++i) {
    rowWriter(0, rd(e1), rf(e1), rl(e1), ri(e1));
  }
  auto table = builder.finalize();

  // loop over elements of state
  for (auto _ : state) {

    // Open file and create tree
    TFile* fout = new TFile("table2tree.root", "RECREATE");
    TTree* tout = new TTree("table2tree", "table2tree");

    // benchmark the CommonDataProcessors::table2tree function
    CommonDataProcessors::table2tree(tout, table, false);

    // clean up
    fout->Close();
    delete fout;
  }

  state.SetBytesProcessed(state.iterations() * state.range(0) * 24);
}

BENCHMARK(BM_Table2Tree)->Range(8, 8 << maxrange);

BENCHMARK_MAIN();
