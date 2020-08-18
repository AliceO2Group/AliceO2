// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/HistogramRegistry.h"
#include "Framework/Logger.h"

#include "TList.h"

#include <benchmark/benchmark.h>
#include <boost/format.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

/// Number of lookups to perform
const int nLookups = 100000;

/// Lookup a histogram by name literal in a HistogramRegistry
static void BM_HashedNameLookup(benchmark::State& state)
{
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<HistogramSpec> specs;
    for (auto i = 0; i < state.range(0); ++i) {
      specs.push_back({(boost::format("histo%1%") % (i + 1)).str().c_str(), (boost::format("Histo %1%") % (i + 1)).str().c_str(), {"TH1F", 100, 0, 1}});
    }
    HistogramRegistry registry{"registry", true, specs};
    state.ResumeTiming();

    for (auto i = 0; i < nLookups; ++i) {
      auto& x = registry.get("histo4");
      benchmark::DoNotOptimize(x);
    }
    state.counters["Average lookup distance"] = ((double)registry.lookup / (double)(state.range(0)));
  }
}

/// Lookup a histogram by name literal in a ROOT TList container
static void BM_StandardNameLookup(benchmark::State& state)
{
  for (auto _ : state) {
    state.PauseTiming();
    TList list;
    list.SetOwner();
    for (auto i = 1; i <= state.range(0); ++i) {
      list.Add(new TH1F((boost::format("histo%1%") % i).str().c_str(), (boost::format("Histo %1%") % i).str().c_str(), 100, 0, 1));
    }
    state.ResumeTiming();

    for (auto i = 0; i < nLookups; ++i) {
      auto x = list.FindObject("histo4");
      benchmark::DoNotOptimize(x);
    }
  }
}
BENCHMARK(BM_HashedNameLookup)->Arg(4)->Arg(8)->Arg(16)->Arg(64)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_StandardNameLookup)->Arg(4)->Arg(8)->Arg(16)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();
