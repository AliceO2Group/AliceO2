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
#include <benchmark/benchmark.h>

#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include <iostream>

// a simple benchmark of the contribution of the pure message creation
// this was important when the benchmarks below included the message
// creation inside the benchmark loop, its somewhat obsolete now but
// we keep it for reference
static void BM_RelayStackLifecycle(benchmark::State& state)
{
  using namespace o2::framework;
  using namespace o2::header;
  DataProcessingHeader dph{0, 1};

  for (auto _ : state) {
    DataHeader dh;
    dh.dataDescription = "CLUSTERS";
    dh.dataOrigin = "TPC";
    dh.subSpecification = 0;

    DataProcessingHeader dph{0, 1};
    Stack stack{dh, dph};
  }
}

BENCHMARK(BM_RelayStackLifecycle);

BENCHMARK_MAIN();
