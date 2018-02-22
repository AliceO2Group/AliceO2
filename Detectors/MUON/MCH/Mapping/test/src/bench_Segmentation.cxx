//
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

//
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche


#include <random>
#include "benchmark/benchmark.h"
#include "MCHMappingInterface/Segmentation.h"

static void segmentationList(benchmark::internal::Benchmark *b)
{
  o2::mch::mapping::forOneDetectionElementOfEachSegmentationType([&b](int detElemId) {
    for (auto bending : {true, false}) {
      {
        b->Args({detElemId, bending});
      }
    }
  });
}

class BenchO2 : public benchmark::Fixture
{
};

BENCHMARK_DEFINE_F(BenchO2, ctor)(benchmark::State &state)
{
  int detElemId = state.range(0);
  bool isBendingPlane = state.range(1);

  for (auto _ : state) {
    o2::mch::mapping::Segmentation seg{detElemId, isBendingPlane};
  }

}

static void benchSegmentationConstruction(benchmark::State &state)
{
  std::vector<int> deids;
  o2::mch::mapping::forOneDetectionElementOfEachSegmentationType([&deids](int detElemId) {
    deids.push_back(detElemId);
  });

  for (auto _ : state) {
    for (auto detElemId: deids) {
      for (auto bending : {true, false}) {
        o2::mch::mapping::Segmentation seg{detElemId, bending};
      }
    }
  }
}

BENCHMARK(benchSegmentationConstruction)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BenchO2, ctor)->Apply(segmentationList)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
