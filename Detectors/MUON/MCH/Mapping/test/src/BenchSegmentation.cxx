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

#include <algorithm>
#include <random>
#include "benchmark/benchmark.h"
#include "MCHMappingInterface/Segmentation.h"

static void segmentationList(benchmark::internal::Benchmark* b)
{
  o2::mch::mapping::forOneDetectionElementOfEachSegmentationType([&b](int detElemId) {
    b->Args({detElemId});
  });
}

class BenchSegO2 : public benchmark::Fixture
{
};

BENCHMARK_DEFINE_F(BenchSegO2, ctor)
(benchmark::State& state)
{
  int detElemId = state.range(0);

  for (auto _ : state) {
    o2::mch::mapping::Segmentation seg{detElemId};
  }
}

namespace
{
std::vector<int> getDetElemIds()
{
  std::vector<int> deids;
  o2::mch::mapping::forEachDetectionElement(
    [&deids](int detElemId) { deids.push_back(detElemId); });
  return deids;
}
} // namespace

static void benchSegmentationCtorAll(benchmark::State& state)
{
  std::vector<int> deids = getDetElemIds();
  for (auto _ : state) {
    for (auto detElemId : deids) {
      o2::mch::mapping::Segmentation seg{detElemId};
    }
  }
}

static void benchSegmentationCtorMap(benchmark::State& state)
{
  std::vector<int> deids = getDetElemIds();
  std::map<int, o2::mch::mapping::Segmentation> cache;

  for (auto _ : state) {
    for (auto detElemId : deids) {
      cache.emplace(detElemId, o2::mch::mapping::Segmentation(detElemId));
    }
  }
}

static void benchSegmentationCtorMapPtr(benchmark::State& state)
{
  std::vector<int> deids = getDetElemIds();
  std::map<int, o2::mch::mapping::Segmentation*> cache;

  for (auto _ : state) {
    for (auto detElemId : deids) {
      cache.emplace(detElemId, new o2::mch::mapping::Segmentation(detElemId));
    }
  }
}
BENCHMARK(benchSegmentationCtorAll)->Unit(benchmark::kMillisecond);
BENCHMARK(benchSegmentationCtorMap)->Unit(benchmark::kMillisecond);
BENCHMARK(benchSegmentationCtorMapPtr)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BenchSegO2, ctor)->Apply(segmentationList)->Unit(benchmark::kMicrosecond);
