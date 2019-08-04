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
#include "MCHMappingInterface/CathodeSegmentation.h"
#include "MCHMappingSegContour/CathodeSegmentationContours.h"

struct TestPoint {
  double x, y;
};

std::vector<TestPoint> generateUniformTestPoints(int n, double xmin, double ymin, double xmax, double ymax)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::vector<TestPoint> testPoints;

  testPoints.resize(n);
  std::uniform_real_distribution<double> distX{xmin, xmax};
  std::uniform_real_distribution<double> distY{ymin, ymax};
  std::generate(testPoints.begin(), testPoints.end(), [&distX, &distY, &mt] {
    return TestPoint{distX(mt), distY(mt)};
  });

  return testPoints;
}

static void segmentationList(benchmark::internal::Benchmark* b)
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

BENCHMARK_DEFINE_F(BenchO2, ctor)
(benchmark::State& state)
{
  int detElemId = state.range(0);
  bool isBendingPlane = state.range(1);

  for (auto _ : state) {
    o2::mch::mapping::CathodeSegmentation seg{detElemId, isBendingPlane};
  }
}

std::vector<int> getDetElemIds()
{
  std::vector<int> deids;
  o2::mch::mapping::forOneDetectionElementOfEachSegmentationType(
    [&deids](int detElemId) { deids.push_back(detElemId); });
  return deids;
}

static void benchCathodeSegmentationConstructionAll(benchmark::State& state)
{
  std::vector<int> deids = getDetElemIds();
  for (auto _ : state) {
    for (auto detElemId : deids) {
      for (auto bending : {true, false}) {
        o2::mch::mapping::CathodeSegmentation seg{detElemId, bending};
      }
    }
  }
}

// note: a bench is not a test, so here we assume findPadByPosition is correct,
// we just time it.
// so you must have a test of it somewhere else.
BENCHMARK_DEFINE_F(BenchO2, findPadByPosition)
(benchmark::State& state)
{
  int detElemId = state.range(0);
  bool isBendingPlane = state.range(1);
  o2::mch::mapping::CathodeSegmentation seg{detElemId, isBendingPlane};
  auto bbox = o2::mch::mapping::getBBox(seg);

  const int n = 100000;
  auto testpoints = generateUniformTestPoints(n, bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax());

  int ntp{0};
  for (auto _ : state) {
    ntp = 0;
    for (auto& tp : testpoints) {
      seg.findPadByPosition(tp.x, tp.y);
      ++ntp;
    }
  }
  state.counters["ntp"] = ntp;
}

BENCHMARK(benchCathodeSegmentationConstructionAll)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BenchO2, findPadByPosition)->Apply(segmentationList)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BenchO2, ctor)->Apply(segmentationList)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
