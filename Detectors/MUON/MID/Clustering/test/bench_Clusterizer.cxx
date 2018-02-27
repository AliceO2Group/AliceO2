// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/test/bench_Clusterizer.cxx
/// \brief  Benchmark clustering device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018

#include "benchmark/benchmark.h"
#include <iostream>
#include <random>
#include "MIDBase/Mapping.h"
#include "DataFormatsMID/StripPattern.h"
#include "Clusterizer.h"

std::vector<o2::mid::ColumnData> generateTestData(int deId, const o2::mid::Mapping& midMapping)
{
  int firstColumnInDE = midMapping.getFirstColumn(deId);

  std::uniform_int_distribution<int> distColumn(firstColumnInDE, 6);
  std::uniform_int_distribution<int> distStrip(0, 15);
  std::uniform_int_distribution<int> distNfired(0, 4);

  std::random_device rd;
  std::mt19937 mt(rd());

  std::vector<o2::mid::ColumnData> patterns;
  int firstColumn = distColumn(mt);
  int lastColumn = firstColumn + 1;
  if (lastColumn > 6) {
    lastColumn = 6;
  }

  for (int icol = firstColumn; icol <= lastColumn; ++icol) {
    o2::mid::ColumnData column;
    column.deId = (uint8_t)deId;
    column.columnId = (uint8_t)icol;
    for (int cathode = 0; cathode < 2; ++cathode) {
      int firstLine = 0;
      int lastLine = 0;
      if (cathode == 0) {
        firstLine = midMapping.getFirstBoardBP(icol, deId);
        lastLine = midMapping.getLastBoardBP(icol, deId);
      }
      for (int iline = firstLine; iline <= lastLine; ++iline) {
        uint16_t pattern = 0;
        int nStrips = distNfired(mt);
        for (int istrip = 0; istrip < nStrips; ++istrip) {
          if (midMapping.stripByLocation(istrip, cathode, iline, column.columnId, column.deId).isValid()) {
            pattern |= (1 << istrip);
          }
        }
        if (cathode == 0) {
          column.patterns.setBendPattern(pattern, iline);
        } else {
          column.patterns.setNonBendPattern(pattern);
        }
      }
    }
    patterns.emplace_back(column);
  }
  return patterns;
}

static void deList(benchmark::internal::Benchmark* bench)
{
  for (int deId = 63; deId < 72; ++deId) {
    bench->Args({ deId });
  }
}

class BenchO2 : public benchmark::Fixture
{
};
BENCHMARK_DEFINE_F(BenchO2, clustering)(benchmark::State& state)
{

  int deId = state.range(0);

  o2::mid::Mapping midMapping;
  o2::mid::Clusterizer clusterizer;
  clusterizer.init();

  double num{ 0 };

  std::vector<o2::mid::ColumnData> inputData;

  for (auto _ : state) {
    state.PauseTiming();
    inputData = generateTestData(deId, midMapping);
    state.ResumeTiming();
    clusterizer.process(inputData);
    ++num;
  }

  state.counters["num"] = benchmark::Counter(num, benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(BenchO2, clustering)->Apply(deList)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
