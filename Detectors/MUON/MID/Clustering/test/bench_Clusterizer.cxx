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
#include <random>
#include <gsl/gsl>
#include "MIDBase/Mapping.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDClustering/PreCluster.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDClustering/Clusterizer.h"

o2::mid::ColumnData& getColumn(std::vector<o2::mid::ColumnData>& patterns, uint8_t icolumn, uint8_t deId)
{
  for (auto& currColumn : patterns) {
    if (currColumn.columnId == icolumn) {
      return currColumn;
    }
  }

  patterns.emplace_back(o2::mid::ColumnData{deId, icolumn});
  return patterns.back();
}

bool addStrip(o2::mid::ColumnData& column, int cathode, int line, int strip)
{
  uint16_t pattern = column.getPattern(cathode, line);
  uint16_t currStrip = (1 << strip);
  if (pattern & currStrip) {
    return false;
  }
  column.addStrip(strip, cathode, line);
  return true;
}

void addNeighbour(std::vector<o2::mid::ColumnData>& patterns, o2::mid::Mapping::MpStripIndex stripIndex, int cathode,
                  int deId, const o2::mid::Mapping& midMapping, int& nAdded, int maxAdded)
{
  std::vector<o2::mid::Mapping::MpStripIndex> neighbours = midMapping.getNeighbours(stripIndex, cathode, deId);
  for (auto& neigh : neighbours) {
    o2::mid::ColumnData& column = getColumn(patterns, static_cast<uint8_t>(neigh.column), static_cast<uint8_t>(deId));
    if (!addStrip(column, cathode, neigh.line, neigh.strip)) {
      continue;
    }
    ++nAdded;
    if (nAdded >= maxAdded) {
      return;
    }
    addNeighbour(patterns, neigh, cathode, deId, midMapping, nAdded, maxAdded);
    if (nAdded >= maxAdded) {
      return;
    }
  }
}

std::vector<o2::mid::ColumnData> generateTestData(int deId, int nClusters, int clusterSize,
                                                  const o2::mid::Mapping& midMapping)
{
  int firstColumnInDE = midMapping.getFirstColumn(deId);

  std::uniform_int_distribution<int> distColumn(firstColumnInDE, 6);

  std::random_device rd;
  std::mt19937 mt(rd());

  std::vector<o2::mid::ColumnData> patterns;
  o2::mid::Mapping::MpStripIndex stripIndex;
  std::vector<o2::mid::Mapping::MpStripIndex> neighbours;

  for (int icl = 0; icl < nClusters; ++icl) {
    int icolumn = distColumn(mt);
    for (int cathode = 0; cathode < 2; ++cathode) {
      int iline = (cathode == 1) ? 0 : midMapping.getFirstBoardBP(icolumn, deId);
      int nStrips = (cathode == 0) ? 16 : midMapping.getNStripsNBP(icolumn, deId);
      std::uniform_int_distribution<int> distStrip(0, nStrips - 1);
      stripIndex.column = icolumn;
      stripIndex.line = iline;
      stripIndex.strip = distStrip(mt);
      o2::mid::ColumnData& column = getColumn(patterns, static_cast<uint8_t>(icolumn), static_cast<uint8_t>(deId));
      addStrip(column, cathode, iline, stripIndex.strip);
      int nAdded = 1;
      if (nAdded < clusterSize) {
        addNeighbour(patterns, stripIndex, cathode, deId, midMapping, nAdded, clusterSize);
      }
    }
  }
  return patterns;
}

class BenchClustering : public benchmark::Fixture
{
 public:
  BenchClustering() : midMapping(), preClusterizer(), clusterizer()
  {
    preClusterizer.init();
    clusterizer.init();
  }
  o2::mid::Mapping midMapping;
  o2::mid::PreClusterizer preClusterizer;
  o2::mid::Clusterizer clusterizer;
};

BENCHMARK_DEFINE_F(BenchClustering, clustering)
(benchmark::State& state)
{

  int deId = state.range(0);
  int nClusters = state.range(1);
  int clusterSize = state.range(2);
  double num{0};

  std::vector<o2::mid::ColumnData> inputData;

  for (auto _ : state) {
    state.PauseTiming();
    inputData = generateTestData(deId, nClusters, clusterSize, midMapping);
    state.ResumeTiming();
    preClusterizer.process(inputData);
    gsl::span<const o2::mid::PreCluster> preClusters(preClusterizer.getPreClusters().data(), preClusterizer.getPreClusters().size());
    clusterizer.process(preClusters);
    ++num;
  }

  state.counters["num"] = benchmark::Counter(num, benchmark::Counter::kIsRate);
}

static void CustomArguments(benchmark::internal::Benchmark* bench)
{
  std::vector<int> deIdList = {63, 66, 67, 68, 69};
  for (auto& deId : deIdList) {
    for (int nClusters = 1; nClusters < 4; ++nClusters) {
      for (int clustSize = 1; clustSize < 4; ++clustSize) {
        bench->Args({deId, nClusters, clustSize});
      }
    }
  }
}

BENCHMARK_REGISTER_F(BenchClustering, clustering)->Apply(CustomArguments)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
