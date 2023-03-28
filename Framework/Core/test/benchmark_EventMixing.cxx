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

#include "Framework/ASoAHelpers.h"
#include "Framework/GroupedCombinations.h"
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <list>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

// Validation of new event mixing: time complexity same as for naive loop

#ifdef __APPLE__
constexpr unsigned int maxPairsRange = 5;
constexpr unsigned int maxFivesRange = 3;
#else
constexpr unsigned int maxPairsRange = 5;
constexpr unsigned int maxFivesRange = 3;
#endif
constexpr int numEventsToMix = 5;
constexpr int numTracksPerEvent = 10000;

using namespace o2::framework;
using namespace o2::soa;

static void BM_EventMixingTraditional(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
  std::uniform_real_distribution<float> uniform_dist_x(-0.065f, 0.073f);
  std::uniform_real_distribution<float> uniform_dist_y(-0.320f, 0.360f);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  std::vector<double> xBins{VARIABLE_WIDTH, -0.064, -0.062, -0.060, 0.066, 0.068, 0.070, 0.072};
  std::vector<double> yBins{VARIABLE_WIDTH, -0.320, -0.301, -0.300, 0.330, 0.340, 0.350, 0.360};
  using BinningType = ColumnBinningPolicy<o2::aod::collision::PosX, o2::aod::collision::PosY>;
  BinningType binningOnPositions{{xBins, yBins}, true}; // true is for 'ignore overflows' (true by default)

  TableBuilder colBuilder, trackBuilder;
  auto rowWriterCol = colBuilder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    float x = uniform_dist_x(e1);
    float y = uniform_dist_y(e1);
    rowWriterCol(0, uniform_dist_int(e1),
                 x, y, uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist_int(e1), uniform_dist(e1),
                 uniform_dist_int(e1),
                 uniform_dist(e1), uniform_dist(e1));
  }
  auto tableCol = colBuilder.finalize();
  o2::aod::Collisions collisions{tableCol};
  std::uniform_int_distribution<int> uniform_dist_col_ind(0, collisions.size());

  auto rowWriterTrack = trackBuilder.cursor<o2::aod::StoredTracks>();
  for (auto i = 0; i < numTracksPerEvent * state.range(0); ++i) {
    rowWriterTrack(0, uniform_dist_col_ind(e1), 0,
                   uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                   uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                   uniform_dist(e1));
  }
  auto tableTrack = trackBuilder.finalize();
  o2::aod::StoredTracks tracks{tableTrack};

  int64_t count = 0;
  int64_t colCount = 0;
  int nBinsTot = (xBins.size() - 2) * (yBins.size() - 2);

  for (auto _ : state) {
    count = 0;
    colCount = 0;
    int n = state.range(0);
    std::vector<std::list<o2::aod::Collisions::iterator>> mixingBufferVector;
    for (int i = 0; i < nBinsTot; i++) {
      mixingBufferVector.push_back(std::list<o2::aod::Collisions::iterator>());
    }
    for (auto& col1 : collisions) {
      int bin = binningOnPositions.getBin({col1.posX(), col1.posY()});
      if (bin == -1) {
        continue;
      }

      auto& mixingBuffer = mixingBufferVector[bin];

      if (mixingBuffer.size() > 0) {
        auto tracks1 = tracks.sliceByCached(o2::aod::track::collisionId, col1.globalIndex());
        for (auto& col2 : mixingBuffer) {
          auto tracks2 = tracks.sliceByCached(o2::aod::track::collisionId, col2.globalIndex());
          for (auto& [t1, t2] : combinations(CombinationsFullIndexPolicy(tracks1, tracks2))) {
            count++;
          }
          colCount++;
        }
        if (mixingBuffer.size() >= numEventsToMix) {
          mixingBuffer.pop_back();
        }
      }
      mixingBuffer.push_front(col1);
    }

    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(colCount);
  }
  state.counters["Mixed track pairs"] = count;
  state.counters["Mixed collision pairs"] = colCount;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_EventMixingTraditional)->RangeMultiplier(2)->Range(4, 8 << maxPairsRange);

static void BM_EventMixingCombinations(benchmark::State& state)
{
  // Seed with a real random value, if available
  std::default_random_engine e1(1234567891);
  std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
  std::uniform_real_distribution<float> uniform_dist_x(-0.065f, 0.073f);
  std::uniform_real_distribution<float> uniform_dist_y(-0.320f, 0.360f);
  std::uniform_int_distribution<int> uniform_dist_int(0, 5);

  std::vector<double> xBins{VARIABLE_WIDTH, -0.064, -0.062, -0.060, 0.066, 0.068, 0.070, 0.072};
  std::vector<double> yBins{VARIABLE_WIDTH, -0.320, -0.301, -0.300, 0.330, 0.340, 0.350, 0.360};
  using BinningType = ColumnBinningPolicy<o2::aod::collision::PosX, o2::aod::collision::PosY>;
  BinningType binningOnPositions{{xBins, yBins}, true}; // true is for 'ignore overflows' (true by default)

  TableBuilder colBuilder, trackBuilder;
  auto rowWriterCol = colBuilder.cursor<o2::aod::Collisions>();
  for (auto i = 0; i < state.range(0); ++i) {
    float x = uniform_dist_x(e1);
    float y = uniform_dist_y(e1);
    rowWriterCol(0, uniform_dist_int(e1),
                 x, y, uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                 uniform_dist_int(e1), uniform_dist(e1),
                 uniform_dist_int(e1),
                 uniform_dist(e1), uniform_dist(e1));
  }
  auto tableCol = colBuilder.finalize();
  o2::aod::Collisions collisions{tableCol};
  std::uniform_int_distribution<int> uniform_dist_col_ind(0, collisions.size());

  auto rowWriterTrack = trackBuilder.cursor<o2::aod::StoredTracks>();
  for (auto i = 0; i < numTracksPerEvent * state.range(0); ++i) {
    rowWriterTrack(0, uniform_dist_col_ind(e1), 0,
                   uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                   uniform_dist(e1), uniform_dist(e1), uniform_dist(e1),
                   uniform_dist(e1));
  }
  auto tableTrack = trackBuilder.finalize();
  o2::aod::StoredTracks tracks{tableTrack};

  int64_t count = 0;
  int64_t colCount = 0;

  for (auto _ : state) {
    count = 0;
    colCount = 0;

    auto tracksTuple = std::make_tuple(tracks);
    SameKindPair<o2::aod::Collisions, o2::aod::StoredTracks, BinningType> pair{binningOnPositions, numEventsToMix - 1, -1, collisions, tracksTuple};
    for (auto& [c1, tracks1, c2, tracks2] : pair) {
      int bin = binningOnPositions.getBin({c1.posX(), c1.posY()});
      for (auto& [t1, t2] : combinations(CombinationsFullIndexPolicy(tracks1, tracks2))) {
        count++;
      }
      colCount++;
    }
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(colCount);
  }
  state.counters["Mixed track pairs"] = count;
  state.counters["Mixed collision pairs"] = colCount;
  state.SetBytesProcessed(state.iterations() * sizeof(float) * count);
}

BENCHMARK(BM_EventMixingCombinations)->RangeMultiplier(2)->Range(4, 8 << maxPairsRange);

BENCHMARK_MAIN();
