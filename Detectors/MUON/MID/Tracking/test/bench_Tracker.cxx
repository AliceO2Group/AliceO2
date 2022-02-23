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

/// \file   MID/Tracking/test/bench_Tracker.cxx
/// \brief  Benchmark tracker device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 March 2018

#include "benchmark/benchmark.h"
#include <random>
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/HitFinder.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/MpArea.h"
#include "MIDTestingSimTools/TrackGenerator.h"
#include "MIDTracking/Tracker.h"

std::vector<o2::mid::Cluster> generateTestData(int nTracks, o2::mid::TrackGenerator& trackGen,
                                               const o2::mid::HitFinder& hitFinder, const o2::mid::Mapping& mapping)
{
  o2::mid::Mapping::MpStripIndex stripIndex;
  o2::mid::MpArea area;
  std::vector<o2::mid::Cluster> clusters;
  o2::mid::Cluster cl;
  std::vector<o2::mid::Track> tracks = trackGen.generate(nTracks);
  for (auto& track : tracks) {
    for (int ich = 0; ich < 4; ++ich) {
      auto hits = hitFinder.getLocalPositions(track, ich);
      bool isFired = false;
      for (auto& hit : hits) {
        int deId = hit.deId;
        float xPos = hit.xCoor;
        float yPos = hit.yCoor;
        stripIndex = mapping.stripByPosition(xPos, yPos, 0, deId, false);
        if (!stripIndex.isValid()) {
          continue;
        }
        cl.deId = deId;
        area = mapping.stripByLocation(stripIndex.strip, 0, stripIndex.line, stripIndex.column, deId);
        cl.yCoor = area.getCenterY();
        cl.yErr = area.getHalfSizeY() / std::sqrt(3.);
        stripIndex = mapping.stripByPosition(xPos, yPos, 1, deId, false);
        area = mapping.stripByLocation(stripIndex.strip, 1, stripIndex.line, stripIndex.column, deId);
        cl.xCoor = area.getCenterX();
        cl.xErr = area.getHalfSizeX() / std::sqrt(3.);
        clusters.push_back(cl);
      } // loop on fired pos
    }   // loop on chambers
  }     // loop on tracks
  return clusters;
}

static void BM_TRACKER(benchmark::State& state)
{
  o2::mid::GeometryTransformer geoTrans = o2::mid::createDefaultTransformer();
  o2::mid::TrackGenerator trackGen;
  o2::mid::HitFinder hitFinder(geoTrans);
  o2::mid::Mapping mapping;
  o2::mid::Tracker tracker(geoTrans);

  int nTracksPerEvent = state.range(0);
  tracker.init((state.range(1) == 1));
  double num{0};

  std::vector<o2::mid::Cluster> inputData;

  for (auto _ : state) {
    state.PauseTiming();
    inputData = generateTestData(nTracksPerEvent, trackGen, hitFinder, mapping);
    state.ResumeTiming();
    tracker.process(inputData);
    ++num;
  }

  state.counters["num"] = benchmark::Counter(num, benchmark::Counter::kIsRate);
}

static void CustomArguments(benchmark::internal::Benchmark* bench)
{
  for (int itrack = 1; itrack <= 8; ++itrack) {
    for (int imethod = 0; imethod < 2; ++imethod) {
      bench->Args({itrack, imethod});
    }
  }
}

BENCHMARK(BM_TRACKER)->Apply(CustomArguments)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
