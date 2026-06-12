// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TRACKINGITSU_INCLUDE_TRACKEXTENSIONHYPOTHESIS_H_
#define TRACKINGITSU_INCLUDE_TRACKEXTENSIONHYPOTHESIS_H_

#include <array>

#include "GPUCommonDef.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "ITStracking/Constants.h"
#include "ITStracking/TrackITSInternal.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::its
{

template <int NLayers>
struct TrackExtensionHypothesis {
  TrackExtensionHypothesis() = default;
  GPUhdi() TrackExtensionHypothesis(const TrackITSInternal<NLayers>& track, bool outward)
  {
    initialiseFromTrack(track, outward);
  }

  GPUhdi() void initialiseFromTrack(const TrackITSInternal<NLayers>& track, bool outward)
  {
    param = outward ? track.paramOut : track.paramIn;
    time = track.time;
    chi2 = track.getChi2();
    nClusters = track.getNClusters();
    edgeLayer = outward ? track.getLastClusterLayer() : track.getFirstClusterLayer();
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      clusters[iLayer] = track.getClusterIndex(iLayer);
    }
  }

  o2::track::TrackParCov param;
  std::array<int, NLayers> clusters{};
  TimeEstBC time;
  float chi2{0.f};
  int nClusters{0};
  int edgeLayer{constants::UnusedIndex};
};

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_TRACKEXTENSIONHYPOTHESIS_H_ */
