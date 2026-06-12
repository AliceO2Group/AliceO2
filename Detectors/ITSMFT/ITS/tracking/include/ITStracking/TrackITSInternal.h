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

#ifndef TRACKINGITSU_INCLUDE_TRACKITSINTERNAL_H_
#define TRACKINGITSU_INCLUDE_TRACKITSINTERNAL_H_

#include <array>

#include "GPUCommonDef.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "ITStracking/Constants.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::its
{

template <int NLayers>
struct TrackITSInternal {
  GPUhdi() TrackITSInternal() { resetClusters(); }

  GPUhdi() void resetClusters()
  {
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      clusters[iLayer] = constants::UnusedIndex;
    }
    nClusters = 0;
  }

  GPUhdi() int getClusterIndex(int layer) const { return clusters[layer]; }

  GPUhdi() void setClusterIndex(int layer, int cluster)
  {
    if (clusters[layer] == constants::UnusedIndex && cluster != constants::UnusedIndex) {
      ++nClusters;
    } else if (clusters[layer] != constants::UnusedIndex && cluster == constants::UnusedIndex) {
      --nClusters;
    }
    clusters[layer] = cluster;
  }

  GPUhdi() int getNClusters() const { return nClusters; }
  GPUhdi() int getNumberOfClusters() const { return nClusters; }
  GPUhdi() float getChi2() const { return chi2; }
  GPUhdi() void setChi2(float value) { chi2 = value; }
  GPUdi() float getPt() const { return paramIn.getPt(); }

  GPUhdi() uint32_t getPattern() const
  {
    uint32_t pattern{0};
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      if (clusters[iLayer] != constants::UnusedIndex) {
        pattern |= (0x1u << iLayer);
      }
    }
    return pattern;
  }

  GPUhdi() int getFirstClusterLayer() const
  {
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      if (clusters[iLayer] != constants::UnusedIndex) {
        return iLayer;
      }
    }
    return constants::UnusedIndex;
  }

  GPUhdi() int getLastClusterLayer() const
  {
    for (int iLayer{NLayers - 1}; iLayer >= 0; --iLayer) {
      if (clusters[iLayer] != constants::UnusedIndex) {
        return iLayer;
      }
    }
    return constants::UnusedIndex;
  }

  o2::track::TrackParCov paramIn;
  o2::track::TrackParCov paramOut;
  std::array<int, NLayers> clusters{};
  TimeEstBC time;
  float chi2{0.f};
  int nClusters{0};
};

template <int NLayers>
GPUhdi() TrackITSExt makeTrackITSExt(const TrackITSInternal<NLayers>& track)
{
  TrackITSExt out;
  out.getParamIn() = track.paramIn;
  out.getParamOut() = track.paramOut;
  out.setChi2(track.chi2);
  out.getTimeStamp() = track.time.makeSymmetrical();
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    if (track.clusters[iLayer] != constants::UnusedIndex) {
      out.setExternalClusterIndex(iLayer, track.clusters[iLayer], true);
    }
  }
  return out;
}

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_TRACKITSINTERNAL_H_ */
