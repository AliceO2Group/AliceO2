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

#ifndef TRACKINGITSU_INCLUDE_TRACKEXTENSIONCANDIDATE_H_
#define TRACKINGITSU_INCLUDE_TRACKEXTENSIONCANDIDATE_H_

#include <array>
#include <cstddef>

#include "GPUCommonDef.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "ITStracking/Constants.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::its
{

inline constexpr unsigned int kExtendedPatternShift = 24;
inline constexpr int kMaxLayersInTrackPattern = 8;

template <int NLayers>
GPUhdi() constexpr uint32_t makeAddedClustersPatternMask()
{
  return (NLayers >= 32) ? 0xffffffffu : ((1u << NLayers) - 1u);
}

template <int NLayers>
GPUhdi() void applyExtendedClustersPattern(TrackITSExt& track, uint32_t diff)
{
  diff &= makeAddedClustersPatternMask<NLayers>();
  track.setUserField(static_cast<uint16_t>(diff));
  if constexpr (NLayers <= kMaxLayersInTrackPattern) {
    track.setPattern(track.getPattern() | (diff << kExtendedPatternShift));
  } else {
    (void)track;
  }
}

template <int NLayers>
GPUhdi() uint32_t getAddedClustersPattern(const TrackITSExt& track)
{
  const auto mask = makeAddedClustersPatternMask<NLayers>();
  if constexpr (NLayers <= kMaxLayersInTrackPattern) {
    const auto diff = (track.getPattern() >> kExtendedPatternShift) & mask;
    if (diff) {
      return diff;
    }
  }
  return track.getUserField() & mask;
}

GPUhdi() void clearAddedClustersPattern(TrackITSExt& track)
{
  track.setUserField(0);
  track.getParamOut().setUserField(0);
}

template <int NLayers>
struct TrackExtensionHypothesis {
  o2::track::TrackParCov param;
  std::array<int, NLayers> clusters{};
  TimeStamp time;
  float chi2{0.f};
  int nClusters{0};
  int edgeLayer{constants::UnusedIndex};
};

template <int NLayers>
struct TrackExtensionCandidate {
  static constexpr float InvalidChi2 = 1.e20f;

  GPUhdi() TrackExtensionCandidate() { reset(); }

  GPUhdi() void reset()
  {
    trackIndex = -1;
    nAddedClusters = 0;
    resultIndex = -1;
    chi2 = InvalidChi2;
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      addedClusters[iLayer] = constants::UnusedIndex;
    }
  }

  GPUhdi() bool isValidForTrack(int index) const
  {
    return trackIndex == index && nAddedClusters > 0;
  }

  int trackIndex{-1};
  std::array<int, NLayers> addedClusters;
  int nAddedClusters{0};
  int resultIndex{-1};
  float chi2{InvalidChi2};
};

template <int NLayers>
GPUhdi() bool isBetterTrackExtensionCandidate(const TrackExtensionCandidate<NLayers>& a, const TrackExtensionCandidate<NLayers>& b)
{
  return (a.nAddedClusters > b.nAddedClusters) || (a.nAddedClusters == b.nAddedClusters && a.chi2 < b.chi2);
}

template <int NLayers>
struct TrackExtensionResult {
  GPUhdi() void reset()
  {
    candidate.reset();
  }

  GPUhdi() bool isValid() const { return candidate.trackIndex >= 0 && candidate.nAddedClusters > 0; }

  TrackExtensionCandidate<NLayers> candidate;
  TrackITSExt track;
};

inline constexpr int MaxTrackExtensionCandidatesPerTrack = 4;

inline constexpr size_t getFlatTrackExtensionCandidateIndex(size_t trackIndex, size_t candidateIndex)
{
  return trackIndex * MaxTrackExtensionCandidatesPerTrack + candidateIndex;
}

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_TRACKEXTENSIONCANDIDATE_H_ */
