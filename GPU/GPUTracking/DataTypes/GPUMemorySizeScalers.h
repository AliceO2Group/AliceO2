// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUMemorySizeScalers.h
/// \author David Rohr

#ifndef O2_GPU_GPUMEMORYSIZESCALERS_H
#define O2_GPU_GPUMEMORYSIZESCALERS_H

#include "GPUDef.h"

namespace GPUCA_NAMESPACE::gpu
{

struct GPUMemorySizeScalers {
  // Input sizes
  size_t nTPCdigits = 0;
  size_t nTPCHits = 0;
  size_t nTRDTracklets = 0;
  size_t nITSTracks = 0;

  // General scaling factor
  double factor = 1;

  // Offset
  double offset = 1000.;
  double hitOffset = 20000;

  // Scaling Factors
  double tpcPeaksPerDigit = 0.2;
  double tpcClustersPerPeak = 0.9;
  double tpcStartHitsPerHit = 0.08;
  double tpcTrackletsPerStartHit = 0.8;
  double tpcTrackletHitsPerHit = 5;
  double tpcSectorTracksPerHit = 0.02;
  double tpcSectorTrackHitsPerHit = 0.8f;
  double tpcMergedTrackPerSliceTrack = 0.9;
  double tpcMergedTrackHitPerSliceHit = 1.1;

  // Upper limits
  size_t tpcMaxPeaks = 1000000000;
  size_t tpcMaxClusters = 620000000;
  size_t tpcMaxStartHits = 1250000;
  size_t tpcMaxRowStartHits = 1000000000;
  size_t tpcMaxTracklets = 1000000;
  size_t tpcMaxTrackletHits = 66000000;
  size_t tpcMaxSectorTracks = 250000;
  size_t tpcMaxSectorTrackHits = 11500000;
  size_t tpcMaxMergedTracks = 5800000;
  size_t tpcMaxMergedTrackHits = 380000000;

  size_t NTPCPeaks(size_t tpcDigits) { return std::min<size_t>(tpcMaxPeaks, hitOffset + tpcDigits * tpcPeaksPerDigit) * factor; }
  size_t NTPCClusters(size_t tpcDigits) { return std::min<size_t>(tpcMaxClusters, tpcClustersPerPeak * NTPCPeaks(tpcDigits)) * factor; }
  size_t NTPCStartHits(size_t tpcHits) { return std::min<size_t>(tpcMaxStartHits, offset + tpcHits * tpcStartHitsPerHit) * factor; }
  size_t NTPCRowStartHits(size_t tpcHits) { return std::min<size_t>(tpcMaxRowStartHits, offset + NTPCStartHits(tpcHits) / GPUCA_ROW_COUNT * 4.) * factor; }
  size_t NTPCTracklets(size_t tpcHits) { return std::min<size_t>(tpcMaxTracklets, NTPCStartHits(tpcHits) * tpcTrackletsPerStartHit) * factor; }
  size_t NTPCTrackletHits(size_t tpcHits) { return std::min<size_t>(tpcMaxTrackletHits, hitOffset + tpcHits * tpcTrackletHitsPerHit) * factor; }
  size_t NTPCSectorTracks(size_t tpcHits) { return std::min<size_t>(tpcMaxSectorTracks, offset + tpcHits * tpcSectorTracksPerHit) * factor; }
  size_t NTPCSectorTrackHits(size_t tpcHits) { return std::min<size_t>(tpcMaxSectorTrackHits, offset + tpcHits * tpcSectorTrackHitsPerHit) * factor; }
  size_t NTPCMergedTracks(size_t tpcSliceTracks) { return std::min<size_t>(tpcMaxMergedTracks, offset + tpcSliceTracks * tpcMergedTrackPerSliceTrack) * factor; }
  size_t NTPCMergedTrackHits(size_t tpcSliceTrackHitss) { return std::min<size_t>(tpcMaxMergedTrackHits, offset + tpcSliceTrackHitss * tpcMergedTrackHitPerSliceHit) * factor; }
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
