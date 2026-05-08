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

/// \file GPUMemorySizeScalers.h
/// \author David Rohr

#ifndef O2_GPU_GPUMEMORYSIZESCALERS_H
#define O2_GPU_GPUMEMORYSIZESCALERS_H

#include "GPUDef.h"
#include "GPUSettings.h"

namespace o2::gpu
{

struct GPUMemorySizeScalers : public GPUSettingsProcessingScaling {
  // Input sizes
  size_t nTPCdigits = 0;
  size_t nTPCHits = 0;
  size_t nTRDTracklets = 0;
  size_t nITSTracks = 0;

  // General scaling factor
  float scalingFactor = 1;
  uint64_t fuzzSeed = 0;
  uint64_t fuzzLimit = 0;
  float temporaryFactor = 1;

  // Upper limits
  size_t tpcMaxPeaks = 20000000;
  size_t tpcMaxClusters = 320000000;
  size_t tpcMaxSectorClusters = 30000000;
  size_t tpcMaxStartHits = 650000;
  size_t tpcMaxRowStartHits = 100000;
  size_t tpcMinRowStartHits = 4000;
  size_t tpcMaxTracklets = 520000;
  size_t tpcMaxTrackletHits = 35000000;
  size_t tpcMaxSectorTracks = 130000;
  size_t tpcMaxSectorTrackHits = 5900000;
  size_t tpcMaxMergedTracks = 3000000;
  size_t tpcMaxMergedTrackHits = 200000000;
  size_t availableMemory = 20500000000;
  bool returnMaxVal = false;
  bool doFuzzing = false;

  void rescaleMaxMem(size_t newAvailableMemory);
  float getScalingFactor();
  void fuzzScalingFactor(uint64_t seed);
  size_t getValue(size_t maxVal, size_t val);
  size_t NTPCPeaks(size_t tpcDigits, bool perSector = false);
  size_t NTPCClusters(size_t tpcDigits, bool perSector = false);
  size_t NTPCStartHits(size_t tpcHits);
  size_t NTPCRowStartHits(size_t tpcHits);
  size_t NTPCTracklets(size_t tpcHits, bool lowField);
  size_t NTPCTrackletHits(size_t tpcHits, bool lowField);
  size_t NTPCSectorTracks(size_t tpcHits);
  size_t NTPCSectorTrackHits(size_t tpcHits, uint8_t withRejection = 0);
  size_t NTPCMergedTracks(size_t tpcSectorTracks);
  size_t NTPCMergedTrackHits(size_t tpcSectorTrackHitss);
  size_t NTPCUnattachedHitsBase1024(int32_t type);
};

} // namespace o2::gpu

#endif
