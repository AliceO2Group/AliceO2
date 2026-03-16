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

/// \file GPUMemorySizeScalers.cxx
/// \author David Rohr

#include "GPUMemorySizeScalers.h"
#include "GPULogging.h"

#include <random>

using namespace o2::gpu;

void GPUMemorySizeScalers::rescaleMaxMem(size_t newAvailableMemory)
{
  GPUMemorySizeScalers tmp;
  double scaleFactor = (double)newAvailableMemory / tmp.availableMemory;
  if (scaleFactor != 1.) {
    GPUInfo("Rescaling buffer size limits from %lu to %lu bytes of memory (factor %f)", tmp.availableMemory, newAvailableMemory, scaleFactor);
  }
  tpcMaxPeaks = (double)tmp.tpcMaxPeaks * scaleFactor;
  tpcMaxClusters = (double)tmp.tpcMaxClusters * scaleFactor;
  tpcMaxStartHits = (double)tmp.tpcMaxStartHits * scaleFactor;
  tpcMaxRowStartHits = (double)tmp.tpcMaxRowStartHits * scaleFactor;
  tpcMaxTracklets = (double)tmp.tpcMaxTracklets * scaleFactor;
  tpcMaxTrackletHits = (double)tmp.tpcMaxTrackletHits * scaleFactor;
  tpcMaxSectorTracks = (double)tmp.tpcMaxSectorTracks * scaleFactor;
  tpcMaxSectorTrackHits = (double)tmp.tpcMaxSectorTrackHits * scaleFactor;
  tpcMaxMergedTracks = (double)tmp.tpcMaxMergedTracks * scaleFactor;
  tpcMaxMergedTrackHits = (double)tmp.tpcMaxMergedTrackHits * scaleFactor;
  availableMemory = newAvailableMemory;
}

float GPUMemorySizeScalers::getScalingFactor()
{
  if (!doFuzzing) {
    return scalingFactor;
  }
  static std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  static std::mt19937 rng;
  if (fuzzSeed) {
    rng = std::mt19937(fuzzSeed);
    fuzzLimit = dist(rng) / 10;
    fuzzSeed = 0;
  }
  if (dist(rng) > fuzzLimit) {
    return scalingFactor;
  }
  return scalingFactor * 0.000001 * dist(rng);
}

void GPUMemorySizeScalers::fuzzScalingFactor(uint64_t seed)
{
  fuzzSeed = seed;
  doFuzzing = true;
}

size_t GPUMemorySizeScalers::getValue(size_t maxVal, size_t val)
{
  return returnMaxVal ? maxVal : (std::min<size_t>(maxVal, offset + val) * (doFuzzing == 0 ? scalingFactor : getScalingFactor()) * temporaryFactor);
}

size_t GPUMemorySizeScalers::NTPCPeaks(size_t tpcDigits, bool perSector) { return getValue(perSector ? tpcMaxPeaks : (GPUCA_NSECTORS * tpcMaxPeaks), hitOffset + tpcDigits * tpcPeaksPerDigit); }
size_t GPUMemorySizeScalers::NTPCClusters(size_t tpcDigits, bool perSector) { return getValue(perSector ? tpcMaxSectorClusters : tpcMaxClusters, (conservativeMemoryEstimate ? 1.0 : tpcClustersPerPeak) * NTPCPeaks(tpcDigits, perSector)); }
size_t GPUMemorySizeScalers::NTPCStartHits(size_t tpcHits) { return getValue(tpcMaxStartHits, tpcHits * tpcStartHitsPerHit); }
size_t GPUMemorySizeScalers::NTPCRowStartHits(size_t tpcHits) { return getValue(tpcMaxRowStartHits, std::max<size_t>(NTPCStartHits(tpcHits) * (tpcHits < 30000000 ? 20 : 12) / GPUCA_ROW_COUNT, tpcMinRowStartHits)); }
size_t GPUMemorySizeScalers::NTPCTracklets(size_t tpcHits, bool lowField) { return getValue(tpcMaxTracklets, NTPCStartHits(tpcHits) * (lowField ? tpcTrackletsPerStartHitLowField : tpcTrackletsPerStartHit)); }
size_t GPUMemorySizeScalers::NTPCTrackletHits(size_t tpcHits, bool lowField) { return getValue(tpcMaxTrackletHits, hitOffset + tpcHits * (lowField ? tpcTrackletHitsPerHitLowField : tpcTrackletHitsPerHit)); }
size_t GPUMemorySizeScalers::NTPCSectorTracks(size_t tpcHits) { return getValue(tpcMaxSectorTracks, tpcHits * tpcSectorTracksPerHit); }
size_t GPUMemorySizeScalers::NTPCSectorTrackHits(size_t tpcHits, uint8_t withRejection) { return getValue(tpcMaxSectorTrackHits, tpcHits * (withRejection ? tpcSectorTrackHitsPerHitWithRejection : tpcSectorTrackHitsPerHit)); }
size_t GPUMemorySizeScalers::NTPCMergedTracks(size_t tpcSectorTracks) { return getValue(tpcMaxMergedTracks, tpcSectorTracks * (conservativeMemoryEstimate ? 1.0 : tpcMergedTrackPerSectorTrack)); }
size_t GPUMemorySizeScalers::NTPCMergedTrackHits(size_t tpcSectorTrackHitss) { return getValue(tpcMaxMergedTrackHits, tpcSectorTrackHitss * tpcMergedTrackHitPerSectorHit); }
size_t GPUMemorySizeScalers::NTPCUnattachedHitsBase1024(int32_t type) { return (returnMaxVal || conservativeMemoryEstimate) ? 1024 : std::min<size_t>(1024, tpcCompressedUnattachedHitsBase1024[type] * (doFuzzing == 0 ? scalingFactor : getScalingFactor()) * temporaryFactor); }
