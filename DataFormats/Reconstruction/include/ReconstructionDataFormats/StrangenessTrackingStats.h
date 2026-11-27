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

/// \file StrangenessTrackingStats.h
/// Per-timeframe counters for strangeness tracking.
#ifndef ALICEO2_STRANGENESSTRACKINGSTATS_H
#define ALICEO2_STRANGENESSTRACKINGSTATS_H

#include <cstdint>
#include <Rtypes.h>

namespace o2::dataformats
{
struct StrangenessTrackingStats {
  uint64_t nInvalidPVTime = 0;
  uint64_t nSeeds = 0;
  uint64_t nUnmatched = 0;
  uint64_t nPropagationFailures = 0;
  uint64_t nMissingHits = 0;
  uint64_t nMotherTruncated = 0;
  uint64_t nTripletProposals = 0;
  uint64_t nTripletTruncated = 0;
  uint8_t decayFollowingEnabled = 0;
  uint8_t tripletsEnabled = 0;
  uint8_t tripletsDeferredZeroField = 0;
  uint8_t itsOnly = 0;
  uint8_t clustersPerLayer = 0;
  uint8_t clusterIDEncoding = 1; // ClusterID.h composed input-slot/index encoding
  float bz = 0.f;
  ClassDefNV(StrangenessTrackingStats, 1);
};

} // namespace o2::dataformats
#endif
