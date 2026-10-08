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

/// \file CosmicTrack.h
/// \brief Matched cosmic track with the raw clusters of its legs and of the road around them, for offline refits

#ifndef ALICEO2_COSMIC_TRACK_H
#define ALICEO2_COSMIC_TRACK_H

#include <vector>
#include <cstdint>
#include <Rtypes.h>
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::dataformats
{

/// raw TPC cluster with its address; transformed coordinates are not stored (re-transform offline with the calibration of the TF)
struct CosmicTPCCluster {
  enum Flags : uint8_t {
    Attached = 0x1, ///< attached to the TPC track of this leg
    Corridor = 0x2, ///< found in the road around the leg
    Used = 0x4,     ///< attached to some TPC track (for corridor clusters: another track, e.g. a split piece of the leg)
    AbsTime = 0x8   ///< found on the other TPC side with the absolute time of the cosmic instead of the time0 of the leg: CosmicTrack::timeTOFMUS
                    ///< if >= 0, else the time of CosmicTrack::cosmic
  };
  o2::tpc::ClusterNative cl{}; ///< raw cluster: time, pad, widths, charges, flags
  uint8_t sector = 0;
  uint8_t row = 0;
  uint8_t flags = 0;
  bool isAttached() const { return flags & Attached; }
  bool isCorridor() const { return flags & Corridor; }
  bool isUsed() const { return flags & Used; }
  bool isAbsTime() const { return flags & AbsTime; }
  ClassDefNV(CosmicTPCCluster, 1);
};

/// origin of the ITS / TOF / TRD hits of a cosmic
enum CosmicHitFlags : uint8_t {
  HitMatched = 0x1,  ///< part of the leg's matched global track
  HitRoad = 0x2,     ///< found by the road search around the cosmic
  HitTOFFlight = 0x4 ///< TOF road hit of the top/bottom pair whose time difference matches the muon's flight (fixes the cosmic's time)
};

/// ITS cluster of a leg: the raw compact cluster (chip, anchor pixel, pattern ID) and its ROF; no coordinates (the dictionary and the
/// geometry are applied offline)
struct CosmicITSCluster {
  uint16_t chipID = 0;
  uint16_t row = 0;
  uint16_t col = 0;
  uint16_t pattID = 0;
  int32_t pattEntry = -1; ///< start of the pattern bytes in CosmicTrack::itsPatterns (pattern not in the dictionary or group pattern), -1: none
  int32_t rofBC = 0;      ///< start of the cluster's ROF in BCs since the start of the TF
  uint8_t leg = 0;        ///< 0 bottom, 1 top
  uint8_t flags = 0;      ///< CosmicHitFlags
  ClassDefNV(CosmicITSCluster, 1);
};

/// TOF cluster of a leg: matched to its track or found on the road
struct CosmicTOFCluster {
  double timeRaw = 0.; ///< raw TOF time [ps] (the calibration is applied offline)
  float tot = 0.f;     ///< time over threshold
  int32_t channel = -1;
  uint8_t leg = 0;
  uint8_t flags = 0; ///< CosmicHitFlags
  ClassDefNV(CosmicTOFCluster, 1);
};

/// TRD tracklet of a leg: attached to its track or found on the road
struct CosmicTRDTracklet {
  uint64_t word = 0;  ///< raw Tracklet64 word
  int32_t trigBC = 0; ///< BC of its trigger since the start of the TF
  uint8_t layer = 0;
  uint8_t leg = 0;
  uint8_t flags = 0; ///< CosmicHitFlags
  ClassDefNV(CosmicTRDTracklet, 1);
};

/// matched cosmic with everything needed for an offline refit
struct CosmicTrack {
  o2::dataformats::TrackCosmics cosmic{};      ///< matcher output (time in mus; the leg references are only valid within the TF)
  o2::tpc::TrackTPC tpcBottom{};               ///< TPC part of the bottom leg (default if none); z refers to its time0; its cluster
                                               ///< references are only valid within the TF: the clusters are in clTPCBottom
  o2::tpc::TrackTPC tpcTop{};                  ///< TPC part of the top leg
  std::vector<CosmicTPCCluster> clTPCBottom;   ///< attached + corridor TPC clusters of the bottom leg
  std::vector<CosmicTPCCluster> clTPCTop;      ///< same for the top leg
  std::vector<CosmicITSCluster> clITS;         ///< ITS clusters of the legs' matched tracks and on the road
  std::vector<uint8_t> itsPatterns;            ///< pattern bytes (row span, column span, bitmap) of the ITS clusters that need them
  std::vector<CosmicTOFCluster> clTOF;         ///< TOF clusters of the legs' matched tracks and on the road
  std::vector<CosmicTRDTracklet> trdTracklets; ///< TRD tracklets of the legs' matched tracks and on the road
  float timeTOFMUS = -1.f;                     ///< time of the cosmic [mus since the TF start, negative before it] from its HitTOFFlight pair (valid if hasTOFTime())
  float scoreTOFPair = -1.f;                   ///< score of the HitTOFFlight pair: road and flight-time residuals squared in units of the cuts (< 0: none)
  float scoreTOFReversed = -1.f;               ///< same for the best pair in the impossible order (bottom hit first) = an accidental coincidence:
                                               ///< QA of the flag's background; comparable to scoreTOFPair = ambiguous TOF time (< 0: none)
  int duplicateOf = -1;                        ///< entry (in the TF) of the cosmic this one duplicates: the same muon matched twice (e.g. a leg split
                                               ///< into two TPC tracks): >= 30 % of the smaller one's TPC clusters shared with a better one (TOF time,
                                               ///< more attached clusters) (-1: none)
  o2::track::TrackParCov polished{};           ///< refit at the TOF time (TPC-only legs, muon mass, energy loss along the flight), at the closest
                                               ///< approach to the beam line; valid if chi2MatchPolished >= 0
  float chi2MatchPolished = -1.f;              ///< chi2 of the polished bottom and top halves at the closest approach (< 0: not polished)
  o2::MCCompLabel label{};                     ///< MC label of the cosmic (MC only)

  bool hasTOFTime() const { return scoreTOFPair >= 0.f; } ///< a HitTOFFlight pair gave the cosmic its time timeTOFMUS
  ClassDefNV(CosmicTrack, 1);
};

/// per-TF quantities of the TPC transformation used in the reconstruction
struct CosmicsTFInfo {
  float vDrift = 0.f; ///< drift velocity of the transformation [cm/time bin]
  float t0 = 0.f;     ///< time offset of the transformation [time bins]
  ClassDefNV(CosmicsTFInfo, 1);
};

} // namespace o2::dataformats

#endif
