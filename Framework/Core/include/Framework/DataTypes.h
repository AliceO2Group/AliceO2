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
#ifndef O2_FRAMEWORK_DATATYPES_H_
#define O2_FRAMEWORK_DATATYPES_H_

#include "CommonConstants/LHCConstants.h"

#include <cstdint>
#include <limits>

namespace o2::aod::collision
{
enum CollisionFlagsRun2 : uint16_t {
  Run2VertexerTracks = 0x1,
  Run2VertexerZ = 0x2,
  Run2Vertexer3D = 0x4,
  // upper 8 bits for flags
  Run2VertexerTracksWithConstraint = 0x10,
  Run2VertexerTracksOnlyFitter = 0x20,
  Run2VertexerTracksMultiVertex = 0x40
};
} // namespace o2::aod::collision
namespace o2::aod::track
{
enum TrackTypeEnum : uint8_t {
  TrackIU = 0,      // track at point of innermost update (not propagated)
  Track = 1,        // propagated track
  StrangeTrack = 2, // track found by strangeness tracking at point of innermost update
  Run2Track = 254,
  Run2Tracklet = 255
};
enum TrackFlags : uint32_t {
  TrackTimeResIsRange = 0x1, // Gaussian or range
  PVContributor = 0x2,       // This track has contributed to the collision vertex fit
  OrphanTrack = 0x4,         // Track has no association with any collision vertex
  TrackTimeAsym = 0x8,       // track with an asymmetric time range
  // NOTE Highest 4 (29..32) bits reserved for PID hypothesis
};
enum TrackFlagsRun2Enum {
  ITSrefit = 0x1, // for SPD tracklets, 0x1 = tracklet from cluster not used in tracking
  TPCrefit = 0x2,
  GoldenChi2 = 0x4,
  // NOTE Highest 4 (29..32) bits reserved for PID hypothesis
};
enum DetectorMapEnum : uint8_t {
  ITS = 0x1,
  TPC = 0x2,
  TRD = 0x4,
  TOF = 0x8
};
enum TRDTrackPattern : uint8_t {
  Layer0 = 0x1,
  Layer1 = 0x2,
  Layer2 = 0x4,
  Layer3 = 0x8,
  Layer4 = 0x10,
  Layer5 = 0x20,
  HasNeighbor = 0x40,
  HasCrossing = 0x80,
};
namespace extensions
{
struct TPCTimeErrEncoding {
  // TPC delta forward & backward packing
  union TPCDeltaTime {
    struct {
      uint16_t timeForward;
      uint16_t timeBackward;
    } __attribute__((packed)) deltas;
    float timeErr;
  } encoding;
  static_assert(sizeof(float) == 2 * sizeof(uint16_t));

  float getTimeErr() const
  {
    return encoding.timeErr;
  }

  // Use all 16 bits of uint16_t to encode delta scale with max precision
  // e.g., TPCTrack::mDeltaFwd * timeScaler
  // max range for the time deltas is 0 - <512 (1<<9) TPC time bins
  static constexpr float timeScaler{(1 << 16) / (1 << 9)};
  // bogus value to max incorrect usae immedately obvious
  static constexpr float invalidValue{std::numeric_limits<float>::min()};
  // convert TPC time bins to ns
  static constexpr float TPCBinNS = 8 * o2::constants::lhc::LHCBunchSpacingNS;

  void setDeltaTFwd(float fwd)
  {
    encoding.deltas.timeForward = static_cast<uint16_t>(fwd * timeScaler);
  }
  void setDeltaTBwd(float bwd)
  {
    encoding.deltas.timeBackward = static_cast<uint16_t>(bwd * timeScaler);
  }

  float getDeltaTFwd() const
  {
    return static_cast<float>(encoding.deltas.timeForward) / timeScaler * TPCBinNS;
  }
  float getDeltaTBwd() const
  {
    return static_cast<float>(encoding.deltas.timeBackward) / timeScaler * TPCBinNS;
  }
};
} // namespace extensions
} // namespace o2::aod::track

namespace o2::aod::fwdtrack
{
enum ForwardTrackTypeEnum : uint8_t {
  GlobalMuonTrack = 0,       // MFT-MCH-MID
  GlobalMuonTrackOtherMatch, // MFT-MCH-MID (MCH-MID used another time)
  GlobalForwardTrack,        // MFT-MCH
  MuonStandaloneTrack,       // MCH-MID
  MCHStandaloneTrack         // MCH
};
} // namespace o2::aod::fwdtrack

namespace o2::aod::mcparticle::enums
{
enum MCParticleFlags : uint8_t {
  ProducedByTransport = 0x1,
  FromBackgroundEvent = 0x2,          // Particle from background event (may have been used several times)
  PhysicalPrimary = 0x4,              // Particle is a physical primary according to ALICE definition
  FromOutOfBunchPileUpCollision = 0x8 // Particle from out-of-bunch pile up collision (currently Run 2 only)
};
} // namespace o2::aod::mcparticle::enums

namespace o2::aod::run2
{
enum Run2EventSelectionCut {
  kINELgtZERO = 0,
  kPileupInMultBins,
  kConsistencySPDandTrackVertices,
  kTrackletsVsClusters,
  kNonZeroNContribs,
  kIncompleteDAQ,
  kPileUpMV,
  kTPCPileUp,
  kTimeRangeCut,
  kEMCALEDCut,
  kAliEventCutsAccepted,
  kIsPileupFromSPD,
  kIsV0PFPileup,
  kIsTPCHVdip,
  kIsTPCLaserWarmUp,
  kTRDHCO, // Offline TRD cosmic trigger decision
  kTRDHJT, // Offline TRD jet trigger decision
  kTRDHSE, // Offline TRD single electron trigger decision
  kTRDHQU, // Offline TRD quarkonium trigger decision
  kTRDHEE  // Offline TRD single-electron-in-EMCAL-acceptance trigger decision
};
} // namespace o2::aod::run2

#endif // O2_FRAMEWORK_DATATYPES_H_
