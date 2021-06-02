// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATATYPES_H_
#define O2_FRAMEWORK_DATATYPES_H_

#include <cstdint>

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
  Track = 0,
  ITSStandaloneTrack,
  Run2Track = 254,
  Run2Tracklet = 255
};
enum TrackFlagsRun2Enum {
  ITSrefit = 0x1,
  TPCrefit = 0x2,
  GoldenChi2 = 0x4
};
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
