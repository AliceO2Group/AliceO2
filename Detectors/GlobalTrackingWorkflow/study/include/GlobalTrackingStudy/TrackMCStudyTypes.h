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

#ifndef O2_TRACKING_STUDY_TYPES_H
#define O2_TRACKING_STUDY_TYPES_H
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/Track.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include <array>

namespace o2::trackstudy
{
struct MCTrackInfo {

  inline float getMCTimeMUS() const { return bcInTF * o2::constants::lhc::LHCBunchSpacingMUS; }
  inline bool hasITSHitOnLr(int i) const { return (pattITSCl & ((0x1 << i) & 0x7f)) != 0; }
  int getNITSClusCont() const;
  int getNITSClusForAB() const;
  int getLowestITSLayer() const;
  int getHighestITSLayer() const;

  o2::track::TrackPar track{};
  o2::MCCompLabel label{};
  float occTPC = -1.f;
  int occITS = -1.f;
  int bcInTF = -1;
  int pdg = 0;
  int pdgParent = 0;
  int16_t nTPCCl = 0;
  int16_t nTPCClShared = 0;
  uint8_t minTPCRow = -1;
  uint8_t maxTPCRow = 0;
  int8_t nITSCl = 0;
  int8_t pattITSCl = 0;
  ClassDefNV(MCTrackInfo, 1);
};

struct RecTrack {
  enum FakeFlag {
    FakeITS = 0x1 << 0,
    FakeTPC = 0x1 << 1,
    FakeTRD = 0x1 << 2,
    FakeTOF = 0x1 << 3,
    FakeITSTPC = 0x1 << 4,
    FakeITSTPCTRD = 0x1 << 5,
    FakeGLO = 0x1 << 7
  };
  o2::track::TrackParCov track{};
  o2::dataformats::VtxTrackIndex gid{};
  o2::dataformats::TimeStampWithError<float, float> ts{};
  o2::MCEventLabel pvLabel{};
  short pvID = -1;
  uint8_t flags = 0;
  uint8_t nClITS = 0;
  uint8_t nClTPC = 0;
  uint8_t pattITS = 0;
  int8_t lowestPadRow = -1;

  bool isFakeGLO() const { return flags & FakeGLO; }
  bool isFakeITS() const { return flags & FakeITS; }
  bool isFakeTPC() const { return flags & FakeTPC; }
  bool isFakeTRD() const { return flags & FakeTRD; }
  bool isFakeTOF() const { return flags & FakeTOF; }
  bool isFakeITSTPC() const { return flags & FakeITSTPC; }

  ClassDefNV(RecTrack, 1);
};

struct TrackFamily { // set of tracks related to the same MC label
  MCTrackInfo mcTrackInfo{};
  std::vector<RecTrack> recTracks{};
  o2::track::TrackParCov trackITSProp{};
  o2::track::TrackParCov trackTPCProp{};
  int8_t entITS = -1;
  int8_t entTPC = -1;
  int8_t entITSTPC = -1;
  int8_t entITSFound = -1; // ITS track for this MC track, regardless if it was matched to TPC of another track
  int8_t flags = 0;
  float tpcT0 = -999.;

  bool contains(const o2::dataformats::VtxTrackIndex& ref) const
  {
    for (const auto& tr : recTracks) {
      if (ref == tr.gid) {
        return true;
      }
    }
    return false;
  }
  const RecTrack& getTrackWithITS() const { return entITS < 0 ? dummyRecTrack : recTracks[entITS]; }
  const RecTrack& getTrackWithTPC() const { return entTPC < 0 ? dummyRecTrack : recTracks[entTPC]; }
  const RecTrack& getTrackWithITSTPC() const { return entITSTPC < 0 ? dummyRecTrack : recTracks[entITSTPC]; }
  const RecTrack& getTrackWithITSFound() const { return entITSFound < 0 ? dummyRecTrack : recTracks[entITSFound]; }
  static RecTrack dummyRecTrack; //

  ClassDefNV(TrackFamily, 1);
};

struct RecPV {
  o2::dataformats::PrimaryVertex pv{};
  o2::MCEventLabel mcEvLbl{};
  ClassDefNV(RecPV, 1);
};

struct MCVertex {
  float getX() const { return pos[0]; }
  float getY() const { return pos[1]; }
  float getZ() const { return pos[2]; }

  std::array<float, 3> pos{0., 0., -1999.f};
  float ts = 0;
  int nTrackSel = 0; // number of selected MC charged tracks
  int ID = -1;
  std::vector<RecPV> recVtx{};
  ClassDefNV(MCVertex, 1);
};

} // namespace o2::trackstudy
#endif
