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
#include "SimulationDataFormat/TrackReference.h"
#include <array>
#include <vector>

namespace o2::trackstudy
{
struct MCTrackInfo {

  inline float getMCTimeMUS() const { return bcInTF * o2::constants::lhc::LHCBunchSpacingMUS; }
  inline bool hasITSHitOnLr(int i) const { return (pattITSCl & ((0x1 << i) & 0x7f)) != 0; }
  int getNITSClusCont() const;
  int getNITSClusForAB() const;
  int getLowestITSLayer() const;
  int getHighestITSLayer() const;
  std::vector<float> occTPCV{};
  std::vector<o2::track::TrackPar> trackRefsTPC{};
  o2::track::TrackPar track{};
  o2::MCCompLabel label{};
  float occTPC = -1.f;
  int occITS = -1.f;
  int bcInTF = -1;
  int pdg = 0;
  int pdgParent = 0;
  int parentEntry = -1;
  int16_t nTPCCl = 0;
  int16_t nTPCClShared = 0;
  int8_t parentDecID = -1;
  uint8_t minTPCRow = -1;
  uint8_t maxTPCRow = 0;
  uint8_t nUsedPadRows = 0;
  uint8_t maxTPCRowInner = 0; // highest row in the sector containing the lowest one
  uint8_t minTPCRowSect = -1;
  uint8_t maxTPCRowSect = -1;
  int8_t nITSCl = 0;
  int8_t pattITSCl = 0;
  uint8_t flags = 0;

  enum Flags : uint32_t { Primary = 0,
                          AddedAtRecStage = 2,
                          BitMask = 0xff };

  bool isPrimary() const { return isBitSet(Primary); }
  bool isAddedAtRecStage() const { return isBitSet(AddedAtRecStage); }
  void setPrimary() { setBit(Primary); }
  void setAddedAtRecStage() { setBit(AddedAtRecStage); }

  uint8_t getBits() const { return flags; }
  bool isBitSet(int bit) const { return flags & (0xff & (0x1 << bit)); }
  void setBits(std::uint8_t b) { flags = b; }
  void setBit(int bit) { flags |= BitMask & (0x1 << bit); }
  void resetBit(int bit) { flags &= ~(BitMask & (0x1 << bit)); }

  o2::track::TrackPar getTrackParTPC(float b, float x = 90) const;
  float getTrackParTPCPar(int i, float b, float x = 90) const;
  float getTrackParTPCPhiSec(float b, float x = 90) const;

  ClassDefNV(MCTrackInfo, 8);
};

struct RecTrack {
  enum FakeFlag {
    FakeITS = 0x1 << 0,
    FakeTPC = 0x1 << 1,
    FakeTRD = 0x1 << 2,
    FakeTOF = 0x1 << 3,
    FakeITSTPC = 0x1 << 4,
    FakeITSTPCTRD = 0x1 << 5,
    HASACSides = 0x1 << 6,
    FakeGLO = 0x1 << 7
  };
  o2::track::TrackParCov track{};
  o2::dataformats::VtxTrackIndex gid{};
  o2::dataformats::TimeStampWithError<float, float> ts{};
  o2::MCEventLabel pvLabel{};
  short pvID = -1;
  uint8_t nClTPCShared = 0;
  uint8_t flags = 0;
  uint8_t nClITS = 0;
  uint8_t nClTPC = 0;
  uint8_t pattITS = 0;
  int8_t lowestPadRow = -1;
  int8_t padFromEdge = -1;
  uint8_t rowMaxTPC = 0;
  uint8_t rowCountTPC = 0;

  bool isFakeGLO() const { return flags & FakeGLO; }
  bool isFakeITS() const { return flags & FakeITS; }
  bool isFakeTPC() const { return flags & FakeTPC; }
  bool isFakeTRD() const { return flags & FakeTRD; }
  bool isFakeTOF() const { return flags & FakeTOF; }
  bool isFakeITSTPC() const { return flags & FakeITSTPC; }
  bool hasACSides() const { return flags & HASACSides; }

  ClassDefNV(RecTrack, 3);
};

struct TrackPairInfo {
  RecTrack tr0;
  RecTrack tr1;
  uint8_t nshTPC = 0;
  uint8_t nshTPCRow = 0;

  int getComb() const { return tr0.track.getSign() != tr1.track.getSign() ? 0 : (tr0.track.getSign() > 0 ? 1 : 2); }
  float getDPhi() const
  {
    float dphi = tr0.track.getPhi() - tr1.track.getPhi();
    if (dphi < -o2::constants::math::PI) {
      dphi += o2::constants::math::TwoPI;
    } else if (dphi > o2::constants::math::PI) {
      dphi -= o2::constants::math::TwoPI;
    }
    return dphi;
  }
  float getDTgl() const { return tr0.track.getTgl() - tr1.track.getTgl(); }

  ClassDefNV(TrackPairInfo, 1)
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
  const RecTrack& getLongestTPCTrack() const
  {
    int n = getLongestTPCTrackEntry();
    return n < 0 ? dummyRecTrack : recTracks[n];
  }
  int getLongestTPCTrackEntry() const;
  int getNTPCClones() const;
  static RecTrack dummyRecTrack; //

  ClassDefNV(TrackFamily, 1);
};

struct ClResTPCCont {
  // contributor to TPC Cluster
  std::array<float, 3> xyz{};
  std::array<float, 3> below{};
  std::array<float, 3> above{};
  float snp = 0.;
  float tgl = 0.;
  float q2pt = 0.;
  bool corrAttach = false;

  int getNExt() const { return (below[0] > 1.) + (above[0] > 1.); }

  float getClX() const { return xyz[0]; }
  float getClY() const { return xyz[1]; }
  float getClZ() const { return xyz[2]; }

  float getDY() const { return xyz[1] - getYRef(); }
  float getDZ() const { return xyz[2] - getZRef(); }

  float getYRef() const
  {
    float y = 0;
    int n = 0;
    if (below[0] > 1.) {
      y += below[1];
      n++;
    }
    if (above[0] > 1.) {
      y += above[1];
      n++;
    }
    return n == 1 ? y : 0.5 * y;
  }

  float getZRef() const
  {
    float z = 0;
    int n = 0;
    if (below[0] > 1.) {
      z += below[2];
      n++;
    }
    if (above[0] > 1.) {
      z += above[2];
      n++;
    }
    return n == 1 ? z : 0.5 * z;
  }

  float getDXMin() const
  {
    float adxA = 1e9, adxB = 1e9;
    if (above[0] > 1.) {
      adxA = xyz[0] - above[0];
    }
    if (below[0] > 1.) {
      adxB = xyz[1] - below[0];
    }
    return std::abs(adxA) < std::abs(adxB) ? adxA : adxB;
  }

  float getDXMax() const
  {
    float adxA = 0, adxB = 0;
    if (above[0] > 1.) {
      adxA = xyz[0] - above[0];
    }
    if (below[0] > 1.) {
      adxB = xyz[0] - below[0];
    }
    return std::abs(adxA) > std::abs(adxB) ? adxA : adxB;
  }

  float getEY() const { return getNExt() > 1 ? below[1] - above[1] : -999; }
  float getEZ() const { return getNExt() > 1 ? below[2] - above[2] : -999; }

  ClassDefNV(ClResTPCCont, 1);
};

struct ClResTPC {
  uint8_t sect = 0;
  uint8_t row = 0;
  uint8_t ncont = 0;
  uint8_t flags = 0;
  uint8_t sigmaTimePacked;
  uint8_t sigmaPadPacked;
  float qmax = 0;
  float qtot = 0;
  float occ = 0;
  float occBin = 0;
  float getSigmaPad() const { return float(sigmaPadPacked) * (1.f / 32); }
  float getSigmaTime() const { return float(sigmaTimePacked) * (1.f / 32); }

  std::vector<ClResTPCCont> contTracks;
  int getNCont() const { return contTracks.size(); }

  float getDY(int i) const { return i < getNCont() ? contTracks[i].getDY() : -999.; }
  float getDZ(int i) const { return i < getNCont() ? contTracks[i].getDZ() : -999.; }
  float getYRef(int i) const { return i < getNCont() ? contTracks[i].getYRef() : -999.; }
  float getZRef(int i) const { return i < getNCont() ? contTracks[i].getZRef() : -999.; }
  float getDXMin(int i) const { return i < getNCont() ? contTracks[i].getDXMin() : -999.; }
  float getDXMax(int i) const { return i < getNCont() ? contTracks[i].getDXMax() : -999.; }
  float getEY(int i) const { return i < getNCont() ? contTracks[i].getEY() : -999.; }
  float getEZ(int i) const { return i < getNCont() ? contTracks[i].getEZ() : -999.; }

  void sortCont()
  {
    std::sort(contTracks.begin(), contTracks.end(), [](const ClResTPCCont& a, const ClResTPCCont& b) {
      float dya = a.getDY(), dyb = b.getDY(), dza = a.getDZ(), dzb = b.getDZ();
      return dya * dya + dza * dza < dyb * dyb + dzb * dzb;
    });
  }

  ClassDefNV(ClResTPC, 2);
};

struct ITSHitInfo {
  o2::BaseCluster<float> clus{};
  o2::TrackReference tref{};
  float trefXT = 0; // track ref tracking frame coordinates
  float trefYT = 0;
  float chipX = 0;
  float chipAlpha = 0;
  ClassDefNV(ITSHitInfo, 1);
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
  std::vector<float> occTPCV{};
  ClassDefNV(MCVertex, 2);
};

} // namespace o2::trackstudy
#endif
