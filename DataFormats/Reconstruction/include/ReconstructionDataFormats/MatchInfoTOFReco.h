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

/// \file MatchInfoTOFReco.h
/// \brief Class to temporary store the output of the matching to TOF in reconstruction

#ifndef ALICEO2_MATCHINFOTOFRECO_H
#define ALICEO2_MATCHINFOTOFRECO_H

#include "ReconstructionDataFormats/MatchInfoTOF.h"

namespace o2
{
namespace dataformats
{
class MatchInfoTOFReco : public MatchInfoTOF
{
  using GTrackID = o2::dataformats::GlobalTrackID;

 public:
  enum TrackType : int8_t { UNCONS = 0,
                            CONSTR,
                            SIZE,
                            TPC = 0,
                            ITSTPC,
                            TPCTRD,
                            ITSTPCTRD,
                            SIZEALL };

  MatchInfoTOFReco(int idLocal, int idxTOFCl, double time, float chi2, o2::track::TrackLTIntegral trkIntLT, GTrackID idxTrack, TrackType trkType, float dt = 0, float z = 0, float dx = 0, float dz = 0) : MatchInfoTOF(idLocal, idxTOFCl, time, chi2, trkIntLT, idxTrack, dt, z, dx, dz), mTrackType(trkType){};

  MatchInfoTOFReco() = default;

  void setFakeMatch() { mFakeMC = true; }
  void resetFakeMatch() { mFakeMC = false; }
  bool isFake() const { return mFakeMC; }

  void setTrackType(TrackType value) { mTrackType = value; }
  TrackType getTrackType() const { return mTrackType; }

 private:
  TrackType mTrackType; ///< track type (TPC, ITSTPC, TPCTRD, ITSTPCTRD)
  bool mFakeMC = false;
  ClassDefNV(MatchInfoTOFReco, 3);
};
} // namespace dataformats
} // namespace o2
#endif
