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
  using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
  using evIdx = o2::dataformats::EvIndex<int, int>;

 public:
  enum TrackType : int8_t { UNCONS = 0,
                            CONSTR,
                            SIZE,
                            TPC = 0,
                            ITSTPC,
                            TPCTRD,
                            ITSTPCTRD,
                            SIZEALL };

  MatchInfoTOFReco(evIdx evIdxTOFCl, float chi2, o2::track::TrackLTIntegral trkIntLT, evGIdx evIdxTrack, TrackType trkType, float dt = 0, float z = 0) : MatchInfoTOF(evIdxTOFCl, chi2, trkIntLT, evIdxTrack, dt, z), mTrackType(trkType){};

  MatchInfoTOFReco() = default;

  void setTrackType(TrackType value) { mTrackType = value; }
  TrackType getTrackType() const { return mTrackType; }

 private:
  TrackType mTrackType; ///< track type (TPC, ITSTPC, TPCTRD, ITSTPCTRD)
  ClassDefNV(MatchInfoTOFReco, 1);
};
} // namespace dataformats
} // namespace o2
#endif
