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

/// \file TrackTPCTOF.h
/// \brief Result of refitting TPC with TOF match constraint
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_TRACKTPCTOF_H
#define ALICEO2_TRACKTPCTOF_H

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace dataformats
{

class TrackTPCTOF : public o2::track::TrackParCov
{
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;

 public:
  TrackTPCTOF() = default;
  ~TrackTPCTOF() = default;
  TrackTPCTOF(const TrackTPCTOF& src) = default;
  TrackTPCTOF(const o2::track::TrackParCov& src) : o2::track::TrackParCov(src) {}

  int getRefMatch() const { return mRefMatch; }
  void setRefMatch(int id) { mRefMatch = id; }

  const timeEst& getTimeMUS() const { return mTimeMUS; }
  timeEst& getTimeMUS() { return mTimeMUS; }
  void setTimeMUS(const timeEst& t) { mTimeMUS = t; }
  void setTimeMUS(float t, float te)
  {
    mTimeMUS.setTimeStamp(t);
    mTimeMUS.setTimeStampError(te);
  }

  void setChi2Refit(float v) { mChi2Refit = v; }
  float getChi2Refit() const { return mChi2Refit; }

  o2::track::TrackParCov& getParamOut() { return mParamOut; }
  const o2::track::TrackParCov& getParamOut() const { return mParamOut; }
  void setParamOut(const o2::track::TrackParCov& v) { mParamOut = v; }

  void print() const;

 private:
  o2::track::TrackParCov mParamOut; // outward fit (in TPC at the moment)
  int mRefMatch = -1;     ///< reference on track-TOF match in its original container
  float mChi2Refit = 0.f; ///< chi2 of the refit
  timeEst mTimeMUS;       ///< time estimate in ns

  ClassDefNV(TrackTPCTOF, 2);
};
} // namespace dataformats
} // namespace o2

#endif
