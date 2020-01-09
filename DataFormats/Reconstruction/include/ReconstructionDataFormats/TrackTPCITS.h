// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackTPCITS.h
/// \brief Result of refitting TPC-ITS matched track
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_TRACKTPCITS_H
#define ALICEO2_TRACKTPCITS_H

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace dataformats
{

class TrackTPCITS : public o2::track::TrackParCov
{
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;

 public:
  TrackTPCITS() = default;
  ~TrackTPCITS() = default;
  TrackTPCITS(const TrackTPCITS& src) = default;
  TrackTPCITS(const o2::track::TrackParCov& src) : o2::track::TrackParCov(src) {}
  TrackTPCITS(const o2::track::TrackParCov& srcIn, const o2::track::TrackParCov& srcOut) : o2::track::TrackParCov(srcIn), mParamOut(srcOut) {}

  int getRefTPC() const { return mRefTPC; }
  int getRefITS() const { return mRefITS; }
  void setRefTPC(int id) { mRefTPC = id; }
  void setRefITS(int id) { mRefITS = id; }

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

  void setChi2Match(float v) { mChi2Match = v; }
  float getChi2Match() const { return mChi2Match; }

  o2::track::TrackParCov& getParamOut() { return mParamOut; }
  const o2::track::TrackParCov& getParamOut() const { return mParamOut; }

  o2::track::TrackLTIntegral& getLTIntegralOut() { return mLTOut; }
  const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mLTOut; }

  void print() const;

 private:
  int mRefTPC = -1;                  ///< reference on ITS track entry in its original container
  int mRefITS = -1;                  ///< reference on TPC track entry in its original container
  float mChi2Refit = 0.f;            ///< chi2 of the refit
  float mChi2Match = 0.f;            ///< chi2 of the match
  timeEst mTimeMUS;                  ///< time estimate in ns
  o2::track::TrackParCov mParamOut;  ///< refitted outer parameter
  o2::track::TrackLTIntegral mLTOut; ///< L,TOF integral calculated during the outward refit
  ClassDefNV(TrackTPCITS, 2);
};
} // namespace dataformats
} // namespace o2

#endif
