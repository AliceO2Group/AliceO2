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
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace dataformats
{

class TrackTPCITS : public o2::track::TrackParCov
{
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;

 public:
  GPUdDefault() TrackTPCITS() = default;
  GPUdDefault() ~TrackTPCITS() = default;
  GPUdDefault() TrackTPCITS(const TrackTPCITS& src) = default;
  GPUd() TrackTPCITS(const o2::track::TrackParCov& src) : o2::track::TrackParCov(src) {}
  GPUd() TrackTPCITS(const o2::track::TrackParCov& srcIn, const o2::track::TrackParCov& srcOut) : o2::track::TrackParCov(srcIn), mParamOut(srcOut) {}

  GPUd() GlobalTrackID getRefTPC() const { return mRefTPC; }
  GPUd() GlobalTrackID getRefITS() const { return mRefITS; }
  GPUd() void setRefTPC(GlobalTrackID id) { mRefTPC = id; }
  GPUd() void setRefITS(GlobalTrackID id) { mRefITS = id; }

  GPUd() const timeEst& getTimeMUS() const { return mTimeMUS; }
  GPUd() timeEst& getTimeMUS() { return mTimeMUS; }
  GPUd() void setTimeMUS(const timeEst& t) { mTimeMUS = t; }
  GPUd() void setTimeMUS(float t, float te)
  {
    mTimeMUS.setTimeStamp(t);
    mTimeMUS.setTimeStampError(te);
  }

  GPUd() void setChi2Refit(float v) { mChi2Refit = v; }
  GPUd() float getChi2Refit() const { return mChi2Refit; }

  GPUd() void setChi2Match(float v) { mChi2Match = v; }
  GPUd() float getChi2Match() const { return mChi2Match; }

  GPUd() o2::track::TrackParCov& getParamOut() { return mParamOut; }
  GPUd() const o2::track::TrackParCov& getParamOut() const { return mParamOut; }

  GPUd() o2::track::TrackLTIntegral& getLTIntegralOut() { return mLTOut; }
  GPUd() const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mLTOut; }

  void print() const;

 private:
  GlobalTrackID mRefTPC;             ///< reference on ITS track entry in its original container
  GlobalTrackID mRefITS;             ///< reference on TPC track entry in its original container
  float mChi2Refit = 0.f;            ///< chi2 of the refit
  float mChi2Match = 0.f;            ///< chi2 of the match
  timeEst mTimeMUS;                  ///< time estimate in ns
  o2::track::TrackParCov mParamOut;  ///< refitted outer parameter
  o2::track::TrackLTIntegral mLTOut; ///< L,TOF integral calculated during the outward refit
  ClassDefNV(TrackTPCITS, 3);
};
} // namespace dataformats
} // namespace o2

#endif
