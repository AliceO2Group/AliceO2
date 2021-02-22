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
/// \brief Result of top-bottom cosmic tracks leg matching
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_TRACKCOSMICS_H
#define ALICEO2_TRACKCOSMICS_H

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace dataformats
{

class TrackCosmics : public o2::track::TrackParCov
{
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;

 public:
  TrackCosmics() = default;
  ~TrackCosmics() = default;
  TrackCosmics(const TrackCosmics& src) = default;
  TrackCosmics(GlobalTrackID btm, GlobalTrackID top, const o2::track::TrackParCov& srcCent, const o2::track::TrackParCov& srcOut, float chi2Ref, float chi2Match, int ncl)
    : o2::track::TrackParCov(srcCent), mParamOut(srcOut), mRefBottom(btm), mRefTop(top), mChi2Refit(chi2Ref), mChi2Match(chi2Match), mNClusters(ncl) {}

  GlobalTrackID getRefBottom() const { return mRefBottom; }
  GlobalTrackID getRefTop() const { return mRefTop; }
  void setRefBottom(GlobalTrackID id) { mRefBottom = id; }
  void setRefTop(GlobalTrackID id) { mRefTop = id; }

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

  int getNClusters() const { return mNClusters; }
  void setNClusters(int n) { mNClusters = n; }

  o2::track::TrackParCov& getParamOut() { return mParamOut; }
  const o2::track::TrackParCov& getParamOut() const { return mParamOut; }

  void print() const;

 private:
  GlobalTrackID mRefBottom;         ///< reference on Bottom leg
  GlobalTrackID mRefTop;            ///< reference on Top leg
  float mChi2Refit = 0.f;           ///< chi2 of the global refit
  float mChi2Match = 0.f;           ///< chi2 of the top/bottom match
  int mNClusters = 0;               ///< total number of fitted clusters
  timeEst mTimeMUS;                 ///< time estimate in ns
  o2::track::TrackParCov mParamOut; ///< refitted outer parameter

  ClassDefNV(TrackCosmics, 1);
};
} // namespace dataformats
} // namespace o2

#endif
