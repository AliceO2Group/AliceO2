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

/// \file GlobalFwdTrack.h
/// \brief Global Forward Muon tracks

#ifndef ALICEO2_TRACKGLOBALFWD_H
#define ALICEO2_TRACKGLOBALFWD_H

#include "ReconstructionDataFormats/TrackFwd.h"
#include "CommonDataFormat/TimeStamp.h"
#include "Math/SMatrix.h"

namespace o2
{
namespace dataformats
{
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;
using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class GlobalFwdTrack : public o2::track::TrackParCovFwd
{
 public:
  GlobalFwdTrack() = default;
  GlobalFwdTrack(const GlobalFwdTrack& t) = default;
  ~GlobalFwdTrack() = default;

  void setMatchingChi2(double chi2) { mMatchingChi2 = chi2; }
  const auto& getMatchingChi2() const { return mMatchingChi2; }

  void setMIDMatchingChi2(double chi2) { mMIDMatchingChi2 = chi2; }
  const auto& getMIDMatchingChi2() const { return mMIDMatchingChi2; }

  void countCandidate() { mNMFTCandidates++; }
  const auto& getNMFTCandidates() const { return mNMFTCandidates; }

  void setCloseMatch() { mCloseMatch = true; }
  const auto& isCloseMatch() const { return mCloseMatch; }

  const timeEst& getTimeMUS() const { return mTimeMUS; }
  timeEst& getTimeMUS() { return mTimeMUS; }
  void setTimeMUS(const timeEst& t) { mTimeMUS = t; }
  void setTimeMUS(float t, float te)
  {
    mTimeMUS.setTimeStamp(t);
    mTimeMUS.setTimeStampError(te);
  }

  SMatrix5 computeResiduals2Cov(const o2::track::TrackParCovFwd& t) const
  {
    SMatrix5 Residuals2Cov;

    Residuals2Cov(0) = (getX() - t.getX()) / TMath::Sqrt(getCovariances()(0, 0) + t.getCovariances()(0, 0));
    Residuals2Cov(1) = (getY() - t.getY()) / TMath::Sqrt(getCovariances()(1, 1) + t.getCovariances()(1, 1));
    Residuals2Cov(2) = (getPhi() - t.getPhi()) / TMath::Sqrt(getCovariances()(2, 2) + t.getCovariances()(2, 2));
    Residuals2Cov(3) = (getTanl() - t.getTanl()) / TMath::Sqrt(getCovariances()(3, 3) + t.getCovariances()(3, 3));
    Residuals2Cov(4) = (getInvQPt() - t.getInvQPt()) / TMath::Sqrt(getCovariances()(4, 4) + t.getCovariances()(4, 4));
    return Residuals2Cov;
  }

  void setMCHTrackID(int ID) { mMCHTrackID = ID; }
  const auto& getMCHTrackID() const { return mMCHTrackID; }
  void setMFTTrackID(int ID) { mMFTTrackID = ID; }
  const auto& getMFTTrackID() const { return mMFTTrackID; }

 private:
  double mMatchingChi2 = 1.0E308; ///< MCH-MFT Matching Chi2
  double mMIDMatchingChi2 = -1.0; ///< MCH-MID Matching Chi2
  int mMFTTrackID = -1;           ///< Track ID of best MFT-match
  int mMCHTrackID = -1;           ///< MCH Track ID
  int mNMFTCandidates = 0;        ///< Number of MFT candidates within search cut
  bool mCloseMatch = false;       ///< Close match = correct MFT pair tested (MC-only)
  float mTrackTime = 0.f;         ///< Track time
  float mTrackTimeRes = 0.f;      ///< Track time resolution
  timeEst mTimeMUS;               ///< time estimate in ns
  ClassDefNV(GlobalFwdTrack, 1);
};

} // namespace dataformats

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dataformats::GlobalFwdTrack> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
