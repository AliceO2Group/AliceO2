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
#include "ReconstructionDataFormats/MatchInfoFwd.h"
#include "Math/SMatrix.h"

namespace o2
{
namespace dataformats
{
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;
using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

class GlobalFwdTrack : public o2::track::TrackParCovFwd, public o2::dataformats::MatchInfoFwd
{
 public:
  GlobalFwdTrack() = default;
  GlobalFwdTrack(const GlobalFwdTrack& t) = default;
  GlobalFwdTrack(o2::track::TrackParCovFwd const& t) { *this = t; }
  ~GlobalFwdTrack() = default;

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

 private:
  ClassDefNV(GlobalFwdTrack, 2);
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
