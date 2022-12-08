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

/// \file TrackMCH.cxx
/// \brief Implementation of the MCH track
///
/// \author Philippe Pillot, Subatech

#include "DataFormatsMCH/TrackMCH.h"

#include <cmath>
#include <limits>
#include <fmt/format.h>
#include <string>
#include <iostream>

namespace o2
{
namespace mch
{

//__________________________________________________________________________
TrackMCH::TrackMCH(double z, const TMatrixD& param, const TMatrixD& cov, double chi2, int firstClIdx, int nClusters,
                   double zAtMID, const TMatrixD& paramAtMID, const TMatrixD& covAtMID)
  : mZ(z), mChi2(chi2), mClusRef(firstClIdx, nClusters), mZAtMID(zAtMID)
{
  /// constructor
  setParameters(param);
  setCovariances(cov);
  setParametersAtMID(paramAtMID);
  setCovariancesAtMID(covAtMID);
}

//__________________________________________________________________________
double TrackMCH::getPx() const
{
  /// return track momentum along x
  return getPz() * mParam[1];
}

//__________________________________________________________________________
double TrackMCH::getPy() const
{
  /// return track momentum along y
  return getPz() * mParam[3];
}

//__________________________________________________________________________
double TrackMCH::getPz() const
{
  /// return track momentum along z
  if (mParam[4] != 0.) {
    return -std::abs(1. / mParam[4]) / std::sqrt(1. + mParam[3] * mParam[3]); // spectro. (z<0)
  } else {
    return -std::numeric_limits<float>::max() / std::sqrt(1. + mParam[3] * mParam[3] + mParam[1] * mParam[1]);
  }
}

//__________________________________________________________________________
double TrackMCH::getP() const
{
  /// return track momentum
  if (mParam[4] != 0.) {
    return std::abs(1. / mParam[4]) / std::sqrt(1. + mParam[3] * mParam[3]) *
           std::sqrt(1. + mParam[3] * mParam[3] + mParam[1] * mParam[1]);
  } else {
    return std::numeric_limits<float>::max();
  }
}

//__________________________________________________________________________
void TrackMCH::setCovariances(const TMatrixD& src, double (&dest)[SCovSize])
{
  /// set the track parameter covariances
  for (int i = 0; i < SNParams; i++) {
    for (int j = 0; j <= i; j++) {
      dest[SCovIdx[i][j]] = src(i, j);
    }
  }
}

//__________________________________________________________________________
InteractionRecord TrackMCH::getMeanIR(uint32_t refOrbit) const
{
  InteractionRecord startIR(0, refOrbit);
  auto trackBCinTF = std::llround(mTimeMUS.getTimeStamp() / constants::lhc::LHCBunchSpacingMUS);
  return startIR + trackBCinTF;
}

std::ostream& operator<<(std::ostream& os, const o2::mch::TrackMCH& t)
{
  os << asString(t);
  return os;
}

std::string asString(const o2::mch::TrackMCH& t)
{
  auto pt = std::sqrt(t.getPx() * t.getPx() + t.getPy() * t.getPy());
  return fmt::format("({:s}) p {:7.2f} pt {:7.2f} nclusters: {} z: {:7.2f}", t.getSign() == -1 ? "-" : "+", t.getP(), pt, t.getNClusters(), t.getZ());
}

} // namespace mch
} // namespace o2
