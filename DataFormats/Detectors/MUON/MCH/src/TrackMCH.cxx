// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace mch
{

//__________________________________________________________________________
TrackMCH::TrackMCH(double z, const TMatrixD& param, const TMatrixD& cov, double chi2, int firstClIdx, int nClusters)
  : mZ(z), mChi2(chi2), mClusRef(firstClIdx, nClusters)
{
  /// constructor
  setParameters(param);
  setCovariances(cov);
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
void TrackMCH::setCovariances(const TMatrixD& cov)
{
  /// set the track parameter covariances
  for (int i = 0; i < SNParams; i++) {
    for (int j = 0; j <= i; j++) {
      mCov[SCovIdx[i][j]] = cov(i, j);
    }
  }
}

} // namespace mch
} // namespace o2
