// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackMFT.cxx
/// \brief Implementation of the MFT track
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#include "DataFormatsMFT/TrackMFT.h"
#include "CommonConstants/MathConstants.h"
#include "DataFormatsITSMFT/Cluster.h"
#include <TMatrixD.h>
#include <TMath.h>

using namespace o2::mft;
using namespace o2::itsmft;
using namespace o2::constants::math;

namespace o2
{
namespace mft
{

//__________________________________________________________________________
TrackMFT::TrackMFT(const Double_t z, const TMatrixD parameters, const TMatrixD covariances, const Double_t chi2)
{
  mZ = z;
  mParameters = parameters;
  mCovariances = covariances;
  mTrackChi2 = chi2;
}

//__________________________________________________________________________
const TMatrixD& TrackMFT::getCovariances() const
{
  /// Return the covariance matrix
  return mCovariances;
}

//__________________________________________________________________________
void TrackMFT::setCovariances(const TMatrixD& covariances)
{
  mCovariances = covariances;
}

//__________________________________________________________________________
void TrackMFT::setCovariances(const Double_t matrix[5][5])
{
  mCovariances = TMatrixD(5, 5, &(matrix[0][0]));
}

} // namespace mft
} // namespace o2
