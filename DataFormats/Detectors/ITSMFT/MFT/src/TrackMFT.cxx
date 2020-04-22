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
#include "MathUtils/Utils.h"
#include <TMath.h>

using namespace o2::mft;
using namespace o2::itsmft;
using namespace o2::constants::math;

namespace o2
{
namespace mft
{

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

//__________________________________________________________________________
TrackMFT::TrackMFT(const Double_t z, const SMatrix5 parameters, const SMatrix55 covariances, const Double_t chi2)
{
  mZ = z;
  mParameters = parameters;
  mCovariances = covariances;
  mTrackChi2 = chi2;
}

//__________________________________________________________________________
const SMatrix55& TrackMFT::getCovariances() const
{
  /// Return the covariance matrix
  return mCovariances;
}

//__________________________________________________________________________
void TrackMFT::setCovariances(const SMatrix55& covariances)
{
  mCovariances = covariances;
}

//_________________________________________________________________________________________________
void TrackMFT::extrapHelixToZ(double zEnd, double Field)
{
  /// Track extrapolated to the plane at "zEnd" considering a helix

  if (getZ() == zEnd) {
    return; // nothing to be done if same z
  }

  // Extrapolate covariances
  extrapHelixToZCov(zEnd, Field);

  // Extrapolate track parameters
  double dZ = (zEnd - getZ()); // Propagate in meters
  double cosphi0, sinphi0;
  o2::utils::sincos(getPhi(), sinphi0, cosphi0);
  double tanl0 = getTanl();
  double invtanl0 = 1.0 / tanl0;
  double invqpt0 = getInvQPt();

  double k = Field * o2::constants::math::B2C;
  double deltax = dZ * cosphi0 * invtanl0 - 0.5 * dZ * dZ * k * invqpt0 * sinphi0 * invtanl0 * invtanl0;
  double deltay = dZ * sinphi0 * invtanl0 + 0.5 * dZ * dZ * k * invqpt0 * cosphi0 * invtanl0 * invtanl0;

  double x = getX() + deltax;
  double y = getY() + deltay;
  double deltaphi = +dZ * k * invqpt0 * invtanl0;

  double phi = getPhi() + deltaphi;
  double tanl = tanl0;
  double invqpt = invqpt0;
  setX(x);
  setY(y);
  setZ(zEnd);
  setPhi(phi);
  setTanl(tanl);
  setInvQPt(invqpt);
}

//__________________________________________________________________________
void TrackMFT::extrapHelixToZCov(double zEnd, double Field)
{

  // Calculate the jacobian related to the track parameters helix extrapolation to "zEnd"
  double dZ = (zEnd - getZ());
  double phi0 = getPhi();
  double tanl0 = getTanl();
  double invqpt0 = getInvQPt();
  double dZ2 = dZ * dZ;
  double cosphi0, sinphi0;
  o2::utils::sincos(phi0, sinphi0, cosphi0);
  double tanl0sq = tanl0 * tanl0;
  double k = Field * o2::constants::math::B2C;
  /*
  TMatrixD jacob(5, 5);
  jacob.UnitMatrix();
  jacob(0, 2) = -dZ2 * k * invqpt0 * cosphi0 * 0.5 / tanl0sq - dZ * sinphi0 / tanl0;
  jacob(0, 3) = dZ2 * k * invqpt0 * sinphi0 / tanl0sq / tanl0 - dZ * cosphi0 / tanl0sq;
  jacob(0, 4) = -dZ2 * k * sinphi0 * 0.5 / tanl0sq;
  jacob(1, 2) = -dZ2 * k * invqpt0 * sinphi0 * 0.5 / tanl0sq + dZ * cosphi0 / tanl0;
  jacob(1, 3) = -dZ2 * k * invqpt0 * cosphi0 / tanl0sq / tanl0 - dZ * sinphi0 / tanl0sq;
  jacob(1, 4) = dZ2 * k * cosphi0 * 0.5 / tanl0sq;
  jacob(2, 3) = -dZ * k * invqpt0 / tanl0sq;
  jacob(2, 4) = dZ * k / tanl0;

  // Extrapolate track parameter covariances to "zEnd"
  TMatrixD tmp(getCovariances(), TMatrixD::kMultTranspose, jacob);
  TMatrixD tmp2(jacob, TMatrixD::kMult, tmp);
  setCovariances(tmp2);
*/
}

//__________________________________________________________________________
void TrackMFT::print() const
{
  /// Printing TrackMFT information
  LOG(INFO) << "TrackMFT: p =" << std::setw(5) << std::setprecision(3) << getP()
            << " Tanl = " << std::setw(5) << std::setprecision(3) << getTanl()
            << " phi = " << std::setw(5) << std::setprecision(3) << getPhi()
            << " pz = " << std::setw(5) << std::setprecision(3) << getPz()
            << " pt = " << std::setw(5) << std::setprecision(3) << getPt()
            << " charge = " << std::setw(5) << std::setprecision(3) << getCharge()
            << " chi2 = " << std::setw(5) << std::setprecision(3) << getTrackChi2() << std::endl;
}

//__________________________________________________________________________
void TrackMFT::printMCCompLabels() const
{
  /// Printing TrackMFT MCLabel information
  LOG(INFO) << "TrackMFT with " << mNPoints << " clusters. MCLabels: " << mMCCompLabels[0] << mMCCompLabels[1] << "..."; //<< mMCCompLabels[2] << mMCCompLabels[3] << mMCCompLabels[4] << mMCCompLabels[5] << mMCCompLabels[6] << mMCCompLabels[7] << mMCCompLabels[8] << mMCCompLabels[9];
}

} // namespace mft
} // namespace o2
