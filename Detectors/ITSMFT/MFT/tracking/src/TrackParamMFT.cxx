// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackParamMFT.cxx
/// \brief Implementation of the MFT track parameters for internal use
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#include "MFTTracking/TrackParamMFT.h"

#include <iomanip>
#include <iostream>

#include <TMath.h>

#include <FairMQLogger.h>

namespace o2
{
namespace mft
{

using namespace std;

//_________________________________________________________________________
TrackParamMFT::TrackParamMFT(const TrackParamMFT& tp)
  : mZ(tp.mZ),
    mParameters(tp.mParameters),
    mClusterPtr(tp.mClusterPtr),
    mRemovable(tp.mRemovable),
    mTrackChi2(tp.mTrackChi2),
    mLocalChi2(tp.mLocalChi2)
{
  /// Copy constructor
  if (tp.mCovariances)
    mCovariances = std::make_unique<TMatrixD>(*(tp.mCovariances));
  if (tp.mPropagator)
    mPropagator = std::make_unique<TMatrixD>(*(tp.mPropagator));
  if (tp.mExtrapParameters)
    mExtrapParameters = std::make_unique<TMatrixD>(*(tp.mExtrapParameters));
  if (tp.mExtrapCovariances)
    mExtrapCovariances = std::make_unique<TMatrixD>(*(tp.mExtrapCovariances));
  if (tp.mSmoothParameters)
    mSmoothParameters = std::make_unique<TMatrixD>(*(tp.mSmoothParameters));
  if (tp.mSmoothCovariances)
    mSmoothCovariances = std::make_unique<TMatrixD>(*(tp.mSmoothCovariances));
}

//_________________________________________________________________________
TrackParamMFT& TrackParamMFT::operator=(const TrackParamMFT& tp)
{
  /// Assignment operator
  if (this == &tp)
    return *this;

  mZ = tp.mZ;

  mParameters = tp.mParameters;

  if (tp.mCovariances) {
    if (mCovariances)
      *mCovariances = *(tp.mCovariances);
    else
      mCovariances = std::make_unique<TMatrixD>(*(tp.mCovariances));
  } else
    mCovariances.reset();

  if (tp.mPropagator) {
    if (mPropagator)
      *mPropagator = *(tp.mPropagator);
    else
      mPropagator = std::make_unique<TMatrixD>(*(tp.mPropagator));
  } else
    mPropagator.reset();

  if (tp.mExtrapParameters) {
    if (mExtrapParameters)
      *mExtrapParameters = *(tp.mExtrapParameters);
    else
      mExtrapParameters = std::make_unique<TMatrixD>(*(tp.mExtrapParameters));
  } else
    mExtrapParameters.reset();

  if (tp.mExtrapCovariances) {
    if (mExtrapCovariances)
      *mExtrapCovariances = *(tp.mExtrapCovariances);
    else
      mExtrapCovariances = std::make_unique<TMatrixD>(*(tp.mExtrapCovariances));
  } else
    mExtrapCovariances.reset();

  if (tp.mSmoothParameters) {
    if (mSmoothParameters)
      *mSmoothParameters = *(tp.mSmoothParameters);
    else
      mSmoothParameters = std::make_unique<TMatrixD>(*(tp.mSmoothParameters));
  } else
    mSmoothParameters.reset();

  if (tp.mSmoothCovariances) {
    if (mSmoothCovariances)
      *mSmoothCovariances = *(tp.mSmoothCovariances);
    else
      mSmoothCovariances = std::make_unique<TMatrixD>(*(tp.mSmoothCovariances));
  } else
    mSmoothCovariances.reset();

  mClusterPtr = tp.mClusterPtr;

  mRemovable = tp.mRemovable;

  mTrackChi2 = tp.mTrackChi2;
  mLocalChi2 = tp.mLocalChi2;

  return *this;
}

//__________________________________________________________________________
void TrackParamMFT::clear()
{
  /// clear memory
  deleteCovariances();
  mPropagator.reset();
  mExtrapParameters.reset();
  mExtrapCovariances.reset();
  mSmoothParameters.reset();
  mSmoothCovariances.reset();
}

//__________________________________________________________________________
const TMatrixD& TrackParamMFT::getCovariances() const
{
  /// Return the covariance matrix (create it before if needed)
  if (!mCovariances) {
    mCovariances = std::make_unique<TMatrixD>(5, 5);
    mCovariances->Zero();
  }
  return *mCovariances;
}

//__________________________________________________________________________
void TrackParamMFT::setCovariances(const TMatrixD& covariances)
{
  /// Set the covariance matrix
  if (mCovariances)
    *mCovariances = covariances;
  else
    mCovariances = std::make_unique<TMatrixD>(covariances);
}

//__________________________________________________________________________
void TrackParamMFT::setCovariances(const Double_t matrix[5][5])
{
  /// Set the covariance matrix
  if (mCovariances)
    mCovariances->SetMatrixArray(&(matrix[0][0]));
  else
    mCovariances = std::make_unique<TMatrixD>(5, 5, &(matrix[0][0]));
}

//__________________________________________________________________________
void TrackParamMFT::setVariances(const Double_t matrix[5][5])
{
  /// Set the diagonal terms of the covariance matrix (variances)
  if (!mCovariances)
    mCovariances = std::make_unique<TMatrixD>(5, 5);
  mCovariances->Zero();
  for (Int_t i = 0; i < 5; i++)
    (*mCovariances)(i, i) = matrix[i][i];
}

//__________________________________________________________________________
void TrackParamMFT::deleteCovariances()
{
  /// Delete the covariance matrix
  mCovariances.reset();
}

//__________________________________________________________________________
const TMatrixD& TrackParamMFT::getPropagator() const
{
  /// Return the propagator (create it before if needed)
  if (!mPropagator) {
    mPropagator = std::make_unique<TMatrixD>(5, 5);
    mPropagator->UnitMatrix();
  }
  return *mPropagator;
}

//__________________________________________________________________________
void TrackParamMFT::resetPropagator()
{
  /// Reset the propagator
  if (mPropagator)
    mPropagator->UnitMatrix();
}

//__________________________________________________________________________
void TrackParamMFT::updatePropagator(const TMatrixD& propagator)
{
  /// Update the propagator
  if (mPropagator)
    *mPropagator = TMatrixD(propagator, TMatrixD::kMult, *mPropagator);
  else
    mPropagator = std::make_unique<TMatrixD>(propagator);
}

//__________________________________________________________________________
const TMatrixD& TrackParamMFT::getExtrapParameters() const
{
  /// Return extrapolated parameters (create it before if needed)
  if (!mExtrapParameters) {
    mExtrapParameters = std::make_unique<TMatrixD>(5, 1);
    mExtrapParameters->Zero();
  }
  return *mExtrapParameters;
}

//__________________________________________________________________________
void TrackParamMFT::setExtrapParameters(const TMatrixD& extrapParameters)
{
  /// Set extrapolated parameters
  if (mExtrapParameters)
    *mExtrapParameters = extrapParameters;
  else
    mExtrapParameters = std::make_unique<TMatrixD>(extrapParameters);
}

//__________________________________________________________________________
const TMatrixD& TrackParamMFT::getExtrapCovariances() const
{
  /// Return the extrapolated covariance matrix (create it before if needed)
  if (!mExtrapCovariances) {
    mExtrapCovariances = std::make_unique<TMatrixD>(5, 5);
    mExtrapCovariances->Zero();
  }
  return *mExtrapCovariances;
}

//__________________________________________________________________________
void TrackParamMFT::setExtrapCovariances(const TMatrixD& extrapCovariances)
{
  /// Set the extrapolated covariance matrix
  if (mExtrapCovariances)
    *mExtrapCovariances = extrapCovariances;
  else
    mExtrapCovariances = std::make_unique<TMatrixD>(extrapCovariances);
}

//__________________________________________________________________________
const TMatrixD& TrackParamMFT::getSmoothParameters() const
{
  /// Return the smoothed parameters (create it before if needed)
  if (!mSmoothParameters) {
    mSmoothParameters = std::make_unique<TMatrixD>(5, 1);
    mSmoothParameters->Zero();
  }
  return *mSmoothParameters;
}

//__________________________________________________________________________
void TrackParamMFT::setSmoothParameters(const TMatrixD& smoothParameters)
{
  /// Set the smoothed parameters
  if (mSmoothParameters) {
    *mSmoothParameters = smoothParameters;
  } else {
    mSmoothParameters = std::make_unique<TMatrixD>(smoothParameters);
  }
}

//__________________________________________________________________________
const TMatrixD& TrackParamMFT::getSmoothCovariances() const
{
  /// Return the smoothed covariance matrix (create it before if needed)
  if (!mSmoothCovariances) {
    mSmoothCovariances = std::make_unique<TMatrixD>(5, 5);
    mSmoothCovariances->Zero();
  }
  return *mSmoothCovariances;
}

//__________________________________________________________________________
void TrackParamMFT::setSmoothCovariances(const TMatrixD& smoothCovariances)
{
  /// Set the smoothed covariance matrix
  if (mSmoothCovariances) {
    *mSmoothCovariances = smoothCovariances;
  } else {
    mSmoothCovariances = std::make_unique<TMatrixD>(smoothCovariances);
  }
}

//__________________________________________________________________________
void TrackParamMFT::print() const
{
  /// Printing TrackParamMFT informations
  LOG(INFO) << "TrackParamMFT: p =" << setw(5) << setprecision(3) << getP()
            << " Tanl = " << setw(5) << setprecision(3) << getTanl()
            << " phi = " << setw(5) << setprecision(3) << getPhi()
            << " pz = " << setw(5) << setprecision(3) << getPz()
            << " pt = " << setw(5) << setprecision(3) << getPt()
            << " charge = " << setw(5) << setprecision(3) << getCharge()
            << " chi2 = " << setw(5) << setprecision(3) << getTrackChi2() << endl;
}

} // namespace mft
} // namespace o2
