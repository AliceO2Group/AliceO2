// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackParam.cxx
/// \brief Implementation of the MCH track parameters for internal use
///
/// \author Philippe Pillot, Subatech

#include "TrackParam.h"

#include <iomanip>
#include <iostream>

#include <TMath.h>

#include <FairMQLogger.h>

#include "Cluster.h"

namespace o2
{
namespace mch
{

using namespace std;

//_________________________________________________________________________
TrackParam::TrackParam(const TrackParam& tp)
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
TrackParam& TrackParam::operator=(const TrackParam& tp)
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
void TrackParam::clear()
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
Double_t TrackParam::px() const
{
  /// return p_x from track parameters
  Double_t pZ;
  if (TMath::Abs(mParameters(4, 0)) > 0) {
    Double_t pYZ = (TMath::Abs(mParameters(4, 0)) > 0) ? TMath::Abs(1.0 / mParameters(4, 0)) : FLT_MAX;
    pZ = -pYZ / (TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0))); // spectro. (z<0)
  } else {
    pZ = -FLT_MAX / TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0) + mParameters(1, 0) * mParameters(1, 0));
  }
  return pZ * mParameters(1, 0);
}

//__________________________________________________________________________
Double_t TrackParam::py() const
{
  /// return p_y from track parameters
  Double_t pZ;
  if (TMath::Abs(mParameters(4, 0)) > 0) {
    Double_t pYZ = (TMath::Abs(mParameters(4, 0)) > 0) ? TMath::Abs(1.0 / mParameters(4, 0)) : FLT_MAX;
    pZ = -pYZ / (TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0))); // spectro. (z<0)
  } else {
    pZ = -FLT_MAX / TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0) + mParameters(1, 0) * mParameters(1, 0));
  }
  return pZ * mParameters(3, 0);
}

//__________________________________________________________________________
Double_t TrackParam::pz() const
{
  /// return p_z from track parameters
  if (TMath::Abs(mParameters(4, 0)) > 0) {
    Double_t pYZ = TMath::Abs(1.0 / mParameters(4, 0));
    return -pYZ / (TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0))); // spectro. (z<0)
  } else
    return -FLT_MAX / TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0) + mParameters(1, 0) * mParameters(1, 0));
}

//__________________________________________________________________________
Double_t TrackParam::p() const
{
  /// return p from track parameters
  if (TMath::Abs(mParameters(4, 0)) > 0) {
    Double_t pYZ = TMath::Abs(1.0 / mParameters(4, 0));
    Double_t pZ = -pYZ / (TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0))); // spectro. (z<0)
    return -pZ * TMath::Sqrt(1.0 + mParameters(3, 0) * mParameters(3, 0) + mParameters(1, 0) * mParameters(1, 0));
  } else
    return FLT_MAX;
}

//__________________________________________________________________________
const TMatrixD& TrackParam::getCovariances() const
{
  /// Return the covariance matrix (create it before if needed)
  if (!mCovariances) {
    mCovariances = std::make_unique<TMatrixD>(5, 5);
    mCovariances->Zero();
  }
  return *mCovariances;
}

//__________________________________________________________________________
void TrackParam::setCovariances(const TMatrixD& covariances)
{
  /// Set the covariance matrix
  if (mCovariances)
    *mCovariances = covariances;
  else
    mCovariances = std::make_unique<TMatrixD>(covariances);
}

//__________________________________________________________________________
void TrackParam::setCovariances(const Double_t matrix[5][5])
{
  /// Set the covariance matrix
  if (mCovariances)
    mCovariances->SetMatrixArray(&(matrix[0][0]));
  else
    mCovariances = std::make_unique<TMatrixD>(5, 5, &(matrix[0][0]));
}

//__________________________________________________________________________
void TrackParam::setVariances(const Double_t matrix[5][5])
{
  /// Set the diagonal terms of the covariance matrix (variances)
  if (!mCovariances)
    mCovariances = std::make_unique<TMatrixD>(5, 5);
  mCovariances->Zero();
  for (Int_t i = 0; i < 5; i++)
    (*mCovariances)(i, i) = matrix[i][i];
}

//__________________________________________________________________________
void TrackParam::deleteCovariances()
{
  /// Delete the covariance matrix
  mCovariances.reset();
}

//__________________________________________________________________________
const TMatrixD& TrackParam::getPropagator() const
{
  /// Return the propagator (create it before if needed)
  if (!mPropagator) {
    mPropagator = std::make_unique<TMatrixD>(5, 5);
    mPropagator->UnitMatrix();
  }
  return *mPropagator;
}

//__________________________________________________________________________
void TrackParam::resetPropagator()
{
  /// Reset the propagator
  if (mPropagator)
    mPropagator->UnitMatrix();
}

//__________________________________________________________________________
void TrackParam::updatePropagator(const TMatrixD& propagator)
{
  /// Update the propagator
  if (mPropagator)
    *mPropagator = TMatrixD(propagator, TMatrixD::kMult, *mPropagator);
  else
    mPropagator = std::make_unique<TMatrixD>(propagator);
}

//__________________________________________________________________________
const TMatrixD& TrackParam::getExtrapParameters() const
{
  /// Return extrapolated parameters (create it before if needed)
  if (!mExtrapParameters) {
    mExtrapParameters = std::make_unique<TMatrixD>(5, 1);
    mExtrapParameters->Zero();
  }
  return *mExtrapParameters;
}

//__________________________________________________________________________
void TrackParam::setExtrapParameters(const TMatrixD& extrapParameters)
{
  /// Set extrapolated parameters
  if (mExtrapParameters)
    *mExtrapParameters = extrapParameters;
  else
    mExtrapParameters = std::make_unique<TMatrixD>(extrapParameters);
}

//__________________________________________________________________________
const TMatrixD& TrackParam::getExtrapCovariances() const
{
  /// Return the extrapolated covariance matrix (create it before if needed)
  if (!mExtrapCovariances) {
    mExtrapCovariances = std::make_unique<TMatrixD>(5, 5);
    mExtrapCovariances->Zero();
  }
  return *mExtrapCovariances;
}

//__________________________________________________________________________
void TrackParam::setExtrapCovariances(const TMatrixD& extrapCovariances)
{
  /// Set the extrapolated covariance matrix
  if (mExtrapCovariances)
    *mExtrapCovariances = extrapCovariances;
  else
    mExtrapCovariances = std::make_unique<TMatrixD>(extrapCovariances);
}

//__________________________________________________________________________
const TMatrixD& TrackParam::getSmoothParameters() const
{
  /// Return the smoothed parameters (create it before if needed)
  if (!mSmoothParameters) {
    mSmoothParameters = std::make_unique<TMatrixD>(5, 1);
    mSmoothParameters->Zero();
  }
  return *mSmoothParameters;
}

//__________________________________________________________________________
void TrackParam::setSmoothParameters(const TMatrixD& smoothParameters)
{
  /// Set the smoothed parameters
  if (mSmoothParameters)
    *mSmoothParameters = smoothParameters;
  else
    mSmoothParameters = std::make_unique<TMatrixD>(smoothParameters);
}

//__________________________________________________________________________
const TMatrixD& TrackParam::getSmoothCovariances() const
{
  /// Return the smoothed covariance matrix (create it before if needed)
  if (!mSmoothCovariances) {
    mSmoothCovariances = std::make_unique<TMatrixD>(5, 5);
    mSmoothCovariances->Zero();
  }
  return *mSmoothCovariances;
}

//__________________________________________________________________________
void TrackParam::setSmoothCovariances(const TMatrixD& smoothCovariances)
{
  /// Set the smoothed covariance matrix
  if (mSmoothCovariances)
    *mSmoothCovariances = smoothCovariances;
  else
    mSmoothCovariances = std::make_unique<TMatrixD>(smoothCovariances);
}

//__________________________________________________________________________
Bool_t TrackParam::isCompatibleTrackParam(const TrackParam& trackParam, Double_t sigma2Cut, Double_t& chi2) const
{
  /// Return kTRUE if the two set of track parameters are compatible within sigma2Cut
  /// Set chi2 to the compatible chi2 value
  /// Note that parameter covariances must exist for at least one set of parameters
  /// Note also that if parameters are not given at the same Z, results will be meaningless

  // reset chi2 value
  chi2 = 0.;

  // ckeck covariance matrices
  if (!mCovariances && !trackParam.mCovariances) {
    LOG(ERROR) << "Covariance matrix must exist for at least one set of parameters";
    return kFALSE;
  }

  Double_t maxChi2 = 5. * sigma2Cut * sigma2Cut; // 5 degrees of freedom

  // check Z parameters
  if (mZ != trackParam.mZ)
    LOG(WARN) << "Parameters are given at different Z position (" << mZ << " : " << trackParam.mZ
              << "): results are meaningless";

  // compute the parameter residuals
  TMatrixD deltaParam(mParameters, TMatrixD::kMinus, trackParam.mParameters);

  // build the error matrix
  TMatrixD weight(5, 5);
  if (mCovariances)
    weight += *mCovariances;
  if (trackParam.mCovariances)
    weight += *(trackParam.mCovariances);

  // invert the error matrix to get the parameter weights if possible
  if (weight.Determinant() == 0) {
    LOG(ERROR) << "Cannot compute the compatibility chi2";
    return kFALSE;
  }
  weight.Invert();

  // compute the compatibility chi2
  TMatrixD tmp(deltaParam, TMatrixD::kTransposeMult, weight);
  TMatrixD mChi2(tmp, TMatrixD::kMult, deltaParam);

  // set chi2 value
  chi2 = mChi2(0, 0);

  // check compatibility
  if (chi2 > maxChi2)
    return kFALSE;

  return kTRUE;
}

//__________________________________________________________________________
TrackParamStruct TrackParam::getTrackParamStruct() const
{
  /// return track parameters in the flat structure

  TrackParamStruct param{};

  param.x = getNonBendingCoor();
  param.y = getBendingCoor();
  param.z = getZ();
  param.px = px();
  param.py = py();
  param.pz = pz();
  param.sign = getCharge();

  return param;
}

//__________________________________________________________________________
void TrackParam::print() const
{
  /// Printing TrackParam informations
  cout << "<TrackParam> Bending P=" << setw(5) << setprecision(3) << 1. / mParameters(4, 0)
       << ", NonBendSlope=" << setw(5) << setprecision(3) << mParameters(1, 0) * 180. / TMath::Pi()
       << ", BendSlope=" << setw(5) << setprecision(3) << mParameters(3, 0) * 180. / TMath::Pi() << ", (x,y,z)_IP=("
       << setw(5) << setprecision(3) << mParameters(0, 0) << "," << setw(5) << setprecision(3) << mParameters(2, 0)
       << "," << setw(5) << setprecision(3) << mZ << ") cm, (px,py,pz)=(" << setw(5) << setprecision(3) << px() << ","
       << setw(5) << setprecision(3) << py() << "," << setw(5) << setprecision(3) << pz() << ") GeV/c, "
       << "local chi2=" << getLocalChi2() << endl;
}

} // namespace mch
} // namespace o2
