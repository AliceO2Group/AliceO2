// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompressionTrackModel.cxx
/// \author David Rohr

#include "GPUTPCCompressionTrackModel.h"
#include "GPUConstantMem.h"
#include "GPUParam.inc"

using namespace GPUCA_NAMESPACE::gpu;

// ATTENTION! This track model is used for the data compression.
// Changes to the propagation and fit will prevent the decompression of data
// encoded with the old version!!!

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
GPUd() void GPUTPCCompressionTrackModel::Init(float x, float y, float z, float alpha, unsigned char qPt, const GPUParam& GPUrestrict() param)
{
  static constexpr float kRho = 1.025e-3f;  // 0.9e-3;
  static constexpr float kRadLen = 29.532f; // 28.94;
  mProp.SetMaterial(kRadLen, kRho);
  mProp.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
  mProp.SetToyMCEventsFlag(false);
  mProp.SetSeedingErrors(true); // Larger errors for seeds, better since we don't start with good hypothesis
  mProp.SetFitInProjections(true);
  mProp.SetPropagateBzOnly(true);
  mProp.SetPolynomialField(&param.polynomialField);
  mTrk.X() = x;
  mTrk.Y() = y;
  mTrk.Z() = z;
  mTrk.SinPhi() = 0;
  mTrk.DzDs() = 0;
  mTrk.QPt() = (qPt - 127.f) * (20.f / 127.f);
  mTrk.ResetCovariance();
  mProp.SetTrack(&mTrk, alpha);
  mParam = &param;
  // GPUInfo("Initialized: x %f y %f z %f alpha %f qPt %f", x, y, z, alpha, mTrk.QPt());
}

GPUd() int GPUTPCCompressionTrackModel::Propagate(float x, float alpha)
{
  int retVal = mProp.PropagateToXAlpha(x, alpha, true);
  // GPUInfo("Propagated to: x %f y %f z %f alpha %f qPt %f", x, mTrk.Y(), mTrk.Z(), alpha, mTrk.QPt());
  return retVal;
}

GPUd() int GPUTPCCompressionTrackModel::Filter(float y, float z, int iRow)
{
  mTrk.ConstrainSinPhi();
  int retVal = mProp.Update(y, z, iRow, *mParam, 0, 0, nullptr, false);
  // GPUInfo("Filtered with %f %f: y %f z %f qPt %f", y, z, mTrk.Y(), mTrk.Z(), mTrk.QPt());
  return retVal;
}

GPUd() int GPUTPCCompressionTrackModel::Mirror()
{
  mProp.Mirror(true);
  // GPUInfo("Mirrored: y %f z %f qPt %f", mTrk.Y(), mTrk.Z(), mTrk.QPt());
  return 0;
}

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)

#include "GPUTPCTrackLinearisation.h"
#include "GPUTPCTracker.h"

GPUd() void GPUTPCCompressionTrackModel::Init(float x, float y, float z, float alpha, unsigned char qPt, const GPUParam& GPUrestrict() param)
{
  mTrk.InitParam();
  mTrk.SetX(x);
  mTrk.SetY(y);
  mTrk.SetZ(z);
  mTrk.SetSinPhi(0);
  mTrk.SetDzDs(0);
  mTrk.SetQPt((qPt - 127.f) * (20.f / 127.f));
  mAlpha = alpha;
  mParam = &param;
  // GPUInfo("Initialized: x %f y %f z %f alpha %f qPt %f", x, y, z, alpha, mTrk.QPt());
}

GPUd() int GPUTPCCompressionTrackModel::Propagate(float x, float alpha)
{
  GPUTPCTrackLinearisation t0(mTrk);
  if (alpha != mAlpha && !mTrk.Rotate(alpha, t0, GPUCA_MAX_SIN_PHI)) {
    return 2;
  }
  int retVal = !mTrk.TransportToX(x, t0, mParam->ConstBz, GPUCA_MAX_SIN_PHI);
  // GPUInfo("Propagated to: x %f y %f z %f alpha %f qPt %f", x, mTrk.Y(), mTrk.Z(), alpha, mTrk.QPt());
  return retVal;
}

GPUd() int GPUTPCCompressionTrackModel::Filter(float y, float z, int iRow)
{
  mTrk.ConstrainSinPhi();
  float err2Y, err2Z;
  GPUTPCTracker::GetErrors2Seeding(*mParam, iRow, mTrk, err2Y, err2Z);
  int retVal = !mTrk.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI, false);
  // GPUInfo("Filtered with %f %f: y %f z %f qPt %f", y, z, mTrk.Y(), mTrk.Z(), mTrk.QPt());
  return retVal;
}

GPUd() int GPUTPCCompressionTrackModel::Mirror()
{
  return 1;
}

#else // Default internal track model for compression

#endif
