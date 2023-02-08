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
  mProp.SetMaterialTPC();
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
  int retVal = !mTrk.TransportToX(x, t0, mParam->constBz, GPUCA_MAX_SIN_PHI);
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

GPUd() void GPUTPCCompressionTrackModel::Init(float x, float y, float z, float alpha, unsigned char qPt, const GPUParam& GPUrestrict() param)
{
  // initialize track model
  mX = x;
  mAlpha = alpha;
  mCosAlpha = CAMath::Cos(alpha);
  mSinAlpha = CAMath::Sin(alpha);
  mP[0] = y;
  mP[1] = z;
  mP[2] = 0.f;
  mP[3] = 0.f;
  mP[4] = (qPt - 127.f) * (20.f / 127.f);
  resetCovariance();
  mNDF = -5;
  mBz = param.constBz;
  float pti = CAMath::Abs(mP[4]);
  if (pti < 1.e-4f) {
    pti = 1.e-4f; // set 10.000 GeV momentum for straight track
  }
  mTrk.x = x;
  mTrk.y = y;
  mTrk.z = z;
  mTrk.q = (mP[4] >= 0) ? 1.f : -1.f;
  mTrk.pt = 1.f / pti;
  mTrk.p = mTrk.pt;
  mTrk.px = mTrk.pt;
  mTrk.py = 0.f;
  mTrk.pz = 0.f;
  mTrk.qpt = mTrk.q * pti;
  calculateMaterialCorrection();
}

GPUd() int GPUTPCCompressionTrackModel::Propagate(float x, float alpha)
{
  // constrain sin(phi)
  if (mP[2] > MaxSinPhi) {
    mP[2] = MaxSinPhi;
  } else if (mP[2] < -MaxSinPhi) {
    mP[2] = -MaxSinPhi;
  }
  // propagate track parameters to specified x
  if (CAMath::Abs(alpha - mAlpha) > 1.e-4) {
    if (rotateToAlpha(alpha) != 0) {
      return -2;
    }
  }
  if (CAMath::Abs(x - mX) < 1.e-7f) {
    mX = x;
    return 0;
  }

  // propagate mTrk to t0e
  PhysicalTrackModel t0e(mTrk);
  float dLp = 0;
  if (CAMath::Abs(x - t0e.x) < 1.e-8f) {
    return 0;
  }
  if (propagateToXBzLightNoUpdate(t0e, x, mBz, dLp)) {
    return 1;
  }
  updatePhysicalTrackValues(t0e);
  if (CAMath::Abs(t0e.sinphi) >= MaxSinPhi) {
    return -3;
  }
  return followLinearization(t0e, mBz, dLp);
}

GPUd() int GPUTPCCompressionTrackModel::Filter(float y, float z, int iRow)
{
  // apply kalman filter update with measurement y/z
  float err2Y, err2Z;
  getClusterRMS2(iRow, z, mTrk.sinphi, mTrk.dzds, err2Y, err2Z);
  if (mNDF == -5) {
    // first measurement: no need to filter, as the result is known in advance. so just set it
    // ignore offline statistical errors for now (as is also done by default)
    mP[0] = y;
    mP[1] = z;
    mC[0] = err2Y;
    mC[2] = err2Z;
    mNDF = -3;
    return 0;
  }

  // constrain sin(phi)
  if (mP[2] > MaxSinPhi) {
    mP[2] = MaxSinPhi;
  } else if (mP[2] < -MaxSinPhi) {
    mP[2] = -MaxSinPhi;
  }

  const float d00 = mC[0], d01 = mC[1], d02 = mC[3], d03 = mC[6], d04 = mC[10];
  const float d10 = mC[1], d11 = mC[2], d12 = mC[4], d13 = mC[7], d14 = mC[11];

  const float z0 = y - mP[0];
  const float z1 = z - mP[1];
  float w0, w1, w2;
  if (mNDF <= 0) {
    w0 = 1.f / (err2Y + d00);
    w1 = 0;
    w2 = 1.f / (err2Z + d11);
  } else {
    w0 = d11 + err2Z, w1 = d10, w2 = d00 + err2Y;
    { // Invert symmetric matrix
      float det = w0 * w2 - w1 * w1;
      if (CAMath::Abs(det) < 1.e-10f) {
        return -1;
      }
      det = 1.f / det;
      w0 = w0 * det;
      w1 = -w1 * det;
      w2 = w2 * det;
    }
  }
  mNDF += 2;

  if (mNDF <= 0) {
    const float k00 = d00 * w0;
    const float k20 = d02 * w0;
    const float k40 = d04 * w0;
    const float k11 = d11 * w2;
    const float k31 = d13 * w2;
    mP[0] += k00 * z0;
    mP[1] += k11 * z1;
    mP[2] += k20 * z0;
    mP[3] += k31 * z1;
    mP[4] += k40 * z0;

    mC[0] -= k00 * d00;
    mC[2] -= k11 * d11;
    mC[3] -= k20 * d00;
    mC[5] -= k20 * d02;
    mC[7] -= k31 * d11;
    mC[9] -= k31 * d13;
    mC[10] -= k00 * d04;
    mC[12] -= k40 * d02;
    mC[14] -= k40 * d04;
  } else {
    const float k00 = d00 * w0 + d01 * w1;
    const float k01 = d00 * w1 + d10 * w2;
    const float k10 = d01 * w0 + d11 * w1;
    const float k11 = d01 * w1 + d11 * w2;
    const float k20 = d02 * w0 + d12 * w1;
    const float k21 = d02 * w1 + d12 * w2;
    const float k30 = d03 * w0 + d13 * w1;
    const float k31 = d03 * w1 + d13 * w2;
    const float k40 = d04 * w0 + d14 * w1;
    const float k41 = d04 * w1 + d14 * w2;

    mP[0] += k00 * z0 + k01 * z1;
    mP[1] += k10 * z0 + k11 * z1;
    mP[2] += k20 * z0 + k21 * z1;
    mP[3] += k30 * z0 + k31 * z1;
    mP[4] += k40 * z0 + k41 * z1;

    mC[0] -= k00 * d00 + k01 * d10;

    mC[2] -= k10 * d01 + k11 * d11;

    mC[3] -= k20 * d00 + k21 * d10;
    mC[5] -= k20 * d02 + k21 * d12;

    mC[7] -= k30 * d01 + k31 * d11;
    mC[9] -= k30 * d03 + k31 * d13;

    mC[10] -= k40 * d00 + k41 * d10;
    mC[12] -= k40 * d02 + k41 * d12;
    mC[14] -= k40 * d04 + k41 * d14;

    mC[1] -= k10 * d00 + k11 * d10;
    mC[4] -= k20 * d01 + k21 * d11;
    mC[6] -= k30 * d00 + k31 * d10;
    mC[8] -= k30 * d02 + k31 * d12;
    mC[11] -= k40 * d01 + k41 * d11;
    mC[13] -= k40 * d03 + k41 * d13;
  }
  return 0;
}

GPUd() int GPUTPCCompressionTrackModel::Mirror()
{
  float dy = -2.f * mTrk.q * mTrk.px / mBz;
  float dS; // path in XY
  {
    float chord = dy;        // chord to the extrapolated point == |dy|*sign(x direction)
    float sa = -mTrk.cosphi; //  sin( half of the rotation angle ) ==  (chord/2) / radius

    // dS = (Pt/b)*2*arcsin( sa )
    //    = (Pt/b)*2*sa*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )
    //    =       chord*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )

    float sa2 = sa * sa;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    // const float k6 = 5.f/112.f;
    dS = chord + chord * sa2 * (k2 + k4 * sa2);
    // dS = sqrtf(pt2)/b*2.*CAMath::ASin( sa );
  }

  if (mTrk.sinphi < 0.f) {
    dS = -dS;
  }

  mTrk.y = mTrk.y + 2.f * dy; // TODO check why dy is added TWICE to the track position
  mTrk.z = mTrk.z + 2.f * mTrk.dzds * dS;
  changeDirection();

  // Energy Loss

  float dL = CAMath::Copysign(dS * mTrk.dlds, -1.f); // we are in flight direction

  float& mC40 = mC[10];
  float& mC41 = mC[11];
  float& mC42 = mC[12];
  float& mC43 = mC[13];
  float& mC44 = mC[14];

  float dLmask = 0.f;
  bool maskMS = (CAMath::Abs(dL) < mMaterial.DLMax);
  if (maskMS) {
    dLmask = dL;
  }
  float dLabs = CAMath::Abs(dLmask);
  float corr = 1.f - mMaterial.EP2 * dLmask;

  float corrInv = 1.f / corr;
  mTrk.px *= corrInv;
  mTrk.py *= corrInv;
  mTrk.pz *= corrInv;
  mTrk.pt *= corrInv;
  mTrk.p *= corrInv;
  mTrk.qpt *= corr;

  mP[4] *= corr;

  mC40 *= corr;
  mC41 *= corr;
  mC42 *= corr;
  mC43 *= corr;
  mC44 = mC44 * corr * corr + dLabs * mMaterial.sigmadE2;

  return 0;
}

GPUd() void GPUTPCCompressionTrackModel::updatePhysicalTrackValues(PhysicalTrackModel& trk)
{
  float px = trk.px;
  if (CAMath::Abs(px) < 1.e-4f) {
    px = CAMath::Copysign(1.e-4f, px);
  }

  trk.pt = sqrt(px * px + trk.py * trk.py);
  float pti = 1.f / trk.pt;
  trk.p = sqrt(px * px + trk.py * trk.py + trk.pz * trk.pz);
  trk.sinphi = trk.py * pti;
  trk.cosphi = px * pti;
  trk.secphi = trk.pt / px;
  trk.dzds = trk.pz * pti;
  trk.dlds = trk.p * pti;
  trk.qpt = trk.q * pti;
}

GPUd() void GPUTPCCompressionTrackModel::changeDirection()
{
  mTrk.py = -mTrk.py;
  mTrk.pz = -mTrk.pz;
  mTrk.q = -mTrk.q;
  mTrk.sinphi = -mTrk.sinphi;
  mTrk.dzds = -mTrk.dzds;
  mTrk.qpt = -mTrk.qpt;
  updatePhysicalTrackValues(mTrk);

  mC[3] = -mC[3];
  mC[4] = -mC[4];
  mC[6] = -mC[6];
  mC[7] = -mC[7];
  mC[10] = -mC[10];
  mC[11] = -mC[11];
}

GPUd() int GPUTPCCompressionTrackModel::rotateToAlpha(float newAlpha)
{
  //
  // Rotate the track coordinate system in XY to the angle newAlpha
  // return value is error code (0==no error)
  //

  float newCosAlpha = CAMath::Cos(newAlpha);
  float newSinAlpha = CAMath::Sin(newAlpha);

  float cc = newCosAlpha * mCosAlpha + newSinAlpha * mSinAlpha; // cos(newAlpha - mAlpha);
  float ss = newSinAlpha * mCosAlpha - newCosAlpha * mSinAlpha; // sin(newAlpha - mAlpha);

  PhysicalTrackModel t0 = mTrk;

  float x0 = mTrk.x;
  float y0 = mTrk.y;
  float px0 = mTrk.px;
  float py0 = mTrk.py;

  if (CAMath::Abs(mP[2]) >= MaxSinPhi || CAMath::Abs(px0) < (1 - MaxSinPhi)) {
    return -1;
  }

  // rotate t0 track
  float px1 = px0 * cc + py0 * ss;
  float py1 = -px0 * ss + py0 * cc;

  {
    t0.x = x0 * cc + y0 * ss;
    t0.y = -x0 * ss + y0 * cc;
    t0.px = px1;
    t0.py = py1;
    updatePhysicalTrackValues(t0);
  }

  if (CAMath::Abs(py1) > MaxSinPhi * mTrk.pt || CAMath::Abs(px1) < (1 - MaxSinPhi)) {
    return -1;
  }

  // calculate X of rotated track:
  float trackX = x0 * cc + ss * mP[0];

  // transport t0 to trackX
  float dLp = 0;
  if (propagateToXBzLightNoUpdate(t0, trackX, mBz, dLp)) {
    return -1;
  }
  updatePhysicalTrackValues(t0);

  if (CAMath::Abs(t0.sinphi) >= MaxSinPhi) {
    return -1;
  }

  // now t0 is rotated and propagated, all checks are passed

  // Rotate track using mTrk for linearisation. After rotation X is not fixed, but has a covariance

  //                    Y  Z Sin DzDs q/p
  // Jacobian J0 = { { j0, 0, 0,  0,  0 }, // Y
  //                 {  0, 1, 0,  0,  0 }, // Z
  //                 {  0, 0, j1, 0,  0 }, // SinPhi
  //                 {  0, 0, 0,  1,  0 }, // DzDs
  //                 {  0, 0, 0,  0,  1 }, // q/p
  //                 { j2, 0, 0,  0,  0 } }// X (rotated )

  float j0 = cc;
  float j1 = px1 / px0;
  float j2 = ss;
  // float dy = mT->Y() - y0;
  // float ds = mT->SinPhi() - mTrk.SinPhi();

  mX = trackX;                   // == x0*cc + ss*mP[0]  == t0.x + j0*dy;
  mP[0] = -x0 * ss + cc * mP[0]; //== t0.y + j0*dy;
  // mP[2] = py1/pt0 + j1*ds; // == t0.sinphi + j1*ds; // use py1, since t0.sinphi can have different sign
  mP[2] = -CAMath::Sqrt(1.f - mP[2] * mP[2]) * ss + mP[2] * cc;

  // Rotate cov. matrix Cr = J0 x C x J0T. Cr has one more row+column for X:
  float* c = mC;

  float c15 = c[0] * j0 * j2;
  float c16 = c[1] * j2;
  float c17 = c[3] * j1 * j2;
  float c18 = c[6] * j2;
  float c19 = c[10] * j2;
  float c20 = c[0] * j2 * j2;

  c[0] *= j0 * j0;
  c[3] *= j0;
  c[10] *= j0;

  c[3] *= j1;
  c[5] *= j1 * j1;
  c[12] *= j1;

  if (setDirectionAlongX(t0)) { // change direction if Px < 0
    mP[2] = -mP[2];
    mP[3] = -mP[3];
    mP[4] = -mP[4];
    c[3] = -c[3]; // covariances with SinPhi
    c[4] = -c[4];
    c17 = -c17;
    c[6] = -c[6]; // covariances with DzDs
    c[7] = -c[7];
    c18 = -c18;
    c[10] = -c[10]; // covariances with QPt
    c[11] = -c[11];
    c19 = -c19;
  }

  // Now fix the X coordinate: so to say, transport track T to fixed X = mX.
  // only covariance changes. Use rotated and transported t0 for linearisation
  float j3 = -t0.py / t0.px;
  float j4 = -t0.pz / t0.px;
  float j5 = t0.qpt * mBz;

  //                    Y  Z Sin DzDs q/p  X
  // Jacobian J1 = { {  1, 0, 0,  0,  0,  j3 }, // Y
  //                 {  0, 1, 0,  0,  0,  j4 }, // Z
  //                 {  0, 0, 1,  0,  0,  j5 }, // SinPhi
  //                 {  0, 0, 0,  1,  0,   0 }, // DzDs
  //                 {  0, 0, 0,  0,  1,   0 } }; // q/p

  float h15 = c15 + c20 * j3;
  float h16 = c16 + c20 * j4;
  float h17 = c17 + c20 * j5;

  c[0] += j3 * (c15 + h15);

  c[2] += j4 * (c16 + h16);

  c[3] += c17 * j3 + h15 * j5;
  c[5] += j5 * (c17 + h17);

  c[7] += c18 * j4;
  // c[ 9] = c[ 9];

  c[10] += c19 * j3;
  c[12] += c19 * j5;
  // c[14] = c[14];

  mAlpha = newAlpha;
  mCosAlpha = newCosAlpha;
  mSinAlpha = newSinAlpha;
  mTrk = t0;

  return 0;
}

GPUd() int GPUTPCCompressionTrackModel::propagateToXBzLightNoUpdate(PhysicalTrackModel& t, float x, float Bz, float& dLp)
{
  //
  // transport the track to X=x in magnetic field B = ( 0, 0, Bz[kG*0.000299792458] )
  // dLp is a return value == path length / track momentum [cm/(GeV/c)]
  // the method returns error code (0 == no error)
  //
  // Additional values are not recalculated, UpdateValues() has to be called afterwards!!
  //
  float b = t.q * Bz;
  float pt2 = t.px * t.px + t.py * t.py;
  float dx = x - t.x;
  float pye = t.py - dx * b; // extrapolated py
  float pxe2 = pt2 - pye * pye;

  if (t.px < (1.f - MaxSinPhi) || pxe2 < (1.f - MaxSinPhi) * (1.f - MaxSinPhi)) {
    return -1; // can not transport to x=x
  }
  float pxe = CAMath::Sqrt(pxe2); // extrapolated px
  float pti = 1.f / CAMath::Sqrt(pt2);

  float ty = (t.py + pye) / (t.px + pxe);
  float dy = dx * ty;
  float dS; // path in XY
  {
    float chord = dx * CAMath::Sqrt(1.f + ty * ty); // chord to the extrapolated point == sqrt(dx^2+dy^2)*sign(dx)
    float sa = 0.5f * chord * b * pti;              //  sin( half of the rotation angle ) ==  (chord/2) / radius

    // dS = (Pt/b)*2*arcsin( sa )
    //    = (Pt/b)*2*sa*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )
    //    =       chord*(1 + 1/6 sa^2 + 3/40 sa^4 + 5/112 sa^6 +... )

    float sa2 = sa * sa;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    // const float k6 = 5.f/112.f;
    dS = chord + chord * sa2 * (k2 + k4 * sa2);
    // dS = sqrt(pt2)/b*2.*CAMath::ASin( sa );
  }

  dLp = pti * dS; // path in XYZ / p == path in XY / pt

  float dz = t.pz * dLp;

  t.x = x;
  t.y += dy;
  t.z += dz;
  t.px = pxe;
  t.py = pye;
  return 0;
}

GPUd() bool GPUTPCCompressionTrackModel::setDirectionAlongX(PhysicalTrackModel& t)
{
  //
  // set direction of movenment collinear to X axis
  // return value is true when direction has been changed
  //
  if (t.px >= 0) {
    return 0;
  }

  t.px = -t.px;
  t.py = -t.py;
  t.pz = -t.pz;
  t.q = -t.q;
  updatePhysicalTrackValues(t);
  return 1;
}

GPUd() int GPUTPCCompressionTrackModel::followLinearization(const PhysicalTrackModel& t0e, float Bz, float dLp)
{
  // t0e is alrerady extrapolated t0

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS = dLp * t0e.pt;
  float dL = CAMath::Abs(dLp * t0e.p);

  dL = -dL; // we are always in flight direction

  float ey = mTrk.sinphi;
  float ex = mTrk.cosphi;
  float exi = mTrk.secphi;
  float ey1 = t0e.sinphi;
  float ex1 = t0e.cosphi;
  float ex1i = t0e.secphi;

  float k = -mTrk.qpt * Bz;
  float dx = t0e.x - mTrk.x;
  float kdx = k * dx;
  float cc = ex + ex1;
  float cci = 1.f / cc;

  float dxcci = dx * cci;
  float hh = dxcci * ex1i * (1.f + ex * ex1 + ey * ey1);

  float j02 = exi * hh;
  float j04 = -Bz * dxcci * hh;
  float j13 = dS;
  float j24 = -dx * Bz;

  float* p = mP;

  float d0 = p[0] - mTrk.y;
  float d1 = p[1] - mTrk.z;
  float d2 = p[2] - mTrk.sinphi;
  float d3 = p[3] - mTrk.dzds;
  float d4 = p[4] - mTrk.qpt;

  float newSinPhi = ey1 + d2 + j24 * d4;
  if (mNDF >= 15 && CAMath::Abs(newSinPhi) > MaxSinPhi) {
    return -4;
  }

  mTrk = t0e;
  mX = t0e.x;
  p[0] = t0e.y + d0 + j02 * d2 + j04 * d4;
  p[1] = t0e.z + d1 + j13 * d3;
  p[2] = newSinPhi;
  p[3] = t0e.dzds + d3;
  p[4] = t0e.qpt + d4;

  float* c = mC;

  float c20 = c[3];
  float c21 = c[4];
  float c22 = c[5];

  float c30 = c[6];
  float c31 = c[7];
  float c32 = c[8];
  float c33 = c[9];

  float c40 = c[10];
  float c41 = c[11];
  float c42 = c[12];
  float c43 = c[13];
  float c44 = c[14];

  if (mNDF <= 0) {
    float c20ph04c42 = c20 + j04 * c42;
    float j02c22 = j02 * c22;
    float j04c44 = j04 * c44;

    float n6 = c30 + j02 * c32 + j04 * c43;
    float n7 = c31 + j13 * c33;
    float n10 = c40 + j02 * c42 + j04c44;
    float n11 = c41 + j13 * c43;
    float n12 = c42 + j24 * c44;

    c[0] += j02 * j02c22 + j04 * j04c44 + 2.f * (j02 * c20ph04c42 + j04 * c40);
    c[1] += j02 * c21 + j04 * c41 + j13 * n6;
    c[2] += j13 * (c31 + n7);
    c[3] = c20ph04c42 + j02c22 + j24 * n10;
    c[4] = c21 + j13 * c32 + j24 * n11;
    c[5] = c22 + j24 * (c42 + n12);
    c[6] = n6;
    c[7] = n7;
    c[8] = c32 + c43 * j24;
    c[10] = n10;
    c[11] = n11;
    c[12] = n12;
  } else {
    float c00 = c[0];
    float c10 = c[1];
    float c11 = c[2];

    float ss = ey + ey1;
    float tg = ss * cci;
    float xx = 1.f - 0.25f * kdx * kdx * (1.f + tg * tg);
    if (xx < 1.e-8f) {
      return -1;
    }
    xx = CAMath::Sqrt(xx);
    float yy = CAMath::Sqrt(ss * ss + cc * cc);

    float j12 = dx * mTrk.dzds * tg * (2.f + tg * (ey * exi + ey1 * ex1i)) / (xx * yy);
    float j14 = 0;
    if (CAMath::Abs(mTrk.qpt) > 1.e-6f) {
      j14 = (2.f * xx * ex1i * dx / yy - dS) * mTrk.dzds / mTrk.qpt;
    } else {
      j14 = -mTrk.dzds * Bz * dx * dx * exi * exi * exi * (0.5f * ey + (1.f / 3.f) * kdx * (1 + 2.f * ey * ey) * exi * exi);
    }

    p[1] += j12 * d2 + j14 * d4;

    float h00 = c00 + c20 * j02 + c40 * j04;
    // float h01 = c10 + c21*j02 + c41*j04;
    float h02 = c20 + c22 * j02 + c42 * j04;
    // float h03 = c30 + c32*j02 + c43*j04;
    float h04 = c40 + c42 * j02 + c44 * j04;

    float h10 = c10 + c20 * j12 + c30 * j13 + c40 * j14;
    float h11 = c11 + c21 * j12 + c31 * j13 + c41 * j14;
    float h12 = c21 + c22 * j12 + c32 * j13 + c42 * j14;
    float h13 = c31 + c32 * j12 + c33 * j13 + c43 * j14;
    float h14 = c41 + c42 * j12 + c43 * j13 + c44 * j14;

    float h20 = c20 + c40 * j24;
    float h21 = c21 + c41 * j24;
    float h22 = c22 + c42 * j24;
    float h23 = c32 + c43 * j24;
    float h24 = c42 + c44 * j24;

    c[0] = h00 + h02 * j02 + h04 * j04;

    c[1] = h10 + h12 * j02 + h14 * j04;
    c[2] = h11 + h12 * j12 + h13 * j13 + h14 * j14;

    c[3] = h20 + h22 * j02 + h24 * j04;
    c[4] = h21 + h22 * j12 + h23 * j13 + h24 * j14;
    c[5] = h22 + h24 * j24;

    c[6] = c30 + c32 * j02 + c43 * j04;
    c[7] = c31 + c32 * j12 + c33 * j13 + c43 * j14;
    c[8] = c32 + c43 * j24;
    // c[ 9] = c33;

    c[10] = c40 + c42 * j02 + c44 * j04;
    c[11] = c41 + c42 * j12 + c43 * j13 + c44 * j14;
    c[12] = c42 + c44 * j24;
    // c[13] = c43;
    // c[14] = c44;
  }

  float& mC22 = c[5];
  float& mC33 = c[9];
  float& mC40 = c[10];
  float& mC41 = c[11];
  float& mC42 = c[12];
  float& mC43 = c[13];
  float& mC44 = c[14];

  float dLmask = 0.f;
  bool maskMS = (CAMath::Abs(dL) < mMaterial.DLMax);
  if (maskMS) {
    dLmask = dL;
  }
  float dLabs = CAMath::Abs(dLmask);

  // Energy Loss
  {
    // std::cout<<"APPLY ENERGY LOSS!!!"<<std::endl;
    float corr = 1.f - mMaterial.EP2 * dLmask;
    float corrInv = 1.f / corr;
    mTrk.px *= corrInv;
    mTrk.py *= corrInv;
    mTrk.pz *= corrInv;
    mTrk.pt *= corrInv;
    mTrk.p *= corrInv;
    mTrk.qpt *= corr;

    p[4] *= corr;

    mC40 *= corr;
    mC41 *= corr;
    mC42 *= corr;
    mC43 *= corr;
    mC44 = mC44 * corr * corr + dLabs * mMaterial.sigmadE2;
  }
  //  Multiple Scattering

  {
    mC22 += dLabs * mMaterial.k22 * mTrk.cosphi * mTrk.cosphi;
    mC33 += dLabs * mMaterial.k33;
    mC43 += dLabs * mMaterial.k43;
    mC44 += dLabs * mMaterial.k44;
  }

  return 0;
}

GPUd() void GPUTPCCompressionTrackModel::calculateMaterialCorrection()
{
  const float mass = 0.13957f;

  float qpt = mTrk.qpt;
  if (CAMath::Abs(qpt) > 20) {
    qpt = 20;
  }

  float w2 = (1.f + mTrk.dzds * mTrk.dzds); //==(P/pt)2
  float pti2 = qpt * qpt;
  if (pti2 < 1.e-4f) {
    pti2 = 1.e-4f;
  }

  float mass2 = mass * mass;
  float beta2 = w2 / (w2 + mass2 * pti2);

  float p2 = w2 / pti2; // impuls 2
  float betheRho = approximateBetheBloch(p2 / mass2) * mMaterial.rho;
  float E = CAMath::Sqrt(p2 + mass2);
  float theta2 = (14.1f * 14.1f / 1.e6f) / (beta2 * p2) * mMaterial.radLenInv;

  mMaterial.EP2 = E / p2;

  // Approximate energy loss fluctuation (M.Ivanov)

  const float knst = 0.07f; // To be tuned.
  mMaterial.sigmadE2 = knst * mMaterial.EP2 * qpt;
  mMaterial.sigmadE2 = mMaterial.sigmadE2 * mMaterial.sigmadE2;

  mMaterial.k22 = theta2 * w2;
  mMaterial.k33 = mMaterial.k22 * w2;
  mMaterial.k43 = 0.f;
  mMaterial.k44 = theta2 * mTrk.dzds * mTrk.dzds * pti2;

  float br = (betheRho > 1.e-8f) ? betheRho : 1.e-8f;
  mMaterial.DLMax = 0.3f * E / br;
  mMaterial.EP2 *= betheRho;
  mMaterial.sigmadE2 = mMaterial.sigmadE2 * betheRho; // + mMaterial.fK44;
}

GPUd() float GPUTPCCompressionTrackModel::approximateBetheBloch(float beta2)
{
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula with
  // the density effect taken into account at beta*gamma > 3.5
  // (the approximation is reasonable only for solid materials)
  //------------------------------------------------------------------

  const float log0 = log(5940.f);
  const float log1 = log(3.5f * 5940.f);

  bool bad = (beta2 >= .999f) || (beta2 < 1.e-8f);

  if (bad) {
    return 0.f;
  }

  float a = beta2 / (1.f - beta2);
  float b = 0.5f * log(a);
  float d = 0.153e-3f / beta2;
  float c = b - beta2;

  float ret = d * (log0 + b + c);
  float case1 = d * (log1 + c);

  if (a > 3.5f * 3.5f) {
    ret = case1;
  }

  return ret;
}

GPUd() void GPUTPCCompressionTrackModel::getClusterRMS2(int iRow, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  // Only O2 geometry considered at the moment. Is AliRoot geometry support needed?
  int rowType = iRow < 97 ? (iRow < 63 ? 0 : 1) : (iRow < 127 ? 2 : 3);
  if (rowType > 2) {
    rowType = 2; // TODO: Add type 3
  }
  constexpr float tpcLength = 250.f - 0.275f;
  z = CAMath::Abs(tpcLength - CAMath::Abs(z));
  float s2 = sinPhi * sinPhi;
  if (s2 > 0.95f * 0.95f) {
    s2 = 0.95f * 0.95f;
  }
  float sec2 = 1.f / (1.f - s2);
  float angleY2 = s2 * sec2;          // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx

  const float* cY = mParamRMS0[0][rowType];
  ErrY2 = cY[0] + cY[1] * z + cY[2] * angleY2;
  ErrY2 *= ErrY2;

  const float* cZ = mParamRMS0[1][rowType];
  ErrZ2 = cZ[0] + cZ[1] * z + cZ[2] * angleZ2;
  ErrZ2 *= ErrZ2;
}

GPUd() void GPUTPCCompressionTrackModel::resetCovariance()
{
  mC[0] = 100.f;
  mC[1] = 0.f;
  mC[2] = 100.f;
  mC[3] = 0.f;
  mC[4] = 0.f;
  mC[5] = 1.f;
  mC[6] = 0.f;
  mC[7] = 0.f;
  mC[8] = 0.f;
  mC[9] = 10.f;
  mC[10] = 0.f;
  mC[11] = 0.f;
  mC[12] = 0.f;
  mC[13] = 0.f;
  mC[14] = 10.f;
}

#endif
