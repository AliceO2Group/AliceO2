// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMPropagator.cxx
/// \author Sergey Gorbunov, David Rohr

#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMPhysicalTrackModel.h"
#include "GPUParam.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUO2DataTypes.h"

#ifndef __OPENCL__
#include <cmath>
#endif

#if defined(GPUCA_GM_USE_FULL_FIELD)
#include "AliTracker.h"
#include "AliMagF.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

GPUd() void GPUTPCGMPropagator::GetBxByBz(float Alpha, float X, float Y, float Z, float B[3]) const
{
  // get global coordinates

  float cs = CAMath::Cos(Alpha);
  float sn = CAMath::Sin(Alpha);

#if defined(GPUCA_GM_USE_FULL_FIELD)
  const double kCLight = 0.000299792458;
  double r[3] = { X * cs - Y * sn, X * sn + Y * cs, Z };
  double bb[3];
  AliTracker::GetBxByBz(r, bb);
  bb[0] *= kCLight;
  bb[1] *= kCLight;
  bb[2] *= kCLight;
/*
  cout<<"AliTracker::GetBz()= "<<AliTracker::GetBz()<<endl;
  cout<<"AliTracker::UniformField() "<<AliTracker::UniformField()<<endl;
  AliMagF* fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  cout<<"Fast field = "<<(void*) fld->GetFastField()<<endl;
  AliMagF::BMap_t  type = fld->GetMapType() ;
  cout<<"Field type: "<<type<<endl;
  //  fMapType==k2BMap_t
*/
#else
  float bb[3];
  switch (mFieldRegion) {
    case ITS:
      mField->GetFieldIts(X * cs - Y * sn, X * sn + Y * cs, Z, bb);
      break;
    case TRD:
      mField->GetFieldTrd(X * cs - Y * sn, X * sn + Y * cs, Z, bb);
      break;
    case TPC:
    default:
      mField->GetField(X * cs - Y * sn, X * sn + Y * cs, Z, bb);
  }

#endif

  // rotate field to local coordinates

  B[0] = bb[0] * cs + bb[1] * sn;
  B[1] = -bb[0] * sn + bb[1] * cs;
  B[2] = bb[2];
  /*if( mToyMCEvents ){ // special treatment for toy monte carlo
    B[0] = 0;
    B[1] = 0;
    B[2] = mField->GetNominalBz();
  }*/
}

GPUd() float GPUTPCGMPropagator::GetBz(float Alpha, float X, float Y, float Z) const
{
  if (mToyMCEvents) { // special treatment for toy monte carlo
    float B[3];
    GetBxByBz(Alpha, X, Y, Z, B);
    return B[2];
  }

  // get global coordinates

  float cs = CAMath::Cos(Alpha);
  float sn = CAMath::Sin(Alpha);

#if defined(GPUCA_GM_USE_FULL_FIELD)
  const double kCLight = 0.000299792458;
  double r[3] = { X * cs - Y * sn, X * sn + Y * cs, Z };
  double bb[3];
  AliTracker::GetBxByBz(r, bb);
  return bb[2] * kCLight;
#else
  switch (mFieldRegion) {
    case ITS:
      return mField->GetFieldItsBz(X * cs - Y * sn, X * sn + Y * cs, Z);
    case TRD:
      return mField->GetFieldTrdBz(X * cs - Y * sn, X * sn + Y * cs, Z);
    case TPC:
    default:
      return mField->GetFieldBz(X * cs - Y * sn, X * sn + Y * cs, Z);
  }

#endif
}

GPUd() int GPUTPCGMPropagator::RotateToAlpha(float newAlpha)
{
  //
  // Rotate the track coordinate system in XY to the angle newAlpha
  // return value is error code (0==no error)
  //

  float cc = CAMath::Cos(newAlpha - mAlpha);
  float ss = CAMath::Sin(newAlpha - mAlpha);

  GPUTPCGMPhysicalTrackModel t0 = mT0;

  float x0 = mT0.X();
  float y0 = mT0.Y();
  float px0 = mT0.Px();
  float py0 = mT0.Py();
  // float pt0 = mT0.GetPt();

  if (CAMath::Abs(mT->GetSinPhi()) >= mMaxSinPhi || CAMath::Abs(px0) < (1 - mMaxSinPhi)) {
    return -1;
  }

  // rotate t0 track
  float px1 = px0 * cc + py0 * ss;
  float py1 = -px0 * ss + py0 * cc;

  {
    t0.X() = x0 * cc + y0 * ss;
    t0.Y() = -x0 * ss + y0 * cc;
    t0.Px() = px1;
    t0.Py() = py1;
    t0.UpdateValues();
  }

  if (CAMath::Abs(py1) > mMaxSinPhi * mT0.GetPt() || CAMath::Abs(px1) < (1 - mMaxSinPhi)) {
    return -1;
  }

  // calculate X of rotated track:
  float trackX = x0 * cc + ss * mT->Y();

  // transport t0 to trackX
  float B[3];
  GetBxByBz(newAlpha, t0.X(), t0.Y(), t0.Z(), B);
  float dLp = 0;
  if (t0.PropagateToXBxByBz(trackX, B[0], B[1], B[2], dLp)) {
    return -1;
  }

  if (CAMath::Abs(t0.SinPhi()) >= mMaxSinPhi) {
    return -1;
  }

  // now t0 is rotated and propagated, all checks are passed

  // Rotate track using mT0 for linearisation. After rotation X is not fixed, but has a covariance

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
  // float ds = mT->SinPhi() - mT0.SinPhi();

  mT->X() = trackX;                  // == x0*cc + ss*mT->Y()  == t0.X() + j0*dy;
  mT->Y() = -x0 * ss + cc * mT->Y(); //== t0.Y() + j0*dy;
  // mT->SinPhi() = py1/pt0 + j1*ds; // == t0.SinPhi() + j1*ds; // use py1, since t0.SinPhi can have different sign
  mT->SinPhi() = -CAMath::Sqrt(1.f - mT->SinPhi() * mT->SinPhi()) * ss + mT->SinPhi() * cc;

  // Rotate cov. matrix Cr = J0 x C x J0T. Cr has one more row+column for X:
  float* c = mT->Cov();

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

  if (!mFitInProjections && mT->NDF() > 0) {
    c[1] *= j0;
    c[6] *= j0;
    c[4] *= j1;
    c[8] *= j1;
  }

  if (t0.SetDirectionAlongX()) { // change direction if Px < 0
    mT->SinPhi() = -mT->SinPhi();
    mT->DzDs() = -mT->DzDs();
    mT->QPt() = -mT->QPt();
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

  // Now fix the X coordinate: so to say, transport track T to fixed X = mT->X().
  // only covariance changes. Use rotated and transported t0 for linearisation
  float j3 = -t0.Py() / t0.Px();
  float j4 = -t0.Pz() / t0.Px();
  float j5 = t0.QPt() * B[2];

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

  if (!mFitInProjections && mT->NDF() > 0) {
    c[1] += c16 * j3 + h15 * j4;
    c[4] += c17 * j4 + h16 * j5;
    c[6] += c18 * j3;
    c[8] += c18 * j5;
    c[11] += c19 * j4;
    // c[13] = c[13];
  }

  mAlpha = newAlpha;
  mT0 = t0;

  return 0;
}

GPUd() int GPUTPCGMPropagator::PropagateToXAlpha(float posX, float posAlpha, bool inFlyDirection)
{
  if (CAMath::Abs(posAlpha - mAlpha) > 1.e-4f) {
    if (RotateToAlpha(posAlpha) != 0) {
      return -2;
    }
  }

  float B[3];
  GetBxByBz(mAlpha, mT0.X(), mT0.Y(), mT0.Z(), B);

  // propagate mT0 to t0e

  GPUTPCGMPhysicalTrackModel t0e(mT0);
  float dLp = 0;
  if (t0e.PropagateToXBxByBz(posX, B[0], B[1], B[2], dLp) && t0e.PropagateToXBzLight(posX, B[2], dLp)) {
    return 1;
  }

  if (CAMath::Abs(t0e.SinPhi()) >= mMaxSinPhi) {
    return -3;
  }

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS = dLp * t0e.Pt();
  float dL = CAMath::Abs(dLp * t0e.P());

  if (inFlyDirection) {
    dL = -dL;
  }

  float ey = mT0.SinPhi();
  float ex = mT0.CosPhi();
  float exi = mT0.SecPhi();
  float ey1 = t0e.SinPhi();
  float ex1 = t0e.CosPhi();
  float ex1i = t0e.SecPhi();

  float bz = B[2];
  float k = -mT0.QPt() * bz;
  float dx = posX - mT0.X();
  float kdx = k * dx;
  float cc = ex + ex1;
  float cci = 1.f / cc;

  float dxcci = dx * cci;
  float hh = dxcci * ex1i * (1.f + ex * ex1 + ey * ey1);
  // float hh = dxcci*ex1i*(2.f+0.5f*kdx*kdx);  //DR: Before was like this!

  float j02 = exi * hh;
  float j04 = -bz * dxcci * hh;
  float j13 = dS;
  float j24 = -dx * bz;

  float* p = mT->Par();

  float d0 = p[0] - mT0.Y();
  float d1 = p[1] - mT0.Z();
  float d2 = p[2] - mT0.SinPhi();
  float d3 = p[3] - mT0.DzDs();
  float d4 = p[4] - mT0.QPt();

  float newSinPhi = t0e.SinPhi() + d2 + j24 * d4;
  if (mT->NDF() >= 15 && CAMath::Abs(newSinPhi) > GPUCA_MAX_SIN_PHI) {
    return (-4);
  }

  mT0 = t0e;
  mT->X() = t0e.X();
  p[0] = t0e.Y() + d0 + j02 * d2 + j04 * d4;
  p[1] = t0e.Z() + d1 + j13 * d3;
  p[2] = newSinPhi;
  p[3] = t0e.DzDs() + d3;
  p[4] = t0e.QPt() + d4;

  float* c = mT->Cov();

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

  if (mFitInProjections || mT->NDF() <= 0) {
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

    float j12 = dx * mT0.DzDs() * tg * (2.f + tg * (ey * exi + ey1 * ex1i)) / (xx * yy);
    float j14 = 0;
    if (CAMath::Abs(mT0.QPt()) > 1.e-6f) {
      j14 = (2.f * xx * ex1i * dx / yy - dS) * mT0.DzDs() / mT0.QPt();
    } else {
      j14 = -mT0.DzDs() * bz * dx * dx * exi * exi * exi * (0.5f * ey + (1.f / 3.f) * kdx * (1 + 2.f * ey * ey) * exi * exi);
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

  if (1 || !mToyMCEvents) {
    // std::cout<<"APPLY ENERGY LOSS!!!"<<std::endl;
    float corr = 1.f - mMaterial.EP2 * dLmask;
    float corrInv = 1.f / corr;
    mT0.Px() *= corrInv;
    mT0.Py() *= corrInv;
    mT0.Pz() *= corrInv;
    mT0.Pt() *= corrInv;
    mT0.P() *= corrInv;
    mT0.QPt() *= corr;

    p[4] *= corr;

    mC40 *= corr;
    mC41 *= corr;
    mC42 *= corr;
    mC43 *= corr;
    mC44 = mC44 * corr * corr + dLabs * mMaterial.sigmadE2;
  } else {
    // std::cout<<"DONT APPLY ENERGY LOSS!!!"<<std::endl;
  }

  //  Multiple Scattering

  if (!mToyMCEvents) {
    mC22 += dLabs * mMaterial.k22 * mT0.CosPhi() * mT0.CosPhi();
    mC33 += dLabs * mMaterial.k33;
    mC43 += dLabs * mMaterial.k43;
    mC44 += dLabs * mMaterial.k44;
  }
  return 0;
}

GPUd() int GPUTPCGMPropagator::GetPropagatedYZ(float x, float& projY, float& projZ)
{
  float bz = GetBz(mAlpha, mT->X(), mT->Y(), mT->Z());
  float k = mT0.QPt() * bz;
  float dx = x - mT->X();
  float kdx = k * dx;
  float ex = mT0.CosPhi();
  float ey = mT0.SinPhi();
  float ey1 = kdx + ey;
  if (CAMath::Abs(ey1) > GPUCA_MAX_SIN_PHI) {
    return 1;
  }
  float ss = ey + ey1;
  float ex1 = CAMath::Sqrt(1.f - ey1 * ey1);
  float cc = ex + ex1;
  float dxcci = dx / cc;
  float dy = dxcci * ss;
  float norm2 = 1.f + ey * ey1 + ex * ex1;
  float dl = dxcci * CAMath::Sqrt(norm2 + norm2);
  float dS;
  {
    float dSin = 0.5f * k * dl;
    float a = dSin * dSin;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    dS = dl + dl * a * (k2 + a * (k4));
  }
  float dz = dS * mT0.DzDs();
  projY = mT->Y() + dy;
  projZ = mT->Z() + dz;
  return 0;
}

/*
GPUd() int GPUTPCGMPropagator::PropagateToXAlphaBz(float posX, float posAlpha, bool inFlyDirection)
{
  if ( CAMath::Abs( posAlpha - mAlpha) > 1.e-4 ) {
    if( RotateToAlpha( posAlpha )!=0 ) return -2;
  }

  float Bz = GetBz( mAlpha, mT0.X(), mT0.Y(), mT0.Z() );

  // propagate mT0 to t0e

  GPUTPCGMPhysicalTrackModel t0e(mT0);
  float dLp = 0;
  if (t0e.PropagateToXBzLight( posX, Bz, dLp )) return 1;
  t0e.UpdateValues();
  if( CAMath::Abs( t0e.SinPhi() ) >= mMaxSinPhi ) return -3;

  // propagate track and cov matrix with derivatives for (0,0,Bz) field

  float dS =  dLp*t0e.Pt();
  float dL =  CAMath::Abs(dLp*t0e.P());

  if( inFlyDirection ) dL = -dL;

  float k  = -mT0.QPt()*Bz;
  float dx = posX - mT0.X();
  float kdx = k*dx;
  float dxcci = dx / (mT0.CosPhi() + t0e.CosPhi());

  float hh = dxcci*t0e.SecPhi()*(2.f+0.5f*kdx*kdx);
  float h02 = mT0.SecPhi()*hh;
  float h04 = -Bz*dxcci*hh;
  float h13 = dS;
  float h24 = -dx*Bz;

  float *p = mT->Par();

  float d0 = p[0] - mT0.Y();
  float d1 = p[1] - mT0.Z();
  float d2 = p[2] - mT0.SinPhi();
  float d3 = p[3] - mT0.DzDs();
  float d4 = p[4] - mT0.QPt();

  float newSinPhi = t0e.SinPhi() +  d2           + h24*d4;
  if (mT->NDF() >= 15 && CAMath::Abs(newSinPhi) > GPUCA_MAX_SIN_PHI) return(-4);

  mT0 = t0e;

  mT->X() = t0e.X();
  p[0] = t0e.Y() + d0    + h02*d2         + h04*d4;
  p[1] = t0e.Z() + d1    + h13*d3;
  p[2] = newSinPhi;
  p[3] = t0e.DzDs() + d3;
  p[4] = t0e.QPt() + d4;

  float *c = mT->Cov();
  float c20 = c[ 3];
  float c21 = c[ 4];
  float c22 = c[ 5];
  float c30 = c[ 6];
  float c31 = c[ 7];
  float c32 = c[ 8];
  float c33 = c[ 9];
  float c40 = c[10];
  float c41 = c[11];
  float c42 = c[12];
  float c43 = c[13];
  float c44 = c[14];

  float c20ph04c42 =  c20 + h04*c42;
  float h02c22 = h02*c22;
  float h04c44 = h04*c44;

  float n6 = c30 + h02*c32 + h04*c43;
  float n7 = c31 + h13*c33;
  float n10 = c40 + h02*c42 + h04c44;
  float n11 = c41 + h13*c43;
  float n12 = c42 + h24*c44;

  c[8] = c32 + h24*c43;

  c[0]+= h02*h02c22 + h04*h04c44 + 2.f*( h02*c20ph04c42  + h04*c40 );

  c[1]+= h02*c21 + h04*c41 + h13*n6;
  c[6] = n6;

  c[2]+= h13*(c31 + n7);
  c[7] = n7;

  c[3] = c20ph04c42 + h02c22  + h24*n10;
  c[10] = n10;

  c[4] = c21 + h13*c32 + h24*n11;
  c[11] = n11;

  c[5] = c22 + h24*( c42 + n12 );
  c[12] = n12;

  // Energy Loss

  float &mC22 = c[5];
  float &mC33 = c[9];
  float &mC40 = c[10];
  float &mC41 = c[11];
  float &mC42 = c[12];
  float &mC43 = c[13];
  float &mC44 = c[14];

  float dLmask = 0.f;
  bool maskMS = ( CAMath::Abs( dL ) < mMaterial.fDLMax );
  if( maskMS ) dLmask = dL;
  float dLabs = CAMath::Abs( dLmask);
  float corr = 1.f - mMaterial.fEP2* dLmask ;

  float corrInv = 1.f/corr;
  mT0.Px()*=corrInv;
  mT0.Py()*=corrInv;
  mT0.Pz()*=corrInv;
  mT0.Pt()*=corrInv;
  mT0.P()*=corrInv;
  mT0.QPt()*=corr;

  p[4]*= corr;

  mC40 *= corr;
  mC41 *= corr;
  mC42 *= corr;
  mC43 *= corr;
  mC44  = mC44*corr*corr + dLabs*mMaterial.fSigmadE2;

  //  Multiple Scattering

  mC22 += dLabs * mMaterial.fK22 * mT0.CosPhi()*mT0.CosPhi();
  mC33 += dLabs * mMaterial.fK33;
  mC43 += dLabs * mMaterial.fK43;
  mC44 += dLabs * mMaterial.fK44;

  return 0;
}
*/

GPUd() void GPUTPCGMPropagator::GetErr2(float& err2Y, float& err2Z, const GPUParam& param, float posZ, int iRow, short clusterState) const
{
  if (mSpecialErrors) {
    param.GetClusterErrors2(iRow, posZ, mT0.GetSinPhi(), mT0.DzDs(), err2Y, err2Z);
  } else {
    param.GetClusterRMS2(iRow, posZ, mT0.GetSinPhi(), mT0.DzDs(), err2Y, err2Z);
  }

  if (clusterState & GPUTPCGMMergedTrackHit::flagEdge) {
    err2Y += 0.35f;
    err2Z += 0.15f;
  }
  if (clusterState & GPUTPCGMMergedTrackHit::flagSingle) {
    err2Y += 0.2f;
    err2Z += 0.2f;
  }
  if (clusterState & (GPUTPCGMMergedTrackHit::flagSplitPad | GPUTPCGMMergedTrackHit::flagShared | GPUTPCGMMergedTrackHit::flagSingle)) {
    err2Y += 0.03f;
    err2Y *= 3;
  }
  if (clusterState & (GPUTPCGMMergedTrackHit::flagSplitTime | GPUTPCGMMergedTrackHit::flagShared | GPUTPCGMMergedTrackHit::flagSingle)) {
    err2Z += 0.03f;
    err2Z *= 3;
  }
  mStatErrors.GetOfflineStatisticalErrors(err2Y, err2Z, mT0.SinPhi(), mT0.DzDs(), clusterState);
}

GPUd() float GPUTPCGMPropagator::PredictChi2(float posY, float posZ, int iRow, const GPUParam& param, short clusterState) const
{
  float err2Y, err2Z;
  GetErr2(err2Y, err2Z, param, posZ, iRow, clusterState);
  return PredictChi2(posY, posZ, err2Y, err2Z);
}

GPUd() float GPUTPCGMPropagator::PredictChi2(float posY, float posZ, float err2Y, float err2Z) const
{
  const float* mC = mT->Cov();
  const float* mP = mT->Par();
  const float z0 = posY - mP[0];
  const float z1 = posZ - mP[1];

  if (!mFitInProjections || mT->NDF() <= 0) {
    const float w0 = 1.f / (err2Y + mC[0]);
    const float w2 = 1.f / (err2Z + mC[2]);
    return w0 * z0 * z0 + w2 * z1 * z1;
  } else {
    float w0 = mC[2] + err2Z, w1 = mC[1], w2 = mC[0] + err2Y;
    { // Invert symmetric matrix
      float det = w0 * w2 - w1 * w1;
      if (CAMath::Abs(det) < 1.e-10f) {
        det = 1.e-10f;
      }
      det = 1.f / det;
      w0 = w0 * det;
      w1 = -w1 * det;
      w2 = w2 * det;
    }
    return CAMath::Abs((w0 * z0 + w1 * z1) * z0) + CAMath::Abs((w1 * z0 + w2 * z1) * z1);
  }
}

GPUd() int GPUTPCGMPropagator::Update(float posY, float posZ, int iRow, const GPUParam& param, short clusterState, bool rejectChi2, bool refit)
{
  float err2Y, err2Z;
  GetErr2(err2Y, err2Z, param, posZ, iRow, clusterState);

  if (mT->NDF() == -5) { // first measurement: no need to filter, as the result is known in advance. just set it.
    mT->ResetCovariance();
    float* mC = mT->Cov();
    float* mP = mT->Par();
    if (refit) {
      mC[14] = CAMath::Max(0.5f, CAMath::Abs(mP[4]));
      mC[5] = CAMath::Max(0.2f, CAMath::Abs(mP[2]) / 2);
      mC[9] = CAMath::Max(0.5f, CAMath::Abs(mP[3]) / 2);
    }
    mP[0] = posY;
    mP[1] = posZ;
    mC[0] = err2Y;
    mC[2] = err2Z;
    mT->NDF() = -3;
    return 0;
  }

  return Update(posY, posZ, clusterState, rejectChi2, err2Y, err2Z);
}

GPUd() int GPUTPCGMPropagator::Update(float posY, float posZ, short clusterState, bool rejectChi2, float err2Y, float err2Z)
{
  float* mC = mT->Cov();
  float* mP = mT->Par();

  float d00 = mC[0], d01 = mC[1], d02 = mC[3], d03 = mC[6], d04 = mC[10];
  float d10 = mC[1], d11 = mC[2], d12 = mC[4], d13 = mC[7], d14 = mC[11];

  float z0 = posY - mP[0];
  float z1 = posZ - mP[1];

  float w0, w1, w2, chiY, chiZ;
  if (mFitInProjections || mT->NDF() <= 0) {
    w0 = 1.f / (err2Y + d00);
    w1 = 0;
    w2 = 1.f / (err2Z + d11);
    chiY = w0 * z0 * z0;
    chiZ = w2 * z1 * z1;
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
    chiY = CAMath::Abs((w0 * z0 + w1 * z1) * z0);
    chiZ = CAMath::Abs((w1 * z0 + w2 * z1) * z1);
  }
  float dChi2 = chiY + chiZ;
  // GPUInfo("hits %d chi2 %f, new %f %f (dy %f dz %f)", N, mChi2, chiY, chiZ, z0, z1);
  if (mSpecialErrors && rejectChi2 && RejectCluster(chiY, chiZ, clusterState)) {
    return 2; // DR: TOTO get rid of stupid specialerror
  }
  mT->Chi2() += dChi2;
  mT->NDF() += 2;

  if (mFitInProjections || mT->NDF() <= 0) {
    float k00 = d00 * w0;
    float k20 = d02 * w0;
    float k40 = d04 * w0;
    float k11 = d11 * w2;
    float k31 = d13 * w2;
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
    float k00 = d00 * w0 + d01 * w1;
    float k01 = d00 * w1 + d10 * w2;
    float k10 = d01 * w0 + d11 * w1;
    float k11 = d01 * w1 + d11 * w2;
    float k20 = d02 * w0 + d12 * w1;
    float k21 = d02 * w1 + d12 * w2;
    float k30 = d03 * w0 + d13 * w1;
    float k31 = d03 * w1 + d13 * w2;
    float k40 = d04 * w0 + d14 * w1;
    float k41 = d04 * w1 + d14 * w2;

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

    if (!mFitInProjections && mT->NDF() >= 0) {
      mC[1] -= k10 * d00 + k11 * d10;

      mC[4] -= k20 * d01 + k21 * d11;

      mC[6] -= k30 * d00 + k31 * d10;
      mC[8] -= k30 * d02 + k31 * d12;

      mC[11] -= k40 * d01 + k41 * d11;
      mC[13] -= k40 * d03 + k41 * d13;
    }
  }
  return 0;
}

//*
//*  Multiple scattering and energy losses
//*

GPUd() float GPUTPCGMPropagator::ApproximateBetheBloch(float beta2)
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
    beta2 = 0.5f;
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
  if (bad) {
    ret = 0.f;
  }

  return ret;
}

GPUd() void GPUTPCGMPropagator::CalculateMaterialCorrection()
{
  //*!

  const float mass = 0.13957f;

  float qpt = mT0.GetQPt();
  if (CAMath::Abs(qpt) > 20) {
    qpt = 20;
  }

  float w2 = (1.f + mT0.GetDzDs() * mT0.GetDzDs()); //==(P/pt)2
  float pti2 = qpt * qpt;
  if (pti2 < 1.e-4f) {
    pti2 = 1.e-4f;
  }

  float mass2 = mass * mass;
  float beta2 = w2 / (w2 + mass2 * pti2);

  float p2 = w2 / pti2; // impuls 2
  float betheRho = ApproximateBetheBloch(p2 / mass2) * mMaterial.rho;
  float E = CAMath::Sqrt(p2 + mass2);
  float theta2 = (14.1f * 14.1f / 1.e6f) / (beta2 * p2) * mMaterial.rhoOverRadLen;

  mMaterial.EP2 = E / p2;

  // Approximate energy loss fluctuation (M.Ivanov)

  const float knst = 0.07f; // To be tuned.
  mMaterial.sigmadE2 = knst * mMaterial.EP2 * qpt;
  mMaterial.sigmadE2 = mMaterial.sigmadE2 * mMaterial.sigmadE2;

  mMaterial.k22 = theta2 * w2;
  mMaterial.k33 = mMaterial.k22 * w2;
  mMaterial.k43 = 0.f;
  mMaterial.k44 = theta2 * mT0.GetDzDs() * mT0.GetDzDs() * pti2;

  float br = (betheRho > 1.e-8f) ? betheRho : 1.e-8f;
  mMaterial.DLMax = 0.3f * E / br;
  mMaterial.EP2 *= betheRho;
  mMaterial.sigmadE2 = mMaterial.sigmadE2 * betheRho; // + mMaterial.fK44;
}

GPUd() void GPUTPCGMPropagator::Rotate180()
{
  mT0.X() = -mT0.X();
  mT0.Y() = -mT0.Y();
  mT0.Q() = -mT0.Q();
  mT0.Pz() = -mT0.Pz();
  mT0.UpdateValues();

  mT->X() = -mT->X();
  mT->Y() = -mT->Y();
  mT->QPt() = -mT->QPt();
  mT->DzDs() = -mT->DzDs();

  mAlpha = mAlpha + M_PI;
  while (mAlpha >= M_PI) {
    mAlpha -= 2 * M_PI;
  }
  while (mAlpha < -M_PI) {
    mAlpha += 2 * M_PI;
  }

  float* c = mT->Cov();
  c[6] = -c[6];
  c[7] = -c[7];
  c[8] = -c[8];
  c[10] = -c[10];
  c[11] = -c[11];
  c[12] = -c[12];
}

GPUd() void GPUTPCGMPropagator::ChangeDirection()
{
  mT0.Py() = -mT0.Py();
  mT0.Pz() = -mT0.Pz();
  mT0.Q() = -mT0.Q();
  mT->SinPhi() = -mT->SinPhi();
  mT->DzDs() = -mT->DzDs();
  mT->QPt() = -mT->QPt();
  mT0.UpdateValues();

  float* c = mT->Cov();
  c[3] = -c[3];
  c[4] = -c[4];
  c[6] = -c[6];
  c[7] = -c[7];
  c[10] = -c[10];
  c[11] = -c[11];
}

GPUd() void GPUTPCGMPropagator::Mirror(bool inFlyDirection)
{
  // mirror the track and the track approximation to the point which has the same X, but located on the other side of trajectory
  float B[3];
  GetBxByBz(mAlpha, mT0.X(), mT0.Y(), mT0.Z(), B);
  float Bz = B[2];
  if (CAMath::Abs(Bz) < 1.e-8f) {
    Bz = 1.e-8f;
  }

  float dy = -2.f * mT0.Q() * mT0.Px() / Bz;
  float dS; // path in XY
  {
    float chord = dy;         // chord to the extrapolated point == |dy|*sign(x direction)
    float sa = -mT0.CosPhi(); //  sin( half of the rotation angle ) ==  (chord/2) / radius

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

  if (mT0.SinPhi() < 0.f) {
    dS = -dS;
  }

  mT0.Y() = mT0.Y() + dy;
  mT0.Z() = mT0.Z() + mT0.DzDs() * dS;
  mT0.Px() = mT0.Px(); // should be positive
  mT->Y() = mT->Y() + dy;
  mT->Z() = mT->Z() + mT0.DzDs() * dS;
  ChangeDirection();

  // Energy Loss
  if (1 || !mToyMCEvents) {
    // std::cout<<"MIRROR: APPLY ENERGY LOSS!!!"<<std::endl;

    float dL = CAMath::Abs(dS * mT0.GetDlDs());

    if (inFlyDirection) {
      dL = -dL;
    }

    float* c = mT->Cov();
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
    float corr = 1.f - mMaterial.EP2 * dLmask;

    float corrInv = 1.f / corr;
    mT0.Px() *= corrInv;
    mT0.Py() *= corrInv;
    mT0.Pz() *= corrInv;
    mT0.Pt() *= corrInv;
    mT0.P() *= corrInv;
    mT0.QPt() *= corr;

    mT->QPt() *= corr;

    mC40 *= corr;
    mC41 *= corr;
    mC42 *= corr;
    mC43 *= corr;
    mC44 = mC44 * corr * corr + dLabs * mMaterial.sigmadE2;
  } else {
    // std::cout<<"MIRROR: DONT APPLY ENERGY LOSS!!!"<<std::endl;
  }
}

GPUd() o2::base::MatBudget GPUTPCGMPropagator::getMatBudget(float* p1, float* p2)
{
#ifdef HAVE_O2HEADERS
  return mMatLUT->getMatBudget(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
#else
  return o2::base::MatBudget();
#endif
}
