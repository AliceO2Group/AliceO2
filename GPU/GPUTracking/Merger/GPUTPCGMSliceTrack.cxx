// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMSliceTrack.cxx
/// \author Sergey Gorbunov, David Rohr

#include "GPUParam.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCGMMerger.h"
#include "GPUTPCConvertImpl.h"
#ifndef __OPENCLCPP__
#include <cmath>
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

bool GPUTPCGMSliceTrack::FilterErrors(const GPUTPCGMMerger* merger, int iSlice, float maxSinPhi, float sinPhiMargin)
{
  float lastX;
  if (merger->Param().earlyTpcTransform) {
    lastX = mOrigTrack->Cluster(mOrigTrack->NClusters() - 1).GetX(); // TODO: Why is this needed, Row2X should work, but looses some tracks
  } else {
    //float lastX = merger->Param().tpcGeometry.Row2X(mOrigTrack->Cluster(mOrigTrack->NClusters() - 1).GetRow()); // TODO: again, why does this reduce efficiency?
    float y, z;
    const GPUTPCSliceOutCluster& clo = mOrigTrack->Cluster(mOrigTrack->NClusters() - 1);
    const ClusterNative& cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clo.GetId()];
    GPUTPCConvertImpl::convert(*merger->GetConstantMem(), iSlice, clo.GetRow(), cl.getPad(), cl.getTime(), lastX, y, z);
  }

  const int N = 3;

  float bz = -merger->Param().ConstBz;

  float k = mQPt * bz;
  float dx = (1.f / N) * (lastX - mX);
  float kdx = k * dx;
  float dxBz = dx * bz;
  float kdx205 = 2.f + kdx * kdx * 0.5f;

  {
    merger->Param().GetClusterErrors2(0, mZ, mSinPhi, mDzDs, mC0, mC2);
    float C0a, C2a;
    merger->Param().GetClusterRMS2(0, mZ, mSinPhi, mDzDs, C0a, C2a);
    if (C0a > mC0) {
      mC0 = C0a;
    }
    if (C2a > mC2) {
      mC2 = C2a;
    }

    mC3 = 0;
    mC5 = 1;
    mC7 = 0;
    mC9 = 1;
    mC10 = 0;
    mC12 = 0;
    mC14 = 10;
  }

  for (int iStep = 0; iStep < N; iStep++) {
    float err2Y, err2Z;

    { // transport block
      float ex = mCosPhi;
      float ey = mSinPhi;
      float ey1 = kdx + ey;
      if (fabsf(ey1) > maxSinPhi) {
        if (ey1 > maxSinPhi && ey1 < maxSinPhi + sinPhiMargin) {
          ey1 = maxSinPhi - 0.01;
        } else if (ey1 > -maxSinPhi - sinPhiMargin) {
          ey1 = -maxSinPhi + 0.01;
        } else {
          return 0;
        }
      }

      float ss = ey + ey1;
      float ex1 = sqrt(1.f - ey1 * ey1);

      float cc = ex + ex1;
      float dxcci = dx / cc;

      float dy = dxcci * ss;
      float norm2 = 1.f + ey * ey1 + ex * ex1;
      float dl = dxcci * sqrt(norm2 + norm2);

      float dS;
      {
        float dSin = 0.5f * k * dl;
        float a = dSin * dSin;
        const float k2 = 1.f / 6.f;
        const float k4 = 3.f / 40.f;
        dS = dl + dl * a * (k2 + a * (k4)); //+ k6*a) );
      }

      float dz = dS * mDzDs;
      float ex1i = 1.f / ex1;
      {
        merger->Param().GetClusterErrors2(0, mZ, mSinPhi, mDzDs, err2Y, err2Z);
        float C0a, C2a;
        merger->Param().GetClusterRMS2(0, mZ, mSinPhi, mDzDs, C0a, C2a);
        if (C0a > err2Y) {
          err2Y = C0a;
        }
        if (C2a > err2Z) {
          err2Z = C2a;
        }
      }

      float hh = kdx205 * dxcci * ex1i;
      float h2 = hh * mSecPhi;

      mX += dx;
      mY += dy;
      mZ += dz;
      mSinPhi = ey1;
      mCosPhi = ex1;
      mSecPhi = ex1i;

      float h4 = bz * dxcci * hh;

      float c20 = mC3;
      float c22 = mC5;
      float c31 = mC7;
      float c33 = mC9;
      float c40 = mC10;
      float c42 = mC12;
      float c44 = mC14;

      float c20ph4c42 = c20 + h4 * c42;
      float h2c22 = h2 * c22;
      float h4c44 = h4 * c44;
      float n7 = c31 + dS * c33;
      float n10 = c40 + h2 * c42 + h4c44;
      float n12 = c42 + dxBz * c44;

      mC0 += h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40);

      mC3 = c20ph4c42 + h2c22 + dxBz * n10;
      mC10 = n10;

      mC5 = c22 + dxBz * (c42 + n12);
      mC12 = n12;

      mC2 += dS * (c31 + n7);
      mC7 = n7;
    } // end transport block

    // Filter block

    float c00 = mC0, c11 = mC2, c20 = mC3, c31 = mC7, c40 = mC10;

    float mS0 = 1.f / (err2Y + c00);
    float mS2 = 1.f / (err2Z + c11);

    // K = CHtS

    float k00, k11, k20, k31, k40;

    k00 = c00 * mS0;
    k20 = c20 * mS0;
    k40 = c40 * mS0;

    mC0 -= k00 * c00;
    mC5 -= k20 * c20;
    mC10 -= k00 * c40;
    mC12 -= k40 * c20;
    mC3 -= k20 * c00;
    mC14 -= k40 * c40;

    k11 = c11 * mS2;
    k31 = c31 * mS2;

    mC7 -= k31 * c11;
    mC2 -= k11 * c11;
    mC9 -= k31 * c31;
  }

  //* Check that the track parameters and covariance matrix are reasonable

  bool ok = CAMath::Finite(mX) && CAMath::Finite(mY) && CAMath::Finite(mZ) && CAMath::Finite(mSinPhi) && CAMath::Finite(mDzDs) && CAMath::Finite(mQPt) && CAMath::Finite(mCosPhi) && CAMath::Finite(mSecPhi) && CAMath::Finite(mZOffset) && CAMath::Finite(mC0) && CAMath::Finite(mC2) &&
            CAMath::Finite(mC3) && CAMath::Finite(mC5) && CAMath::Finite(mC7) && CAMath::Finite(mC9) && CAMath::Finite(mC10) && CAMath::Finite(mC12) && CAMath::Finite(mC14);

  if (mC0 <= 0.f || mC2 <= 0.f || mC5 <= 0.f || mC9 <= 0.f || mC14 <= 0.f || mC0 > 5.f || mC2 > 5.f || mC5 > 2.f || mC9 > 2.f) {
    ok = 0;
  }

  if (ok) {
    ok = ok && (mC3 * mC3 <= mC5 * mC0) && (mC7 * mC7 <= mC9 * mC2) && (mC10 * mC10 <= mC14 * mC0) && (mC12 * mC12 <= mC14 * mC5);
  }

  return ok;
}

bool GPUTPCGMSliceTrack::TransportToX(float x, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi, bool doCov) const
{
  Bz = -Bz;
  float ex = mCosPhi;
  float ey = mSinPhi;
  float k = mQPt * Bz;
  float dx = x - mX;
  float ey1 = k * dx + ey;

  if (fabsf(ey1) > maxSinPhi) {
    return 0;
  }

  float ex1 = sqrt(1.f - ey1 * ey1);
  float dxBz = dx * Bz;

  float ss = ey + ey1;
  float cc = ex + ex1;
  float dxcci = dx / cc;
  float norm2 = 1.f + ey * ey1 + ex * ex1;

  float dy = dxcci * ss;

  float dS;
  {
    float dl = dxcci * sqrt(norm2 + norm2);
    float dSin = 0.5f * k * dl;
    float a = dSin * dSin;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    // const float k6 = 5.f/112.f;
    dS = dl + dl * a * (k2 + a * (k4)); //+ k6*a) );
  }

  float dz = dS * mDzDs;

  b.SetPar(0, mY + dy);
  b.SetPar(1, mZ + dz);
  b.SetPar(2, ey1);
  b.SetPar(3, mDzDs);
  b.SetPar(4, mQPt);
  b.SetZOffset(mZOffset);

  if (!doCov) {
    return (1);
  }

  float ex1i = 1.f / ex1;
  float hh = dxcci * ex1i * norm2;
  float h2 = hh * mSecPhi;
  float h4 = Bz * dxcci * hh;

  float c20 = mC3;
  float c22 = mC5;
  float c31 = mC7;
  float c33 = mC9;
  float c40 = mC10;
  float c42 = mC12;
  float c44 = mC14;

  float c20ph4c42 = c20 + h4 * c42;
  float h2c22 = h2 * c22;
  float h4c44 = h4 * c44;
  float n7 = c31 + dS * c33;

  if (fabsf(mQPt) > 6.66) // Special treatment for low Pt
  {
    b.SetCov(0, CAMath::Max(mC0, mC0 + h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40))); // Do not decrease Y cov for matching!
    float C2tmp = dS * 2.f * c31;
    if (C2tmp < 0) {
      C2tmp = 0;
    }
    b.SetCov(1, mC2 + C2tmp + dS * dS * c33); // Incorrect formula, correct would be "dS * (c31 + n7)", but we need to make sure cov(Z) increases regardless of the direction of the propagation
  } else {
    b.SetCov(0, mC0 + h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40));
    b.SetCov(1, mC2 + dS * (c31 + n7));
  }
  b.SetCov(2, c22 + dxBz * (c42 + c42 + dxBz * c44));
  b.SetCov(3, c33);
  b.SetCov(4, c44);
  b.SetCovD(0, c20ph4c42 + h2c22 + dxBz * (c40 + h2 * c42 + h4c44));
  b.SetCovD(1, n7);

  b.LimitCov();

  return 1;
}

bool GPUTPCGMSliceTrack::TransportToXAlpha(float newX, float sinAlpha, float cosAlpha, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi) const
{
  //*

  float c00 = mC0;
  float c11 = mC2;
  float c20 = mC3;
  float c22 = mC5;
  float c31 = mC7;
  float c33 = mC9;
  float c40 = mC10;
  float c42 = mC12;
  float c44 = mC14;

  float x, y;
  float z = mZ;
  float sinPhi = mSinPhi;
  float cosPhi = mCosPhi;
  float secPhi = mSecPhi;
  float dzds = mDzDs;
  float qpt = mQPt;

  // Rotate the coordinate system in XY on the angle alpha
  {
    float sP = sinPhi, cP = cosPhi;
    cosPhi = cP * cosAlpha + sP * sinAlpha;
    sinPhi = -cP * sinAlpha + sP * cosAlpha;

    if (CAMath::Abs(sinPhi) > GPUCA_MAX_SIN_PHI || CAMath::Abs(cP) < 1.e-2) {
      return 0;
    }

    secPhi = 1. / cosPhi;
    float j0 = cP * secPhi;
    float j2 = cosPhi / cP;
    x = mX * cosAlpha + mY * sinAlpha;
    y = -mX * sinAlpha + mY * cosAlpha;

    c00 *= j0 * j0;
    c40 *= j0;

    c22 *= j2 * j2;
    c42 *= j2;
    if (cosPhi < 0.f) { // rotate to 180'
      cosPhi = -cosPhi;
      secPhi = -secPhi;
      sinPhi = -sinPhi;
      dzds = -dzds;
      qpt = -qpt;
      c20 = -c20;
      c31 = -c31;
      c40 = -c40;
    }
  }

  Bz = -Bz;
  float ex = cosPhi;
  float ey = sinPhi;
  float k = qpt * Bz;
  float dx = newX - x;
  float ey1 = k * dx + ey;

  if (fabsf(ey1) > maxSinPhi) {
    return 0;
  }

  float ex1 = sqrt(1.f - ey1 * ey1);

  float dxBz = dx * Bz;

  float ss = ey + ey1;
  float cc = ex + ex1;
  float dxcci = dx / cc;
  float norm2 = 1.f + ey * ey1 + ex * ex1;

  float dy = dxcci * ss;

  float dS;
  {
    float dl = dxcci * sqrt(norm2 + norm2);
    float dSin = 0.5f * k * dl;
    float a = dSin * dSin;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    // const float k6 = 5.f/112.f;
    dS = dl + dl * a * (k2 + a * (k4)); //+ k6*a) );
  }

  float ex1i = 1.f / ex1;
  float dz = dS * dzds;

  float hh = dxcci * ex1i * norm2;
  float h2 = hh * secPhi;
  float h4 = Bz * dxcci * hh;

  float c20ph4c42 = c20 + h4 * c42;
  float h2c22 = h2 * c22;
  float h4c44 = h4 * c44;
  float n7 = c31 + dS * c33;

  b.SetPar(0, y + dy);
  b.SetPar(1, z + dz);
  b.SetPar(2, ey1);
  b.SetPar(3, dzds);
  b.SetPar(4, qpt);
  b.SetZOffset(mZOffset);

  b.SetCov(0, c00 + h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40));
  b.SetCov(1, c11 + dS * (c31 + n7));
  b.SetCov(2, c22 + dxBz * (c42 + c42 + dxBz * c44));
  b.SetCov(3, c33);
  b.SetCov(4, c44);
  b.SetCovD(0, c20ph4c42 + h2c22 + dxBz * (c40 + h2 * c42 + h4c44));
  b.SetCovD(1, n7);

  b.LimitCov();

  return 1;
}
