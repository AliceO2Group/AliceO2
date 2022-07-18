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

/// \file GPUTPCGMSliceTrack.cxx
/// \author Sergey Gorbunov, David Rohr

#include "GPUParam.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCGMMerger.h"
#include "GPUTPCConvertImpl.h"
#include "GPUParam.inc"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

GPUd() void GPUTPCGMSliceTrack::Set(const GPUTPCGMMerger* merger, const GPUTPCTrack* sliceTr, float alpha, int slice)
{
  const GPUTPCBaseTrackParam& t = sliceTr->Param();
  mOrigTrack = sliceTr;
  mParam.mX = t.GetX();
  mParam.mY = t.GetY();
  mParam.mZ = t.GetZ();
  mParam.mDzDs = t.GetDzDs();
  mParam.mSinPhi = t.GetSinPhi();
  mParam.mQPt = t.GetQPt();
  mParam.mCosPhi = sqrt(1.f - mParam.mSinPhi * mParam.mSinPhi);
  mParam.mSecPhi = 1.f / mParam.mCosPhi;
  mAlpha = alpha;
  mSlice = slice;
  if (merger->Param().par.earlyTpcTransform) {
    mTZOffset = t.GetZOffset();
  } else {
    mTZOffset = merger->GetConstantMem()->calibObjects.fastTransform->convZOffsetToVertexTime(slice, t.GetZOffset(), merger->Param().par.continuousMaxTimeBin);
  }
  mNClusters = sliceTr->NHits();
}

GPUd() void GPUTPCGMSliceTrack::Set(const GPUTPCGMTrackParam& trk, const GPUTPCTrack* sliceTr, float alpha, int slice)
{
  mOrigTrack = sliceTr;
  mParam.mX = trk.GetX();
  mParam.mY = trk.GetY();
  mParam.mZ = trk.GetZ();
  mParam.mDzDs = trk.GetDzDs();
  mParam.mSinPhi = trk.GetSinPhi();
  mParam.mQPt = trk.GetQPt();
  mParam.mCosPhi = sqrt(1.f - mParam.mSinPhi * mParam.mSinPhi);
  mParam.mSecPhi = 1.f / mParam.mCosPhi;
  mAlpha = alpha;
  mSlice = slice;
  mTZOffset = trk.GetTZOffset();
  mNClusters = sliceTr->NHits();
  mParam.mC0 = trk.GetCov(0);
  mParam.mC2 = trk.GetCov(2);
  mParam.mC3 = trk.GetCov(3);
  mParam.mC5 = trk.GetCov(5);
  mParam.mC7 = trk.GetCov(7);
  mParam.mC9 = trk.GetCov(9);
  mParam.mC10 = trk.GetCov(10);
  mParam.mC12 = trk.GetCov(12);
  mParam.mC14 = trk.GetCov(14);
}

GPUd() void GPUTPCGMSliceTrack::SetParam2(const GPUTPCGMTrackParam& trk)
{
  mParam2.mX = trk.GetX();
  mParam2.mY = trk.GetY();
  mParam2.mZ = trk.GetZ();
  mParam2.mDzDs = trk.GetDzDs();
  mParam2.mSinPhi = trk.GetSinPhi();
  mParam2.mQPt = trk.GetQPt();
  mParam2.mCosPhi = sqrt(1.f - mParam2.mSinPhi * mParam2.mSinPhi);
  mParam2.mSecPhi = 1.f / mParam2.mCosPhi;
  mParam2.mC0 = trk.GetCov(0);
  mParam2.mC2 = trk.GetCov(2);
  mParam2.mC3 = trk.GetCov(3);
  mParam2.mC5 = trk.GetCov(5);
  mParam2.mC7 = trk.GetCov(7);
  mParam2.mC9 = trk.GetCov(9);
  mParam2.mC10 = trk.GetCov(10);
  mParam2.mC12 = trk.GetCov(12);
  mParam2.mC14 = trk.GetCov(14);
}

GPUd() bool GPUTPCGMSliceTrack::FilterErrors(const GPUTPCGMMerger* merger, int iSlice, float maxSinPhi, float sinPhiMargin)
{
  float lastX;
  if (merger->Param().par.earlyTpcTransform && !merger->Param().rec.tpc.mergerReadFromTrackerDirectly) {
    lastX = mOrigTrack->OutTrackCluster(mOrigTrack->NHits() - 1).GetX(); // TODO: Why is this needed, Row2X should work, but looses some tracks
  } else {
    //float lastX = merger->Param().tpcGeometry.Row2X(mOrigTrack->Cluster(mOrigTrack->NClusters() - 1).GetRow()); // TODO: again, why does this reduce efficiency?
    float y, z;
    const GPUTPCSliceOutCluster* clo;
    int row, index;
    if (merger->Param().rec.tpc.mergerReadFromTrackerDirectly) {
      const GPUTPCTracker& trk = merger->GetConstantMem()->tpcTrackers[iSlice];
      const GPUTPCHitId& ic = trk.TrackHits()[mOrigTrack->FirstHitID() + mOrigTrack->NHits() - 1];
      index = trk.Data().ClusterDataIndex(trk.Data().Row(ic.RowIndex()), ic.HitIndex()) + merger->GetConstantMem()->ioPtrs.clustersNative->clusterOffset[iSlice][0];
      row = ic.RowIndex();
    } else {
      clo = &mOrigTrack->OutTrackCluster(mOrigTrack->NHits() - 1);
      index = clo->GetId();
      row = clo->GetRow();
    }
    const ClusterNative& cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[index];
    GPUTPCConvertImpl::convert(*merger->GetConstantMem(), iSlice, row, cl.getPad(), cl.getTime(), lastX, y, z);
  }

  const int N = 3;

  float bz = -merger->Param().constBz;

  float k = mParam.mQPt * bz;
  float dx = (1.f / N) * (lastX - mParam.mX);
  float kdx = k * dx;
  float dxBz = dx * bz;
  float kdx205 = 2.f + kdx * kdx * 0.5f;

  {
    merger->Param().GetClusterErrors2(0, mParam.mZ, mParam.mSinPhi, mParam.mDzDs, mParam.mC0, mParam.mC2);
    float C0a, C2a;
    merger->Param().GetClusterRMS2(0, mParam.mZ, mParam.mSinPhi, mParam.mDzDs, C0a, C2a);
    if (C0a > mParam.mC0) {
      mParam.mC0 = C0a;
    }
    if (C2a > mParam.mC2) {
      mParam.mC2 = C2a;
    }

    mParam.mC3 = 0;
    mParam.mC5 = 1;
    mParam.mC7 = 0;
    mParam.mC9 = 1;
    mParam.mC10 = 0;
    mParam.mC12 = 0;
    mParam.mC14 = 10;
  }

  for (int iStep = 0; iStep < N; iStep++) {
    float err2Y, err2Z;

    { // transport block
      float ex = mParam.mCosPhi;
      float ey = mParam.mSinPhi;
      float ey1 = kdx + ey;
      if (CAMath::Abs(ey1) > maxSinPhi) {
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

      float dz = dS * mParam.mDzDs;
      float ex1i = 1.f / ex1;
      {
        merger->Param().GetClusterErrors2(0, mParam.mZ, mParam.mSinPhi, mParam.mDzDs, err2Y, err2Z);
        float C0a, C2a;
        merger->Param().GetClusterRMS2(0, mParam.mZ, mParam.mSinPhi, mParam.mDzDs, C0a, C2a);
        if (C0a > err2Y) {
          err2Y = C0a;
        }
        if (C2a > err2Z) {
          err2Z = C2a;
        }
      }

      float hh = kdx205 * dxcci * ex1i;
      float h2 = hh * mParam.mSecPhi;

      mParam.mX += dx;
      mParam.mY += dy;
      mParam.mZ += dz;
      mParam.mSinPhi = ey1;
      mParam.mCosPhi = ex1;
      mParam.mSecPhi = ex1i;

      float h4 = bz * dxcci * hh;

      float c20 = mParam.mC3;
      float c22 = mParam.mC5;
      float c31 = mParam.mC7;
      float c33 = mParam.mC9;
      float c40 = mParam.mC10;
      float c42 = mParam.mC12;
      float c44 = mParam.mC14;

      float c20ph4c42 = c20 + h4 * c42;
      float h2c22 = h2 * c22;
      float h4c44 = h4 * c44;
      float n7 = c31 + dS * c33;
      float n10 = c40 + h2 * c42 + h4c44;
      float n12 = c42 + dxBz * c44;

      mParam.mC0 += h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40);

      mParam.mC3 = c20ph4c42 + h2c22 + dxBz * n10;
      mParam.mC10 = n10;

      mParam.mC5 = c22 + dxBz * (c42 + n12);
      mParam.mC12 = n12;

      mParam.mC2 += dS * (c31 + n7);
      mParam.mC7 = n7;
    } // end transport block

    // Filter block

    float c00 = mParam.mC0, c11 = mParam.mC2, c20 = mParam.mC3, c31 = mParam.mC7, c40 = mParam.mC10;

    float mS0 = 1.f / (err2Y + c00);
    float mS2 = 1.f / (err2Z + c11);

    // K = CHtS

    float k00, k11, k20, k31, k40;

    k00 = c00 * mS0;
    k20 = c20 * mS0;
    k40 = c40 * mS0;

    mParam.mC0 -= k00 * c00;
    mParam.mC5 -= k20 * c20;
    mParam.mC10 -= k00 * c40;
    mParam.mC12 -= k40 * c20;
    mParam.mC3 -= k20 * c00;
    mParam.mC14 -= k40 * c40;

    k11 = c11 * mS2;
    k31 = c31 * mS2;

    mParam.mC7 -= k31 * c11;
    mParam.mC2 -= k11 * c11;
    mParam.mC9 -= k31 * c31;
  }

  //* Check that the track parameters and covariance matrix are reasonable

  bool ok = CAMath::Finite(mParam.mX) && CAMath::Finite(mParam.mY) && CAMath::Finite(mParam.mZ) && CAMath::Finite(mParam.mSinPhi) && CAMath::Finite(mParam.mDzDs) && CAMath::Finite(mParam.mQPt) && CAMath::Finite(mParam.mCosPhi) && CAMath::Finite(mParam.mSecPhi) && CAMath::Finite(mTZOffset) && CAMath::Finite(mParam.mC0) && CAMath::Finite(mParam.mC2) &&
            CAMath::Finite(mParam.mC3) && CAMath::Finite(mParam.mC5) && CAMath::Finite(mParam.mC7) && CAMath::Finite(mParam.mC9) && CAMath::Finite(mParam.mC10) && CAMath::Finite(mParam.mC12) && CAMath::Finite(mParam.mC14);

  if (mParam.mC0 <= 0.f || mParam.mC2 <= 0.f || mParam.mC5 <= 0.f || mParam.mC9 <= 0.f || mParam.mC14 <= 0.f || mParam.mC0 > 5.f || mParam.mC2 > 5.f || mParam.mC5 > 2.f || mParam.mC9 > 2.f) {
    ok = 0;
  }

  if (ok) {
    ok = ok && (mParam.mC3 * mParam.mC3 <= mParam.mC5 * mParam.mC0) && (mParam.mC7 * mParam.mC7 <= mParam.mC9 * mParam.mC2) && (mParam.mC10 * mParam.mC10 <= mParam.mC14 * mParam.mC0) && (mParam.mC12 * mParam.mC12 <= mParam.mC14 * mParam.mC5);
  }

  return ok;
}

GPUd() bool GPUTPCGMSliceTrack::TransportToX(GPUTPCGMMerger* merger, float x, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi, bool doCov) const
{
  Bz = -Bz;
  float ex = mParam.mCosPhi;
  float ey = mParam.mSinPhi;
  float k = mParam.mQPt * Bz;
  float dx = x - mParam.mX;
  float ey1 = k * dx + ey;

  if (CAMath::Abs(ey1) > maxSinPhi) {
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

  float dz = dS * mParam.mDzDs;

  b.SetPar(0, mParam.mY + dy);
  b.SetPar(1, mParam.mZ + dz);
  b.SetPar(2, ey1);
  b.SetPar(3, mParam.mDzDs);
  b.SetPar(4, mParam.mQPt);
  if (merger->Param().par.earlyTpcTransform) {
    b.SetZOffsetLinear(mTZOffset);
  } else {
    b.SetZOffsetLinear(merger->GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(mSlice, mTZOffset, merger->Param().par.continuousMaxTimeBin));
  }

  if (!doCov) {
    return (1);
  }

  float ex1i = 1.f / ex1;
  float hh = dxcci * ex1i * norm2;
  float h2 = hh * mParam.mSecPhi;
  float h4 = Bz * dxcci * hh;

  float c20 = mParam.mC3;
  float c22 = mParam.mC5;
  float c31 = mParam.mC7;
  float c33 = mParam.mC9;
  float c40 = mParam.mC10;
  float c42 = mParam.mC12;
  float c44 = mParam.mC14;

  float c20ph4c42 = c20 + h4 * c42;
  float h2c22 = h2 * c22;
  float h4c44 = h4 * c44;
  float n7 = c31 + dS * c33;

  if (CAMath::Abs(mParam.mQPt) > 6.66) // Special treatment for low Pt
  {
    b.SetCov(0, CAMath::Max(mParam.mC0, mParam.mC0 + h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40))); // Do not decrease Y cov for matching!
    float C2tmp = dS * 2.f * c31;
    if (C2tmp < 0) {
      C2tmp = 0;
    }
    b.SetCov(1, mParam.mC2 + C2tmp + dS * dS * c33); // Incorrect formula, correct would be "dS * (c31 + n7)", but we need to make sure cov(Z) increases regardless of the direction of the propagation
  } else {
    b.SetCov(0, mParam.mC0 + h2 * h2c22 + h4 * h4c44 + 2.f * (h2 * c20ph4c42 + h4 * c40));
    b.SetCov(1, mParam.mC2 + dS * (c31 + n7));
  }
  b.SetCov(2, c22 + dxBz * (c42 + c42 + dxBz * c44));
  b.SetCov(3, c33);
  b.SetCov(4, c44);
  b.SetCovD(0, c20ph4c42 + h2c22 + dxBz * (c40 + h2 * c42 + h4c44));
  b.SetCovD(1, n7);

  b.LimitCov();

  return 1;
}

GPUd() bool GPUTPCGMSliceTrack::TransportToXAlpha(GPUTPCGMMerger* merger, float newX, float sinAlpha, float cosAlpha, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi) const
{
  //*

  float c00 = mParam.mC0;
  float c11 = mParam.mC2;
  float c20 = mParam.mC3;
  float c22 = mParam.mC5;
  float c31 = mParam.mC7;
  float c33 = mParam.mC9;
  float c40 = mParam.mC10;
  float c42 = mParam.mC12;
  float c44 = mParam.mC14;

  float x, y;
  float z = mParam.mZ;
  float sinPhi = mParam.mSinPhi;
  float cosPhi = mParam.mCosPhi;
  float secPhi = mParam.mSecPhi;
  float dzds = mParam.mDzDs;
  float qpt = mParam.mQPt;

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
    x = mParam.mX * cosAlpha + mParam.mY * sinAlpha;
    y = -mParam.mX * sinAlpha + mParam.mY * cosAlpha;

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

  if (CAMath::Abs(ey1) > maxSinPhi) {
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
  if (merger->Param().par.earlyTpcTransform) {
    b.SetZOffsetLinear(mTZOffset);
  } else {
    b.SetZOffsetLinear(merger->GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(mSlice, mTZOffset, merger->Param().par.continuousMaxTimeBin));
  }

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

GPUd() void GPUTPCGMSliceTrack::CopyBaseTrackCov()
{
  const float* GPUrestrict() cov = mOrigTrack->Param().mC;
  mParam.mC0 = cov[0];
  mParam.mC2 = cov[2];
  mParam.mC3 = cov[3];
  mParam.mC5 = cov[5];
  mParam.mC7 = cov[7];
  mParam.mC9 = cov[9];
  mParam.mC10 = cov[10];
  mParam.mC12 = cov[12];
  mParam.mC14 = cov[14];
}
