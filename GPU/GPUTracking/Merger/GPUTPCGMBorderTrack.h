// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMBorderTrack.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMBORDERTRACK_H
#define GPUTPCGMBORDERTRACK_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCGMBorderTrack
 *
 * The class describes TPC slice tracks at sector borders.
 * Used in GPUTPCGMMerger
 *
 */
class GPUTPCGMBorderTrack
{
 public:
  struct Range {
    int fId;
    float fMin, fMax;
    static bool CompMin(const Range& a, const Range& b) { return a.fMin < b.fMin; }
    static bool CompMax(const Range& a, const Range& b) { return a.fMax < b.fMax; }
  };

  GPUd() int TrackID() const { return mTrackID; }
  GPUd() short NClusters() const { return mNClusters; }
  GPUd() short Row() const { return mRow; }
  GPUd() const float* Par() const { return mP; }
  GPUd() float ZOffset() const { return mZOffset; }
  GPUd() const float* Cov() const { return mC; }
  GPUd() const float* CovD() const { return mD; }

  GPUd() void SetTrackID(int v) { mTrackID = v; }
  GPUd() void SetNClusters(short v) { mNClusters = v; }
  GPUd() void SetRow(short v) { mRow = v; }
  GPUd() void SetPar(int i, float x) { mP[i] = x; }
  GPUd() void SetZOffset(float v) { mZOffset = v; }
  GPUd() void SetCov(int i, float x) { mC[i] = x; }
  GPUd() void SetCovD(int i, float x) { mD[i] = x; }

  GPUd() static bool CheckChi2(float x1, float y1, float cx1, float cxy1, float cy1, float x2, float y2, float cx2, float cxy2, float cy2, float chi2cut)
  {
    //* Calculate Chi2/ndf deviation
    float dx = x1 - x2;
    float dy = y1 - y2;
    float cx = cx1 + cx2;
    float cxy = cxy1 + cxy2;
    float cy = cy1 + cy2;
    float det = cx * cy - cxy * cxy;
    // printf("Res %f Det %f Cut %f %s   -   ", ( cy*dx - (cxy+cxy)*dy )*dx + cx*dy*dy, det, (det + det) * chi2cut, (CAMath::Abs(( cy*dx - (cxy+cxy)*dy )*dx + cx*dy*dy) < CAMath::Abs((det+det)*chi2cut)) ? "OK" : "Fail");
    return (CAMath::Abs((cy * dx - (cxy + cxy) * dy) * dx + cx * dy * dy) < CAMath::Abs((det + det) * chi2cut));
  }

  GPUd() bool CheckChi2Y(const GPUTPCGMBorderTrack& t, float chi2cut) const
  {
    float d = mP[0] - t.mP[0];
    return (d * d < chi2cut * (mC[0] + t.mC[0]));
  }

  GPUd() bool CheckChi2Z(const GPUTPCGMBorderTrack& t, float chi2cut) const
  {
    float d = mP[1] - t.mP[1] + (mZOffset - t.mZOffset);
    return (d * d < chi2cut * (mC[1] + t.mC[1]));
  }

  GPUd() bool CheckChi2QPt(const GPUTPCGMBorderTrack& t, float chi2cut) const
  {
    float d = mP[4] - t.mP[4];
    if (CAMath::Abs(d) > 0.3f && CAMath::Abs(d) > 0.5f * CAMath::Min(CAMath::Abs(mP[4]), CAMath::Abs(t.mP[4]))) {
      return false; // Crude cut to avoid some bogus matches, TODO: recheck
    }
    return (d * d < chi2cut * (mC[4] + t.mC[4]));
  }

  GPUd() bool CheckChi2YS(const GPUTPCGMBorderTrack& t, float chi2cut) const { return CheckChi2(mP[0], mP[2], mC[0], mD[0], mC[2], t.mP[0], t.mP[2], t.mC[0], t.mD[0], t.mC[2], chi2cut); }

  GPUd() bool CheckChi2ZT(const GPUTPCGMBorderTrack& t, float chi2cut) const { return CheckChi2(mP[1], mP[3], mC[1], mD[1], mC[3], t.mP[1] + (t.mZOffset - mZOffset), t.mP[3], t.mC[1], t.mD[1], t.mC[3], chi2cut); }

  GPUd() void LimitCov()
  {
    // TODO: Why are Cov entries so large?
    for (int i = 0; i < 2; i++) {
      if (mC[i] > 5.f) {
        mC[i] = 5.f;
      }
    }
    for (int i = 2; i < 4; i++) {
      if (mC[i] > 0.5f) {
        mC[i] = 0.5f;
      }
    }
    float maxCov4 = CAMath::Max(0.5f, mP[4] * mP[4] * 0.25f);
    if (mC[4] > maxCov4) {
      mC[4] = maxCov4;
    }
    for (int i = 0; i < 2; i++) {
      if (mD[i] > 0.5f) {
        mD[i] = 0.5f;
      }
    }
    if (mD[0] * mD[0] > mC[0] * mC[2]) {
      mD[0] = 0.f;
    }
    if (mD[1] * mD[1] > mC[1] * mC[3]) {
      mD[1] = 0.f;
    }
  }

 private:
  int mTrackID;     // track index
  short mNClusters; // n clusters
  short mRow;
  float mP[5];
  float mZOffset;
  float mC[5];
  float mD[2];
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
