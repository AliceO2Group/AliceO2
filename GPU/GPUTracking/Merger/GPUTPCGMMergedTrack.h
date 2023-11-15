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

/// \file GPUTPCGMMergedTrack.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMMERGEDTRACK_H
#define GPUTPCGMMERGEDTRACK_H

#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMMergedTrackHit.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCGMMergedTrack
 *
 * The class is used to store merged tracks in GPUTPCGMMerger
 */
class GPUTPCGMMergedTrack
{
 public:
  GPUd() unsigned int NClusters() const { return mNClusters; }
  GPUd() unsigned int NClustersFitted() const { return mNClustersFitted; }
  GPUd() unsigned int FirstClusterRef() const { return mFirstClusterRef; }
  GPUd() const GPUTPCGMTrackParam& GetParam() const { return mParam; }
  GPUd() float GetAlpha() const { return mAlpha; }
  GPUd() GPUTPCGMTrackParam& Param()
  {
    return mParam;
  }
  GPUd() float& Alpha()
  {
    return mAlpha;
  }
  GPUd() float LastX() const { return mLastX; }
  GPUd() float LastY() const { return mLastY; }
  GPUd() float LastZ() const { return mLastZ; }
  GPUd() bool OK() const { return mFlags & 0x01; }
  GPUd() bool Looper() const { return mFlags & 0x02; }
  GPUd() bool CSide() const { return mFlags & 0x04; }
  GPUd() bool CCE() const { return mFlags & 0x08; }
  GPUd() bool MergedLooper() const { return mFlags & 0x10; }

  GPUd() void SetNClusters(int v) { mNClusters = v; }
  GPUd() void SetNClustersFitted(int v) { mNClustersFitted = v; }
  GPUd() void SetFirstClusterRef(int v) { mFirstClusterRef = v; }
  GPUd() void SetParam(const GPUTPCGMTrackParam& v) { mParam = v; }
  GPUd() void SetAlpha(float v) { mAlpha = v; }
  GPUd() void SetLastX(float v) { mLastX = v; }
  GPUd() void SetLastY(float v) { mLastY = v; }
  GPUd() void SetLastZ(float v) { mLastZ = v; }
  GPUd() void SetOK(bool v)
  {
    if (v) {
      mFlags |= 0x01;
    } else {
      mFlags &= 0xFE;
    }
  }
  GPUd() void SetLooper(bool v)
  {
    if (v) {
      mFlags |= 0x02;
    } else {
      mFlags &= 0xFD;
    }
  }
  GPUd() void SetCSide(bool v)
  {
    if (v) {
      mFlags |= 0x04;
    } else {
      mFlags &= 0xFB;
    }
  }
  GPUd() void SetCCE(bool v)
  {
    if (v) {
      mFlags |= 0x08;
    } else {
      mFlags &= 0xF7;
    }
  }
  GPUd() void SetMergedLooper(bool v)
  {
    if (v) {
      mFlags |= 0x10;
    } else {
      mFlags &= 0xEF;
    }
  }
  GPUd() void SetFlags(unsigned char v) { mFlags = v; }
  GPUd() void SetLegs(unsigned char v) { mLegs = v; }
  GPUd() unsigned char Legs() const { return mLegs; }

  GPUd() const gputpcgmmergertypes::GPUTPCOuterParam& OuterParam() const { return mOuterParam; }
  GPUd() gputpcgmmergertypes::GPUTPCOuterParam& OuterParam() { return mOuterParam; }

 private:
  GPUTPCGMTrackParam mParam;                        //* fitted track parameters
  gputpcgmmergertypes::GPUTPCOuterParam mOuterParam; //* outer param

  float mAlpha;                  //* alpha angle
  float mLastX;                  //* outer X
  float mLastY;                  //* outer Y
  float mLastZ;                  //* outer Z
  unsigned int mFirstClusterRef; //* index of the first track cluster in corresponding cluster arrays
  unsigned int mNClusters;       //* number of track clusters
  unsigned int mNClustersFitted; //* number of clusters used in fit
  unsigned char mFlags;
  unsigned char mLegs;

#if !defined(GPUCA_STANDALONE)
  ClassDefNV(GPUTPCGMMergedTrack, 0);
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
