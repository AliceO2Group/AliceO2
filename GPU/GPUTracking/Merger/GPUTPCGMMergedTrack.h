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

namespace o2::gpu
{
/**
 * @class GPUTPCGMMergedTrack
 *
 * The class is used to store merged tracks in GPUTPCGMMerger
 */
class GPUTPCGMMergedTrack
{
 public:
  GPUd() uint32_t NClusters() const { return mNClusters; }
  GPUd() uint32_t NClustersFitted() const { return mNClustersFitted; }
  GPUd() uint32_t FirstClusterRef() const { return mFirstClusterRef; }
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
  GPUd() bool OK() const { return mFlags & 0x01; }
  GPUd() bool Looper() const { return mFlags & 0x02; }
  GPUd() bool CSide() const { return mFlags & 0x04; }
  GPUd() bool CCE() const { return mFlags & 0x08; }
  GPUd() bool MergedLooperUnconnected() const { return mFlags & 0x10; }
  GPUd() bool MergedLooperConnected() const { return mFlags & 0x20; }
  GPUd() bool MergedLooper() const { return mFlags & 0x30; }
  GPUd() int32_t PrevSegment() const { return mPrevSegment; }
  GPUd() uint8_t Leg() const { return mLeg; }
  GPUd() uint8_t Flags() const { return mFlags; }

  GPUd() void SetNClusters(int32_t v) { mNClusters = v; }
  GPUd() void SetNClustersFitted(int32_t v) { mNClustersFitted = v; }
  GPUd() void SetFirstClusterRef(int32_t v) { mFirstClusterRef = v; }
  GPUd() void SetParam(const GPUTPCGMTrackParam& v) { mParam = v; }
  GPUd() void SetAlpha(float v) { mAlpha = v; }
  GPUd() void SetPrevSegment(int32_t v) { mPrevSegment = v; }
  GPUd() void SetLeg(uint8_t v) { mLeg = v; }
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
  GPUd() void SetMergedLooperUnconnected(bool v)
  {
    if (v) {
      mFlags |= 0x10;
    } else {
      mFlags &= 0xEF;
    }
  }
  GPUd() void SetMergedLooperConnected(bool v)
  {
    if (v) {
      mFlags |= 0x20;
    } else {
      mFlags &= 0xDF;
    }
  }
  GPUd() void SetFlags(uint8_t v) { mFlags = v; }

  GPUd() const gputpcgmmergertypes::GPUTPCOuterParam& OuterParam() const { return mOuterParam; }
  GPUd() gputpcgmmergertypes::GPUTPCOuterParam& OuterParam() { return mOuterParam; }

 private:
  GPUTPCGMTrackParam mParam;                         //* fitted track parameters
  gputpcgmmergertypes::GPUTPCOuterParam mOuterParam; //* outer param

  float mAlpha;              //* alpha angle
  uint32_t mFirstClusterRef; //* index of the first track cluster in corresponding cluster arrays
  int32_t mPrevSegment;      //* next segment in case of looping track
  uint16_t mNClusters;       //* number of track clusters
  uint16_t mNClustersFitted; //* number of clusters used in fit
  uint8_t mFlags;
  uint8_t mLeg;

#if !defined(GPUCA_STANDALONE)
  ClassDefNV(GPUTPCGMMergedTrack, 0);
#endif
};
} // namespace o2::gpu

#endif
