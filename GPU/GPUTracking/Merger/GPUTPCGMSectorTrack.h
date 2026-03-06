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

/// \file GPUTPCGMSectorTrack.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMSECTORTRACK_H
#define GPUTPCGMSECTORTRACK_H

#include "GPUTPCTrack.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUCommonMath.h"
#include "GPUO2DataTypes.h"

namespace o2::gpu
{
/**
 * @class GPUTPCGMSectorTrack
 *
 * The class describes TPC sector tracks used in GPUTPCGMMerger
 */
class GPUTPCGMMerger;
class GPUTPCGMSectorTrack
{
 public:
  GPUd() float Alpha() const { return mAlpha; }
  GPUd() uint8_t Sector() const { return mSector; }
  GPUd() bool CSide() const { return mSector >= 18; }
  GPUd() int32_t NClusters() const { return mNClusters; }
  GPUd() int32_t PrevNeighbour() const { return mNeighbour[0]; }
  GPUd() int32_t NextNeighbour() const { return mNeighbour[1]; }
  GPUd() int32_t Neighbour(int32_t i) const { return mNeighbour[i]; }
  GPUd() int32_t PrevSegmentNeighbour() const { return mSegmentNeighbour[0]; }
  GPUd() int32_t NextSegmentNeighbour() const { return mSegmentNeighbour[1]; }
  GPUd() int32_t SegmentNeighbour(int32_t i) const { return mSegmentNeighbour[i]; }
  GPUd() int32_t AnyNeighbour(int32_t i) const
  {
    return (i < 2) ? mSegmentNeighbour[i] : mNeighbour[i - 2];
  }
  GPUd() const GPUTPCTrack* OrigTrack() const { return mOrigTrack; }
  GPUd() float X() const { return mParam.mX; }
  GPUd() float Y() const { return mParam.mY; }
  GPUd() float Z() const { return mParam.mZ; }
  GPUd() float SinPhi() const { return mParam.mSinPhi; }
  GPUd() float CosPhi() const { return mParam.mCosPhi; }
  GPUd() float SecPhi() const { return mParam.mSecPhi; }
  GPUd() float DzDs() const { return mParam.mDzDs; }
  GPUd() float QPt() const { return mParam.mQPt; }
  GPUd() const auto& Param() const { return mParam; }
  GPUd() float TOffset() const { return mTOffset; }

  GPUd() int32_t LocalTrackId() const { return mLocalTrackId; }
  GPUd() void SetLocalTrackId(int32_t v) { mLocalTrackId = v; }
  GPUd() int32_t ExtrapolatedTrackId(int32_t n) const { return mExtrapolatedTrackIds[n]; }
  GPUd() void SetExtrapolatedTrackId(int32_t n, int32_t v) { mExtrapolatedTrackIds[n] = v; }
  GPUd() int32_t* ExtrapolatedTrackIds() { return mExtrapolatedTrackIds; }

  GPUd() float MaxClusterT() const { return CAMath::Max(mClusterT[0], mClusterT[1]); }
  GPUd() float MinClusterT() const { return CAMath::Min(mClusterT[0], mClusterT[1]); }
  GPUd() float ClusterT0() const { return mClusterT[0]; }
  GPUd() float ClusterTN() const { return mClusterT[1]; }
  GPUd() void SetClusterT(float v1, float v2)
  {
    mClusterT[0] = v1;
    mClusterT[1] = v2;
  }

  GPUd() void Set(const GPUTPCGMTrackParam& trk, const GPUTPCTrack* sectorTr, float alpha, int32_t sector);
  GPUd() void SetParam2(const GPUTPCGMTrackParam& trk);
  GPUd() void Set(const GPUTPCGMMerger* merger, const GPUTPCTrack* sectorTr, float alpha, int32_t sector);
  GPUd() void UseParam2() { mParam = mParam2; } // TODO: Clean this up!
  GPUd() void SetX2(float v) { mParam2.mX = v; }
  GPUd() float X2() const { return mParam2.mX; }

  GPUd() void SetGlobalSectorTrackCov()
  {
    mParam.mC0 = 1;
    mParam.mC2 = 1;
    mParam.mC3 = 0;
    mParam.mC5 = 1;
    mParam.mC7 = 0;
    mParam.mC9 = 1;
    mParam.mC10 = 0;
    mParam.mC12 = 0;
    mParam.mC14 = 10;
  }

  GPUd() void SetNClusters(int32_t v) { mNClusters = v; }
  GPUd() void SetPrevNeighbour(int32_t v) { mNeighbour[0] = v; }
  GPUd() void SetNextNeighbour(int32_t v) { mNeighbour[1] = v; }
  GPUd() void SetNeighbor(int32_t v, int32_t i) { mNeighbour[i] = v; }
  GPUd() void SetPrevSegmentNeighbour(int32_t v) { mSegmentNeighbour[0] = v; }
  GPUd() void SetNextSegmentNeighbour(int32_t v) { mSegmentNeighbour[1] = v; }

  GPUd() void CopyParamFrom(const GPUTPCGMSectorTrack& t)
  {
    mParam.mX = t.mParam.mX;
    mParam.mY = t.mParam.mY;
    mParam.mZ = t.mParam.mZ;
    mParam.mSinPhi = t.mParam.mSinPhi;
    mParam.mDzDs = t.mParam.mDzDs;
    mParam.mQPt = t.mParam.mQPt;
    mParam.mCosPhi = t.mParam.mCosPhi;
    mParam.mSecPhi = t.mParam.mSecPhi;
    mAlpha = t.mAlpha;
  }

  GPUd() bool FilterErrors(const GPUTPCGMMerger* merger, int32_t iSector, float maxSinPhi = GPUCA_MAX_SIN_PHI, float sinPhiMargin = 0.f);
  template <int I = 0>
  GPUd() bool TransportToX(GPUTPCGMMerger* merger, float x, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi, bool doCov = true) const;
  GPUd() bool TransportToXAlpha(GPUTPCGMMerger* merger, float x, float sinAlpha, float cosAlpha, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi) const;
  GPUd() void CopyBaseTrackCov();
  struct sectorTrackParam {
    float mX, mY, mZ, mSinPhi, mDzDs, mQPt, mCosPhi, mSecPhi; // parameters
    float mC0, mC2, mC3, mC5, mC7, mC9, mC10, mC12, mC14;     // covariances
  };

 private:
  const GPUTPCTrack* mOrigTrack;    // pointer to original sector track
  sectorTrackParam mParam;          // Track parameters
  sectorTrackParam mParam2;         // Parameters at other side
  float mTOffset;                   // Z offset with early transform, T offset otherwise
  float mAlpha;                     // alpha angle
  float mClusterT[2];               // Minimum maximum cluster T
  int32_t mNClusters;               // N clusters
  int32_t mNeighbour[2];            //
  int32_t mSegmentNeighbour[2];     //
  int32_t mLocalTrackId;            // Corrected local track id in terms of GMSectorTracks array for extrapolated tracks, UNDEFINED for local tracks!
  int32_t mExtrapolatedTrackIds[2]; // IDs of associated extrapolated tracks
  uint8_t mSector;                  // sector of this track segment

  ClassDefNV(GPUTPCGMSectorTrack, 1);
};
} // namespace o2::gpu

#endif
