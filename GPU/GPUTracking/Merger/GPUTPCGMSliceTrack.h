// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMSliceTrack.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMSLICETRACK_H
#define GPUTPCGMSLICETRACK_H

#include "GPUTPCTrack.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUCommonMath.h"
#include "GPUO2DataTypes.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCGMSliceTrack
 *
 * The class describes TPC slice tracks used in GPUTPCGMMerger
 */
class GPUTPCGMMerger;
class GPUTPCGMSliceTrack
{
 public:
  GPUd() float Alpha() const { return mAlpha; }
  GPUd() unsigned char Slice() const { return mSlice; }
  GPUd() char CSide() const { return mSlice >= 18; }
  GPUd() int NClusters() const { return mNClusters; }
  GPUd() int PrevNeighbour() const { return mNeighbour[0]; }
  GPUd() int NextNeighbour() const { return mNeighbour[1]; }
  GPUd() int Neighbour(int i) const { return mNeighbour[i]; }
  GPUd() int PrevSegmentNeighbour() const { return mSegmentNeighbour[0]; }
  GPUd() int NextSegmentNeighbour() const { return mSegmentNeighbour[1]; }
  GPUd() int SegmentNeighbour(int i) const { return mSegmentNeighbour[i]; }
  GPUd() const GPUTPCTrack* OrigTrack() const { return mOrigTrack; }
  GPUd() float X() const { return mX; }
  GPUd() float Y() const { return mY; }
  GPUd() float Z() const { return mZ; }
  GPUd() float SinPhi() const { return mSinPhi; }
  GPUd() float CosPhi() const { return mCosPhi; }
  GPUd() float SecPhi() const { return mSecPhi; }
  GPUd() float DzDs() const { return mDzDs; }
  GPUd() float QPt() const { return mQPt; }
  GPUd() float TZOffset() const { return mTZOffset; }
  GPUd() unsigned char Leg() const { return mLeg; }

  GPUd() int LocalTrackId() const { return mLocalTrackId; }
  GPUd() void SetLocalTrackId(int v) { mLocalTrackId = v; }
  GPUd() int GlobalTrackId(int n) const { return mGlobalTrackIds[n]; }
  GPUd() void SetGlobalTrackId(int n, int v) { mGlobalTrackIds[n] = v; }

  GPUd() float MaxClusterZT() const { return CAMath::Max(mClusterZT[0], mClusterZT[1]); }
  GPUd() float MinClusterZT() const { return CAMath::Min(mClusterZT[0], mClusterZT[1]); }
  GPUd() float ClusterZT0() const { return mClusterZT[0]; }
  GPUd() float ClusterZTN() const { return mClusterZT[1]; }
  GPUd() void SetClusterZT(float v1, float v2)
  {
    mClusterZT[0] = v1;
    mClusterZT[1] = v2;
  }

  GPUd() void Set(const GPUTPCGMTrackParam& trk, const GPUTPCTrack* sliceTr, float alpha, int slice);
  GPUd() void Set(const GPUTPCGMMerger* merger, const GPUTPCTrack* sliceTr, float alpha, int slice);

  GPUd() void SetGlobalSectorTrackCov()
  {
    mC0 = 1;
    mC2 = 1;
    mC3 = 0;
    mC5 = 1;
    mC7 = 0;
    mC9 = 1;
    mC10 = 0;
    mC12 = 0;
    mC14 = 10;
  }

  GPUd() void SetNClusters(int v) { mNClusters = v; }
  GPUd() void SetPrevNeighbour(int v) { mNeighbour[0] = v; }
  GPUd() void SetNextNeighbour(int v) { mNeighbour[1] = v; }
  GPUd() void SetNeighbor(int v, int i) { mNeighbour[i] = v; }
  GPUd() void SetPrevSegmentNeighbour(int v) { mSegmentNeighbour[0] = v; }
  GPUd() void SetNextSegmentNeighbour(int v) { mSegmentNeighbour[1] = v; }
  GPUd() void SetLeg(unsigned char v) { mLeg = v; }

  GPUd() void CopyParamFrom(const GPUTPCGMSliceTrack& t)
  {
    mX = t.mX;
    mY = t.mY;
    mZ = t.mZ;
    mSinPhi = t.mSinPhi;
    mDzDs = t.mDzDs;
    mQPt = t.mQPt;
    mCosPhi = t.mCosPhi, mSecPhi = t.mSecPhi;
    mAlpha = t.mAlpha;
  }

  GPUd() bool FilterErrors(const GPUTPCGMMerger* merger, int iSlice, float maxSinPhi = GPUCA_MAX_SIN_PHI, float sinPhiMargin = 0.f);
  GPUd() bool TransportToX(GPUTPCGMMerger* merger, float x, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi, bool doCov = true) const;
  GPUd() bool TransportToXAlpha(GPUTPCGMMerger* merger, float x, float sinAlpha, float cosAlpha, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi) const;
  GPUd() void CopyBaseTrackCov();

 private:
  const GPUTPCTrack* mOrigTrack;                            // pointer to original slice track
  float mX, mY, mZ, mSinPhi, mDzDs, mQPt, mCosPhi, mSecPhi; // parameters
  float mTZOffset;                                          // Z offset with early transform, T offset otherwise
  float mC0, mC2, mC3, mC5, mC7, mC9, mC10, mC12, mC14;     // covariances
  float mAlpha;                                             // alpha angle
  float mClusterZT[2];                                      // Minimum maximum cluster Z / T
  int mNClusters;                                           // N clusters
  int mNeighbour[2];                                        //
  int mSegmentNeighbour[2];                                 //
  int mLocalTrackId;                                        // Corrected local track id in terms of GMSliceTracks array
  int mGlobalTrackIds[2];                                   // IDs of associated global tracks
  unsigned char mSlice;                                     // slice of this track segment
  unsigned char mLeg;                                       // Leg of this track segment
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
