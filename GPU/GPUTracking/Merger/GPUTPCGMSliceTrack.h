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

#include "GPUTPCSliceOutTrack.h"
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
class GPUTPCGMSliceTrack
{
 public:
  float Alpha() const { return mAlpha; }
  char Slice() const { return (char)mSlice; }
  char CSide() const { return mSlice >= 18; }
  int NClusters() const { return mNClusters; }
  int PrevNeighbour() const { return mNeighbour[0]; }
  int NextNeighbour() const { return mNeighbour[1]; }
  int Neighbour(int i) const { return mNeighbour[i]; }
  int PrevSegmentNeighbour() const { return mSegmentNeighbour[0]; }
  int NextSegmentNeighbour() const { return mSegmentNeighbour[1]; }
  int SegmentNeighbour(int i) const { return mSegmentNeighbour[i]; }
  const GPUTPCSliceOutTrack* OrigTrack() const { return mOrigTrack; }
  float X() const { return mX; }
  float Y() const { return mY; }
  float Z() const { return mZ; }
  float SinPhi() const { return mSinPhi; }
  float CosPhi() const { return mCosPhi; }
  float SecPhi() const { return mSecPhi; }
  float DzDs() const { return mDzDs; }
  float QPt() const { return mQPt; }
  float ZOffset() const { return mZOffset; }
  float Leg() const { return mLeg; }

  int LocalTrackId() const { return mLocalTrackId; }
  void SetLocalTrackId(int v) { mLocalTrackId = v; }
  int GlobalTrackId(int n) const { return mGlobalTrackIds[n]; }
  void SetGlobalTrackId(int n, int v) { mGlobalTrackIds[n] = v; }

  float MaxClusterZ() { return CAMath::Max(mOrigTrack->Clusters()->GetZ(), (mOrigTrack->Clusters() + mOrigTrack->NClusters() - 1)->GetZ()); }
  float MinClusterZ() { return CAMath::Min(mOrigTrack->Clusters()->GetZ(), (mOrigTrack->Clusters() + mOrigTrack->NClusters() - 1)->GetZ()); }

  void Set(const GPUTPCSliceOutTrack* sliceTr, float alpha, int slice)
  {
    const GPUTPCBaseTrackParam& t = sliceTr->Param();
    mOrigTrack = sliceTr;
    mX = t.GetX();
    mY = t.GetY();
    mZ = t.GetZ();
    mDzDs = t.GetDzDs();
    mSinPhi = t.GetSinPhi();
    mQPt = t.GetQPt();
    mCosPhi = sqrt(1.f - mSinPhi * mSinPhi);
    mSecPhi = 1.f / mCosPhi;
    mAlpha = alpha;
    mSlice = slice;
    mZOffset = t.GetZOffset();
    mNClusters = sliceTr->NClusters();
  }

  void SetGlobalSectorTrackCov()
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

  void SetNClusters(int v) { mNClusters = v; }
  void SetPrevNeighbour(int v) { mNeighbour[0] = v; }
  void SetNextNeighbour(int v) { mNeighbour[1] = v; }
  void SetNeighbor(int v, int i) { mNeighbour[i] = v; }
  void SetPrevSegmentNeighbour(int v) { mSegmentNeighbour[0] = v; }
  void SetNextSegmentNeighbour(int v) { mSegmentNeighbour[1] = v; }
  void SetLeg(unsigned char v) { mLeg = v; }

  void CopyParamFrom(const GPUTPCGMSliceTrack& t)
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

  bool FilterErrors(const GPUParam& param, float maxSinPhi = GPUCA_MAX_SIN_PHI, float sinPhiMargin = 0.f);

  bool TransportToX(float x, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi, bool doCov = true) const;

  bool TransportToXAlpha(float x, float sinAlpha, float cosAlpha, float Bz, GPUTPCGMBorderTrack& b, float maxSinPhi) const;

 private:
  const GPUTPCSliceOutTrack* mOrigTrack;                    // pointer to original slice track
  float mX, mY, mZ, mSinPhi, mDzDs, mQPt, mCosPhi, mSecPhi; // parameters
  float mZOffset;
  float mC0, mC2, mC3, mC5, mC7, mC9, mC10, mC12, mC14; // covariances
  float mAlpha;                                         // alpha angle
  int mSlice;                                           // slice of this track segment
  int mNClusters;                                       // N clusters
  int mNeighbour[2];                                    //
  int mSegmentNeighbour[2];                             //
  int mLocalTrackId;                                    // Corrected local track id in terms of GMSliceTracks array
  int mGlobalTrackIds[2];                               // IDs of associated global tracks
  unsigned char mLeg;                                   // Leg of this track segment
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
