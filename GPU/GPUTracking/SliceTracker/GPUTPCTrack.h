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

/// \file GPUTPCTrack.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCTRACK_H
#define GPUTPCTRACK_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"
#include "GPUTPCSliceOutCluster.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCTrack
 *
 * The class describes the [partially] reconstructed TPC track [candidate].
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 * The track parameters at both ends are stored separately in the GPUTPCEndPoint class
 */
MEM_CLASS_PRE()
class GPUTPCTrack
{
 public:
#if !defined(GPUCA_GPUCODE)
  GPUTPCTrack() : mFirstHitID(0), mNHits(0), mLocalTrackId(-1), mParam()
  {
  }
  ~GPUTPCTrack() CON_DEFAULT;
#endif //! GPUCA_GPUCODE

  GPUhd() int32_t NHits() const { return mNHits; }
  GPUhd() int32_t LocalTrackId() const { return mLocalTrackId; }
  GPUhd() int32_t FirstHitID() const { return mFirstHitID; }
  GPUhd() MakeType(const MEM_LG(GPUTPCBaseTrackParam) &) Param() const { return mParam; }

  GPUhd() void SetNHits(int32_t v) { mNHits = v; }
  GPUhd() void SetLocalTrackId(int32_t v) { mLocalTrackId = v; }
  GPUhd() void SetFirstHitID(int32_t v) { mFirstHitID = v; }

  MEM_TEMPLATE()
  GPUhd() void SetParam(const MEM_TYPE(GPUTPCBaseTrackParam) & v) { mParam = v; }

  // Only if used as replacement for SliceOutTrack
  GPUhd() static int32_t GetSize(int32_t nClust) { return sizeof(GPUTPCTrack) + nClust * sizeof(GPUTPCSliceOutCluster); }
  GPUhd() const GPUTPCTrack* GetNextTrack() const { return (const GPUTPCTrack*)(((char*)this) + GetSize(mNHits)); }
  GPUhd() GPUTPCTrack* NextTrack() { return (GPUTPCTrack*)(((char*)this) + GetSize(mNHits)); }
  GPUhd() void SetOutTrackCluster(int32_t i, const GPUTPCSliceOutCluster& v) { ((GPUTPCSliceOutCluster*)((char*)this + sizeof(*this)))[i] = v; }
  GPUhd() const GPUTPCSliceOutCluster* OutTrackClusters() const { return (const GPUTPCSliceOutCluster*)((char*)this + sizeof(*this)); }
  GPUhd() const GPUTPCSliceOutCluster& OutTrackCluster(int32_t i) const { return OutTrackClusters()[i]; }

 private:
  int32_t mFirstHitID;   // index of the first track cell in the track->cell pointer array
  int32_t mNHits;        // number of track cells
  int32_t mLocalTrackId; // Id of local track this global track belongs to, index of this track itself if it is a local track
  MEM_LG(GPUTPCBaseTrackParam)
  mParam; // track parameters

 private:
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACK_H
