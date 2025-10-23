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

namespace o2::gpu
{
/**
 * @class GPUTPCTrack
 *
 * The class describes the [partially] reconstructed TPC track [candidate].
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 * The track parameters at both ends are stored separately in the GPUTPCEndPoint class
 */
class GPUTPCTrack
{
 public:
#if !defined(GPUCA_GPUCODE)
  GPUTPCTrack() : mFirstHitID(0), mNHits(0), mLocalTrackId(-1), mParam()
  {
  }
  ~GPUTPCTrack() = default;
#endif //! GPUCA_GPUCODE

  GPUhd() int32_t NHits() const { return mNHits; }
  GPUhd() int32_t LocalTrackId() const { return mLocalTrackId; }
  GPUhd() int32_t FirstHitID() const { return mFirstHitID; }
  GPUhd() const GPUTPCBaseTrackParam& Param() const { return mParam; }

  GPUhd() void SetNHits(int32_t v) { mNHits = v; }
  GPUhd() void SetLocalTrackId(int32_t v) { mLocalTrackId = v; }
  GPUhd() void SetFirstHitID(int32_t v) { mFirstHitID = v; }

  GPUhd() void SetParam(const GPUTPCBaseTrackParam& v) { mParam = v; }

 private:
  int32_t mFirstHitID;         // index of the first track cell in the track->cell pointer array
  int32_t mNHits;              // number of track cells
  int32_t mLocalTrackId;       // Id of local track this extrapolated track belongs to, index of this track itself if it is a local track
  GPUTPCBaseTrackParam mParam; // track parameters

 private:
};
} // namespace o2::gpu

#endif // GPUTPCTRACK_H
