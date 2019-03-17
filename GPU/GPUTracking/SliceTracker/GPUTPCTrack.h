// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCtrack
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
  GPUTPCTrack() : mAlive(0), mFirstHitID(0), mNHits(0), mLocalTrackId(-1), mParam()
  {
  }
  ~GPUTPCTrack() CON_DEFAULT;
#endif //! GPUCA_GPUCODE

  GPUhd() char Alive() const
  {
    return mAlive;
  }
  GPUhd() int NHits() const { return mNHits; }
  GPUhd() int LocalTrackId() const { return mLocalTrackId; }
  GPUhd() int FirstHitID() const { return mFirstHitID; }
  GPUhd() MakeType(const MEM_LG(GPUTPCBaseTrackParam) &) Param() const { return mParam; }

  GPUhd() void SetAlive(bool v) { mAlive = v; }
  GPUhd() void SetNHits(int v) { mNHits = v; }
  GPUhd() void SetLocalTrackId(int v) { mLocalTrackId = v; }
  GPUhd() void SetFirstHitID(int v) { mFirstHitID = v; }

  MEM_TEMPLATE()
  GPUhd() void SetParam(const MEM_TYPE(GPUTPCBaseTrackParam) & v) { mParam = v; }

 private:
  char mAlive;       // flag for mark tracks used by the track merger
  int mFirstHitID;   // index of the first track cell in the track->cell pointer array
  int mNHits;        // number of track cells
  int mLocalTrackId; // Id of local track this global track belongs to, index of this track itself if it is a local track
  MEM_LG(GPUTPCBaseTrackParam)
  mParam; // track parameters

 private:
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACK_H
