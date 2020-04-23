// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTracklet.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKLET_H
#define GPUTPCTRACKLET_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCTracklet
 *
 * The class describes the reconstructed TPC track candidate.
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 */
MEM_CLASS_PRE()
class GPUTPCTracklet
{
 public:
#if !defined(GPUCA_GPUCODE)
  GPUTPCTracklet() : mNHits(0), mFirstRow(0), mLastRow(0), mParam(), mHitWeight(0), mFirstHit(0){};
#endif //! GPUCA_GPUCODE

  GPUhd() int NHits() const
  {
    return mNHits;
  }
  GPUhd() int FirstRow() const { return mFirstRow; }
  GPUhd() int LastRow() const { return mLastRow; }
  GPUhd() int HitWeight() const { return mHitWeight; }
  GPUhd() unsigned int FirstHit() const { return mFirstHit; }
  GPUhd() MakeType(const MEM_LG(GPUTPCBaseTrackParam) &) Param() const { return mParam; }

  GPUhd() void SetNHits(int v)
  {
    mNHits = v;
  }
  GPUhd() void SetFirstRow(int v) { mFirstRow = v; }
  GPUhd() void SetLastRow(int v) { mLastRow = v; }
  GPUhd() void SetFirstHit(unsigned int v) { mFirstHit = v; }
  MEM_CLASS_PRE2()
  GPUhd() void SetParam(const MEM_LG2(GPUTPCBaseTrackParam) & v) { mParam = reinterpret_cast<const MEM_LG(GPUTPCBaseTrackParam)&>(v); }
  GPUhd() void SetHitWeight(const int w) { mHitWeight = w; }

 private:
  int mNHits;    // N hits
  int mFirstRow; // first TPC row
  int mLastRow;  // last TPC row
  MEM_LG(GPUTPCBaseTrackParam)
  mParam;                 // tracklet parameters
  int mHitWeight;         // Hit Weight of Tracklet
  unsigned int mFirstHit; // first hit in row hit array
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKLET_H
