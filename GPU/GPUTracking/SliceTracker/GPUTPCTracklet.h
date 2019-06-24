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
  GPUTPCTracklet() : mNHits(0), mFirstRow(0), mLastRow(0), mParam(), mHitWeight(0){};
#endif //! GPUCA_GPUCODE

  GPUhd() int NHits() const
  {
    return mNHits;
  }
  GPUhd() int FirstRow() const { return mFirstRow; }
  GPUhd() int LastRow() const { return mLastRow; }
  GPUhd() int HitWeight() const { return mHitWeight; }
  GPUhd() MakeType(const MEM_LG(GPUTPCBaseTrackParam) &) Param() const { return mParam; }
#ifndef GPUCA_EXTERN_ROW_HITS
  GPUhd() calink RowHit(int i) const
  {
    return mRowHits[i];
  }
  GPUhd() const calink* RowHits() const { return (mRowHits); }
  GPUhd() void SetRowHit(int irow, int ih) { mRowHits[irow] = ih; }
#endif // GPUCA_EXTERN_ROW_HITS

  GPUhd() void SetNHits(int v)
  {
    mNHits = v;
  }
  GPUhd() void SetFirstRow(int v) { mFirstRow = v; }
  GPUhd() void SetLastRow(int v) { mLastRow = v; }
  MEM_CLASS_PRE2()
  GPUhd() void SetParam(const MEM_LG2(GPUTPCBaseTrackParam) & v) { mParam = reinterpret_cast<const MEM_LG(GPUTPCBaseTrackParam)&>(v); }
  GPUhd() void SetHitWeight(const int w) { mHitWeight = w; }

 private:
  int mNHits;    // N hits
  int mFirstRow; // first TPC row
  int mLastRow;  // last TPC row
  MEM_LG(GPUTPCBaseTrackParam)
  mParam; // tracklet parameters
#ifndef GPUCA_EXTERN_ROW_HITS
  calink mRowHits[GPUCA_ROW_COUNT + 1]; // hit index for each TPC row
#endif                                  // GPUCA_EXTERN_ROW_HITS
  int mHitWeight;                       // Hit Weight of Tracklet
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKLET_H
