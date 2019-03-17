// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCRow.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCROW_H
#define GPUTPCROW_H

#include "GPUTPCDef.h"
#include "GPUTPCGrid.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCRow
 *
 * The GPUTPCRow class is a hit and cells container for one TPC row.
 * It is the internal class of the GPUTPCTracker algorithm.
 *
 */
MEM_CLASS_PRE()
class GPUTPCRow
{
  MEM_CLASS_PRE2()
  friend class GPUTPCSliceData;

 public:
#if !defined(GPUCA_GPUCODE)
  GPUTPCRow();
#endif //! GPUCA_GPUCODE

  GPUhd() int NHits() const
  {
    return mNHits;
  }
  GPUhd() float X() const { return mX; }
  GPUhd() float MaxY() const { return mMaxY; }
  GPUhd() MakeType(const GPUTPCGrid&) Grid() const { return mGrid; }

  GPUhd() float Hy0() const { return mHy0; }
  GPUhd() float Hz0() const { return mHz0; }
  GPUhd() float HstepY() const { return mHstepY; }
  GPUhd() float HstepZ() const { return mHstepZ; }
  GPUhd() float HstepYi() const { return mHstepYi; }
  GPUhd() float HstepZi() const { return mHstepZi; }
  GPUhd() int FullSize() const { return mFullSize; }
  GPUhd() int HitNumberOffset() const { return mHitNumberOffset; }
  GPUhd() unsigned int FirstHitInBinOffset() const { return mFirstHitInBinOffset; }

 private:
  int mNHits;       // number of hits
  float mX;         // X coordinate of the row
  float mMaxY;      // maximal Y coordinate of the row
  GPUTPCGrid mGrid; // grid of hits

  // hit packing:
  float mHy0;     // offset
  float mHz0;     // offset
  float mHstepY;  // step size
  float mHstepZ;  // step size
  float mHstepYi; // inverse step size
  float mHstepZi; // inverse step size

  int mFullSize;        // size of this row in Tracker::mRowData
  int mHitNumberOffset; // index of the first hit in the hit array, used as
  // offset in GPUTPCSliceData::LinkUp/DownData/HitDataY/...
  unsigned int mFirstHitInBinOffset; // offset in Tracker::mRowData to find the FirstHitInBin
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCROW_H
