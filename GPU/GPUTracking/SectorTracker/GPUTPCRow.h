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

/// \file GPUTPCRow.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCROW_H
#define GPUTPCROW_H

#include "GPUTPCDef.h"
#include "GPUTPCGrid.h"

namespace o2::gpu
{
/**
 * @class GPUTPCRow
 *
 * The GPUTPCRow class is a hit and cells container for one TPC row.
 * It is the internal class of the GPUTPCTracker algorithm.
 *
 */
class GPUTPCRow
{
  friend class GPUTPCTrackingData;

 public:
  GPUhd() uint32_t NHits() const
  {
    return mNHits;
  }
  GPUhd() float X() const { return mX; }
  GPUhd() float MaxY() const { return mMaxY; }
  GPUhd() const GPUTPCGrid& Grid() const { return mGrid; }

  GPUhd() float Hy0() const { return mHy0; }
  GPUhd() float Hz0() const { return mHz0; }
  GPUhd() float HstepY() const { return mHstepY; }
  GPUhd() float HstepZ() const { return mHstepZ; }
  GPUhd() float HstepYi() const { return mHstepYi; }
  GPUhd() float HstepZi() const { return mHstepZi; }
  GPUhd() uint32_t HitNumberOffset() const { return mHitNumberOffset; }
  GPUhd() uint32_t FirstHitInBinOffset() const { return mFirstHitInBinOffset; }
  GPUhd() static float getTPCMaxY1X() { return 0.1763269f; } // 0.1763269 = tan(2Pi / (2 * 18))
  GPUhd() float getTPCMaxY() const { return getTPCMaxY1X() * mX; }

 private:
  friend class GPUTPCNeighboursFinder;
  friend class GPUTPCStartHitsFinder;

  uint32_t mNHits;  // number of hits
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

  uint32_t mHitNumberOffset;     // index of the first hit in the hit array, used as offset in GPUTPCTrackingData::LinkUp/DownData/HitDataY/...
  uint32_t mFirstHitInBinOffset; // offset in Tracker::mRowData to find the FirstHitInBin
};
} // namespace o2::gpu

#endif // GPUTPCROW_H
