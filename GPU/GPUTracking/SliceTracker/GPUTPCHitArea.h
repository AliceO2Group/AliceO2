// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCHitArea.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCHITAREA_H
#define GPUTPCHITAREA_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCHit;
class GPUTPCGrid;
MEM_CLASS_PRE()
class GPUTPCTracker;
MEM_CLASS_PRE()
class GPUTPCRow;
MEM_CLASS_PRE()
class GPUTPCSliceData;

/**
 * @class GPUTPCHitArea
 *
 * This class is used to _iterate_ over the hit data via GetNext
 */
class GPUTPCHitArea
{
 public:
  MEM_TEMPLATE()
  GPUd() void Init(const MEM_TYPE(GPUTPCRow) & row, GPUconstantref() const MEM_GLOBAL(GPUTPCSliceData) & slice, float y, float z, float dy, float dz);

  /**
 * look up the next hit in the requested area.
 * Sets h to the coordinates and returns the index for the hit data
 */
  MEM_TEMPLATE()
  GPUd() int GetNext(GPUconstantref() const MEM_GLOBAL(GPUTPCTracker) & tracker, const MEM_TYPE(GPUTPCRow) & row, GPUconstantref() const MEM_GLOBAL(GPUTPCSliceData) & slice, GPUTPCHit* h);

  float Y() const { return mY; }
  float Z() const { return mZ; }
  float MinZ() const { return mMinZ; }
  float MaxZ() const { return mMaxZ; }
  float MinY() const { return mMinY; }
  float MaxY() const { return mMaxY; }
  int BZmax() const { return mBZmax; }
  int BDY() const { return mBDY; }
  int IndYmin() const { return mIndYmin; }
  int Iz() const { return mIz; }
  int HitYfst() const { return mHitYfst; }
  int HitYlst() const { return mHitYlst; }
  int Ih() const { return mIh; }
  int Ny() const { return mNy; }
  int HitOffset() const { return mHitOffset; }

 protected:
  float mY;       // search coordinates
  float mZ;       // search coordinates
  float mMinZ;    // search coordinates
  float mMaxZ;    // search coordinates
  float mMinY;    // search coordinates
  float mMaxY;    // search coordinates
  int mBZmax;     // maximal Z bin index
  int mBDY;       // Y distance of bin indexes
  int mIndYmin;   // minimum index for
  int mIz;        // current Z bin index (incremented while iterating)
  int mHitYfst;   //
  int mHitYlst;   //
  int mIh;        // some XXX index in the hit data
  int mNy;        // Number of bins in Y direction
  int mHitOffset; // global hit offset XXX what's that?
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCHITAREA_H
