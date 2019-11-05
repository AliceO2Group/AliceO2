// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCHitArea.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCGrid.h"
#include "GPUTPCHit.h"
#include "GPUTPCHitArea.h"
#include "GPUTPCTracker.h"
#include "GPUConstantMem.h"
using namespace GPUCA_NAMESPACE::gpu;

MEM_CLASS_PRE()
class GPUTPCRow;

MEM_TEMPLATE()
GPUd() void GPUTPCHitArea::Init(const MEM_TYPE(GPUTPCRow) & row, GPUconstantref() const MEM_GLOBAL(GPUTPCSliceData) & slice, float y, float z, float dy, float dz)
{
  // initialisation
  mHitOffset = row.HitNumberOffset();
  mY = y;
  mZ = z;
  mMinZ = z - dz;
  mMaxZ = z + dz;
  mMinY = y - dy;
  mMaxY = y + dy;
  int bYmin, bZmin, bYmax; // boundary bin indexes
  row.Grid().GetBin(mMinY, mMinZ, &bYmin, &bZmin);
  row.Grid().GetBin(mMaxY, mMaxZ, &bYmax, &mBZmax);
  mBDY = bYmax - bYmin + 1; // bin index span in y direction
  mNy = row.Grid().Ny();
  mIndYmin = bZmin * mNy + bYmin; // same as grid.GetBin(mMinY, mMinZ), i.e. the smallest bin index of interest
  // mIndYmin + mBDY then is the largest bin index of interest with the same Z
  mIz = bZmin;

// for given mIz (which is min atm.) get
#ifdef GPUCA_TEXTURE_FETCH_NEIGHBORS
  mHitYfst = tex1Dfetch(gAliTexRefu, ((char*)slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + mIndYmin);
  mHitYlst = tex1Dfetch(gAliTexRefu, ((char*)slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + mIndYmin + mBDY);
#else
  mHitYfst = slice.FirstHitInBin(row, mIndYmin);        // first and
  mHitYlst = slice.FirstHitInBin(row, mIndYmin + mBDY); // last hit index in the bin
#endif // GPUCA_TEXTURE_FETCH_NEIGHBORS
  mIh = mHitYfst;
}

MEM_TEMPLATE()
GPUd() int GPUTPCHitArea::GetNext(GPUconstantref() const MEM_GLOBAL(GPUTPCTracker) & tracker, const MEM_TYPE(GPUTPCRow) & row, GPUconstantref() const MEM_GLOBAL(GPUTPCSliceData) & slice, GPUTPCHit* h)
{
  // get next hit index

  // min coordinate
  const float y0 = row.Grid().YMin();
  const float z0 = row.Grid().ZMin();

  // step vector
  const float stepY = row.HstepY();
  const float stepZ = row.HstepZ();

  int ret = -1;
  do {
    while (mIh >= mHitYlst) {
      if (mIz >= mBZmax) {
        return -1;
      }
      // go to next z and start y from the min again
      ++mIz;
      mIndYmin += mNy;
#ifdef GPUCA_TEXTURE_FETCH_NEIGHBORS
      mHitYfst = tex1Dfetch(gAliTexRefu, ((char*)slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + mIndYmin);
      mHitYlst = tex1Dfetch(gAliTexRefu, ((char*)slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + mIndYmin + mBDY);
#else
      mHitYfst = slice.FirstHitInBin(row, mIndYmin);
      mHitYlst = slice.FirstHitInBin(row, mIndYmin + mBDY);
#endif
      mIh = mHitYfst;
    }

#ifdef GPUCA_TEXTURE_FETCH_NEIGHBORS
    cahit2 tmpval = tex1Dfetch(gAliTexRefu2, ((char*)slice.HitData(row) - slice.GPUTextureBaseConst()) / sizeof(cahit2) + mIh);
    h->SetY(y0 + tmpval.x * stepY);
    h->SetZ(z0 + tmpval.y * stepZ);
#else
    h->SetY(y0 + tracker.HitDataY(row, mIh) * stepY);
    h->SetZ(z0 + tracker.HitDataZ(row, mIh) * stepZ);
#endif

    if (h->Z() > mMaxZ || h->Z() < mMinZ || h->Y() < mMinY || h->Y() > mMaxY) {
      mIh++;
      continue;
    }
    ret = mIh;
    mIh++;
    break;
  } while (1);
  return ret;
}
