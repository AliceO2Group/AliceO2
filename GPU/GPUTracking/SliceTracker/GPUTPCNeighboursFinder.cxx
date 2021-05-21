// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNeighboursFinder.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCHit.h"
#include "GPUTPCNeighboursFinder.h"
#include "GPUTPCTracker.h"
//#include "GPUCommonMath.h"
#include "GPUDefMacros.h"
using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCNeighboursFinder::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & s, processorType& GPUrestrict() tracker)
{
  //* find neighbours

#ifdef GPUCA_GPUCODE
  for (unsigned int i = iThread; i < sizeof(MEM_PLAIN(GPUTPCRow)) / sizeof(int); i += nThreads) {
    reinterpret_cast<GPUsharedref() int*>(&s.mRow)[i] = reinterpret_cast<GPUglobalref() int*>(&tracker.SliceDataRows()[iBlock])[i];
    if (iBlock >= 2 && iBlock < GPUCA_ROW_COUNT - 2) {
      reinterpret_cast<GPUsharedref() int*>(&s.mRowUp)[i] = reinterpret_cast<GPUglobalref() int*>(&tracker.SliceDataRows()[iBlock + 2])[i];
      reinterpret_cast<GPUsharedref() int*>(&s.mRowDown)[i] = reinterpret_cast<GPUglobalref() int*>(&tracker.SliceDataRows()[iBlock - 2])[i];
    }
  }
  GPUbarrier();
  const GPUsharedref() MEM_LOCAL(GPUTPCRow) & GPUrestrict() row = s.mRow;
  const GPUsharedref() MEM_LOCAL(GPUTPCRow) & GPUrestrict() rowUp = s.mRowUp;
  const GPUsharedref() MEM_LOCAL(GPUTPCRow) & GPUrestrict() rowDn = s.mRowDown;
#else
  const GPUglobalref() MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.mData.mRows[iBlock];
  const GPUglobalref() MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowUp = tracker.mData.mRows[iBlock + 2];
  const GPUglobalref() MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowDn = tracker.mData.mRows[iBlock - 2];
#endif

  if (iThread == 0) {
    s.mIRow = iBlock;
    s.mIRowUp = iBlock + 2;
    s.mIRowDn = iBlock - 2;
    if (s.mIRow < GPUCA_ROW_COUNT) {
      s.mNHits = row.mNHits;
      if ((s.mIRow >= 2) && (s.mIRow <= GPUCA_ROW_COUNT - 3)) {
        // the axis perpendicular to the rows
        const float xDn = rowDn.mX;
        const float x = row.mX;
        const float xUp = rowUp.mX;

        // distance of the rows (absolute and relative)
        s.mUpDx = xUp - x;
        s.mDnDx = xDn - x;
        s.mUpTx = xUp / x;
        s.mDnTx = xDn / x;
      }
    }
  }
  GPUbarrier();

  // local copies

  if ((s.mIRow <= 1) || (s.mIRow >= GPUCA_ROW_COUNT - 2) || (rowUp.mNHits < 0) || (rowDn.mNHits < 0)) {
    const int lHitNumberOffset = row.mHitNumberOffset;
    for (int ih = iThread; ih < s.mNHits; ih += nThreads) {
      tracker.mData.mLinkUpData[lHitNumberOffset + ih] = CALINK_INVAL;
      tracker.mData.mLinkDownData[lHitNumberOffset + ih] = CALINK_INVAL;
    }
    return;
  }

#define UnrollGlobal 4
#define MaxShared GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP
#if MaxShared < GPUCA_MAXN
#define MaxGlobal ((GPUCA_MAXN - MaxShared - 1) / UnrollGlobal + 1) * UnrollGlobal
#else
#define MaxGlobal 0
#endif
#define MaxTotal MaxShared + MaxGlobal

  const float chi2Cut = 3.f * 3.f * 4 * (s.mUpDx * s.mUpDx + s.mDnDx * s.mDnDx);
  // float chi2Cut = 3.*3.*(s.mUpDx*s.mUpDx + s.mDnDx*s.mDnDx ); //SG

  const int lHitNumberOffset = row.mHitNumberOffset;
  const int lHitNumberOffsetUp = rowUp.mHitNumberOffset;
  const int lHitNumberOffsetDn = rowDn.mHitNumberOffset;
  const unsigned int lFirstHitInBinOffsetUp = rowUp.mFirstHitInBinOffset;
  const unsigned int lFirstHitInBinOffsetDn = rowDn.mFirstHitInBinOffset;
  const GPUglobalref() calink* GPUrestrict() lFirstHitInBin = tracker.mData.mFirstHitInBin;
  const GPUglobalref() cahit2* GPUrestrict() pHitData = tracker.mData.mHitData;

  const float y0 = row.mGrid.mYMin;
  const float z0 = row.mGrid.mZMin;
  const float stepY = row.mHstepY;
  const float stepZ = row.mHstepZ;

  const float y0Up = rowUp.mGrid.mYMin;
  const float z0Up = rowUp.mGrid.mZMin;
  const float stepYUp = rowUp.mHstepY;
  const float stepZUp = rowUp.mHstepZ;

  const float y0Dn = rowDn.mGrid.mYMin;
  const float z0Dn = rowDn.mGrid.mZMin;
  const float stepYDn = rowDn.mHstepY;
  const float stepZDn = rowDn.mHstepZ;

  const float kAngularMultiplier = tracker.mConstantMem->param.rec.tpc.searchWindowDZDR;
  const float kAreaSizeY = tracker.mConstantMem->param.rec.tpc.neighboursSearchArea;
  const float kAreaSizeZUp = kAngularMultiplier != 0.f ? (s.mUpDx * kAngularMultiplier) : kAreaSizeY;
  const float kAreaSizeZDn = kAngularMultiplier != 0.f ? (-s.mDnDx * kAngularMultiplier) : kAreaSizeY;
  const float kAreaSlopeZUp = kAngularMultiplier != 0.f ? 1.f : s.mUpTx;
  const float kAreaSlopeZDn = kAngularMultiplier != 0.f ? 1.f : s.mDnTx;

#if MaxGlobal > 0
  calink neighUp[MaxGlobal];
  float yzUp[2 * MaxGlobal];
#endif

  for (int ih = iThread; ih < s.mNHits; ih += nThreads) {

    const GPUglobalref() cahit2& hitData = pHitData[lHitNumberOffset + ih];
    const float y = y0 + hitData.x * stepY;
    const float z = z0 + hitData.y * stepZ;

    int nNeighUp = 0;
    float minZ, maxZ, minY, maxY;
    int binYmin, binYmax, binZmin, binZmax;
    int nY;

    { // area in the upper row
      const float yy = y * s.mUpTx;
      const float zz = z * kAreaSlopeZUp;
      minZ = zz - kAreaSizeZUp;
      maxZ = zz + kAreaSizeZUp;
      minY = yy - kAreaSizeY;
      maxY = yy + kAreaSizeY;
      rowUp.Grid().GetBin(minY, minZ, &binYmin, &binZmin);
      rowUp.Grid().GetBin(maxY, maxZ, &binYmax, &binZmax);
      nY = rowUp.Grid().Ny();
    }

    for (int k1 = binZmin; k1 <= binZmax && (nNeighUp < MaxTotal); k1++) {
      int iMin = lFirstHitInBin[lFirstHitInBinOffsetUp + k1 * nY + binYmin];
      int iMax = lFirstHitInBin[lFirstHitInBinOffsetUp + k1 * nY + binYmax + 1];
      GPUCA_UNROLL(U(4), U(2))
      for (int i = iMin; i < iMax && (nNeighUp < MaxTotal); i++) {
        const GPUglobalref() cahit2& hitDataUp = pHitData[lHitNumberOffsetUp + i];
        GPUTPCHit h;
        h.mY = y0Up + (hitDataUp.x) * stepYUp;
        h.mZ = z0Up + (hitDataUp.y) * stepZUp;

        if (h.mY < minY || h.mY > maxY || h.mZ < minZ || h.mZ > maxZ) {
          continue;
        }

#if MaxGlobal > 0
#if MaxShared == 0
        if (true) {
#else
        if (nNeighUp >= MaxShared) {
#endif
          neighUp[nNeighUp - MaxShared] = (calink)i;
          yzUp[2 * (nNeighUp - MaxShared)] = s.mDnDx * (h.Y() - y);
          yzUp[2 * (nNeighUp - MaxShared) + 1] = s.mDnDx * (h.Z() - z);
        } else
#endif
        {
#if MaxShared > 0
          s.mB[nNeighUp][iThread] = (calink)i;
          s.mA1[nNeighUp][iThread] = s.mDnDx * (h.Y() - y);
          s.mA2[nNeighUp][iThread] = s.mDnDx * (h.Z() - z);
#endif
        }
        nNeighUp++;
      }
    }

#if MaxShared > 0 // init a rest of the shared array
    for (int iUp = nNeighUp; iUp < MaxShared; iUp++) {
      s.mA1[iUp][iThread] = -1.e10f;
      s.mA2[iUp][iThread] = -1.e10f;
      s.mB[iUp][iThread] = (calink)-1;
    }
#endif

#if MaxGlobal > 0 // init a rest of the UnrollGlobal chunk of the global array
    int Nrest = nNeighUp - MaxShared;
    int N4 = (Nrest / UnrollGlobal) * UnrollGlobal;
    if (N4 < Nrest) {
      N4 += UnrollGlobal;
      GPUCA_UNROLL(U(UnrollGlobal - 1), U(UnrollGlobal - 1))
      for (int k = 0; k < UnrollGlobal - 1; k++) {
        if (Nrest + k < N4) {
          yzUp[2 * (Nrest + k)] = -1.e10f;
          yzUp[2 * (Nrest + k) + 1] = -1.e10f;
          neighUp[Nrest + k] = (calink)-1;
        }
      }
    }
#endif

    { // area in the lower row
      const float yy = y * s.mDnTx;
      const float zz = z * kAreaSlopeZDn;
      minZ = zz - kAreaSizeZDn;
      maxZ = zz + kAreaSizeZDn;
      minY = yy - kAreaSizeY;
      maxY = yy + kAreaSizeY;
    }
    rowDn.Grid().GetBin(minY, minZ, &binYmin, &binZmin);
    rowDn.Grid().GetBin(maxY, maxZ, &binYmax, &binZmax);
    nY = rowDn.Grid().Ny();

    int linkUp = -1;
    int linkDn = -1;
    float bestD = chi2Cut;

    for (int k1 = binZmin; k1 <= binZmax; k1++) {
      int iMin = lFirstHitInBin[lFirstHitInBinOffsetDn + k1 * nY + binYmin];
      int iMax = lFirstHitInBin[lFirstHitInBinOffsetDn + k1 * nY + binYmax + 1];
      for (int i = iMin; i < iMax; i++) {
        const GPUglobalref() cahit2& hitDataDn = pHitData[lHitNumberOffsetDn + i];
        float yDn = y0Dn + (hitDataDn.x) * stepYDn;
        float zDn = z0Dn + (hitDataDn.y) * stepZDn;

        if (yDn < minY || yDn > maxY || zDn < minZ || zDn > maxZ) {
          continue;
        }

        float yDnProjUp = s.mUpDx * (yDn - y);
        float zDnProjUp = s.mUpDx * (zDn - z);

#if MaxShared > 0
        GPUCA_UNROLL(U(MaxShared), U(MaxShared))
        for (int iUp = 0; iUp < MaxShared; iUp++) {
          const float dy = yDnProjUp - s.mA1[iUp][iThread];
          const float dz = zDnProjUp - s.mA2[iUp][iThread];
          const float d = dy * dy + dz * dz;
          if (d < bestD) {
            bestD = d;
            linkDn = i;
            linkUp = iUp;
          }
        }
#endif

#if MaxGlobal > 0
        for (int iUp = 0; iUp < N4; iUp += UnrollGlobal) {
          GPUCA_UNROLL(U(UnrollGlobal), U(UnrollGlobal))
          for (int k = 0; k < UnrollGlobal; k++) {
            int jUp = iUp + k;
            const float dy = yDnProjUp - yzUp[2 * jUp];
            const float dz = zDnProjUp - yzUp[2 * jUp + 1];
            const float d = dy * dy + dz * dz;
            if (d < bestD) {
              bestD = d;
              linkDn = i;
              linkUp = MaxShared + jUp;
            }
          }
        }
#endif
      }
    }

    if (linkUp >= 0) {
#if MaxShared > 0 && MaxGlobal > 0
      linkUp = (linkUp >= MaxShared) ? neighUp[linkUp - MaxShared] : s.mB[linkUp][iThread];
#elif MaxShared > 0
      linkUp = s.mB[linkUp][iThread];
#else
      linkUp = neighUp[linkUp];
#endif
    }

    tracker.mData.mLinkUpData[lHitNumberOffset + ih] = linkUp;
    tracker.mData.mLinkDownData[lHitNumberOffset + ih] = linkDn;
  }
}
