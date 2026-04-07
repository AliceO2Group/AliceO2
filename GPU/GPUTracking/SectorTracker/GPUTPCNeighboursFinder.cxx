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

/// \file GPUTPCNeighboursFinder.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCHit.h"
#include "GPUTPCNeighboursFinder.h"
#include "GPUTPCTracker.h"
// #include "GPUCommonMath.h"
#include "GPUDefMacros.h"
using namespace o2::gpu;

template <>
GPUdii() void GPUTPCNeighboursFinder::Thread<0>(int32_t /*nBlocks*/, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  //* find neighbours

#ifdef GPUCA_GPUCODE
  for (uint32_t i = iThread; i < sizeof(GPUTPCRow) / sizeof(int32_t); i += nThreads) {
    reinterpret_cast<GPUsharedref() int32_t*>(&s.mRow)[i] = reinterpret_cast<GPUglobalref() int32_t*>(&tracker.TrackingDataRows()[iBlock])[i];
    if (iBlock >= 2 && iBlock < GPUCA_NROWS - 2) {
      reinterpret_cast<GPUsharedref() int32_t*>(&s.mRowUp)[i] = reinterpret_cast<GPUglobalref() int32_t*>(&tracker.TrackingDataRows()[iBlock + 2])[i];
      reinterpret_cast<GPUsharedref() int32_t*>(&s.mRowDown)[i] = reinterpret_cast<GPUglobalref() int32_t*>(&tracker.TrackingDataRows()[iBlock - 2])[i];
    }
  }
  GPUbarrier();
  const GPUsharedref() GPUTPCRow& GPUrestrict() row = s.mRow;
  const GPUsharedref() GPUTPCRow& GPUrestrict() rowUp = s.mRowUp;
  const GPUsharedref() GPUTPCRow& GPUrestrict() rowDn = s.mRowDown;
#else
  const GPUglobalref() GPUTPCRow& GPUrestrict() row = tracker.mData.mRows[iBlock];
  const GPUglobalref() GPUTPCRow& GPUrestrict() rowUp = tracker.mData.mRows[iBlock + 2];
  const GPUglobalref() GPUTPCRow& GPUrestrict() rowDn = tracker.mData.mRows[iBlock - 2];
#endif

  if (iThread == 0) {
    s.mIRow = iBlock;
    s.mIRowUp = iBlock + 2;
    s.mIRowDn = iBlock - 2;
    if (s.mIRow < GPUCA_NROWS) {
      s.mNHits = row.mNHits;
      if ((s.mIRow >= 2) && (s.mIRow <= GPUCA_NROWS - 3)) {
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

  if ((s.mIRow <= 1) || (s.mIRow >= GPUCA_NROWS - 2) || (rowUp.mNHits <= 0) || (rowDn.mNHits <= 0)) {
    const int32_t lHitNumberOffset = row.mHitNumberOffset;
    for (int32_t ih = iThread; ih < s.mNHits; ih += nThreads) {
      tracker.mData.mLinkUpData[lHitNumberOffset + ih] = CALINK_INVAL;
      tracker.mData.mLinkDownData[lHitNumberOffset + ih] = CALINK_INVAL;
    }
    return;
  }

  static constexpr uint32_t UNROLL_GLOBAL = GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_GLOBAL > 1 ? GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_GLOBAL : 1;
  static_assert(GPUCA_MAXN % UNROLL_GLOBAL == 0);
  static constexpr uint32_t MAX_SHARED = GPUCA_PAR_NEIGHBOURS_FINDER_MAX_NNEIGHUP;
  static constexpr uint32_t MAX_GLOBAL = (MAX_SHARED < GPUCA_MAXN) ? (((GPUCA_MAXN - MAX_SHARED - 1) / UNROLL_GLOBAL + 1) * UNROLL_GLOBAL) : 0;
  static constexpr uint32_t MAX_TOTAL = MAX_SHARED + MAX_GLOBAL;

  const float chi2Cut = 3.f * 3.f * 4 * (s.mUpDx * s.mUpDx + s.mDnDx * s.mDnDx);
  // float chi2Cut = 3.f*3.f*(s.mUpDx*s.mUpDx + s.mDnDx*s.mDnDx ); //SG

  const int32_t lHitNumberOffset = row.mHitNumberOffset;
  const int32_t lHitNumberOffsetUp = rowUp.mHitNumberOffset;
  const int32_t lHitNumberOffsetDn = rowDn.mHitNumberOffset;
  const uint32_t lFirstHitInBinOffsetUp = rowUp.mFirstHitInBinOffset;
  const uint32_t lFirstHitInBinOffsetDn = rowDn.mFirstHitInBinOffset;
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

  calink neighUp[MAX_GLOBAL];
  float yzUp[2 * MAX_GLOBAL];

  for (int32_t ih = iThread; ih < s.mNHits; ih += nThreads) {

    const GPUglobalref() cahit2& hitData = pHitData[lHitNumberOffset + ih];
    const float y = y0 + hitData.x * stepY;
    const float z = z0 + hitData.y * stepZ;

    uint32_t nNeighUp = 0;
    float minZ, maxZ, minY, maxY;
    int32_t binYmin, binYmax, binZmin, binZmax;
    int32_t nY;

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

    for (int32_t k1 = binZmin; k1 <= binZmax && (nNeighUp < MAX_TOTAL); k1++) {
      int32_t iMin = lFirstHitInBin[lFirstHitInBinOffsetUp + k1 * nY + binYmin];
      int32_t iMax = lFirstHitInBin[lFirstHitInBinOffsetUp + k1 * nY + binYmax + 1];
      GPUCA_UNROLL(U(4), U(2))
      for (int32_t i = iMin; i < iMax && (nNeighUp < MAX_TOTAL); i++) {
        const GPUglobalref() cahit2& hitDataUp = pHitData[lHitNumberOffsetUp + i];
        GPUTPCHit h;
        h.mY = y0Up + (hitDataUp.x) * stepYUp;
        h.mZ = z0Up + (hitDataUp.y) * stepZUp;

        if (h.mY < minY || h.mY > maxY || h.mZ < minZ || h.mZ > maxZ) {
          continue;
        }

        const bool inGlobal = nNeighUp >= MAX_SHARED;
        if constexpr (MAX_GLOBAL > 0) {
          if (inGlobal) {
            neighUp[nNeighUp - MAX_SHARED] = (calink)i;
            yzUp[2 * (nNeighUp - MAX_SHARED)] = s.mDnDx * (h.Y() - y);
            yzUp[2 * (nNeighUp - MAX_SHARED) + 1] = s.mDnDx * (h.Z() - z);
          }
        }
        if constexpr (MAX_SHARED > 0) {
          if (!inGlobal) {
            s.mB[nNeighUp][iThread] = (calink)i;
            s.mA1[nNeighUp][iThread] = s.mDnDx * (h.Y() - y);
            s.mA2[nNeighUp][iThread] = s.mDnDx * (h.Z() - z);
          }
        }
        nNeighUp++;
      }
    }

    if constexpr (MAX_SHARED > 0 && GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_SHARED) { // init the rest of the shared array
      for (uint32_t iUp = nNeighUp; iUp < MAX_SHARED; iUp++) {
        s.mA1[iUp][iThread] = -1.e10f;
        s.mA2[iUp][iThread] = -1.e10f;
        s.mB[iUp][iThread] = (calink)-1;
      }
    }

    const uint32_t nRest = nNeighUp - MAX_SHARED;
    uint32_t nRestUnrolled = (nRest / UNROLL_GLOBAL) * UNROLL_GLOBAL;
    if constexpr (MAX_GLOBAL > 1) { // init the rest of the UNROLL_GLOBAL chunk of the global array
      if (nNeighUp > MAX_SHARED && nRestUnrolled < nRest) {
        nRestUnrolled += UNROLL_GLOBAL;
        GPUCA_UNROLL(U(UNROLL_GLOBAL - 1), U(UNROLL_GLOBAL - 1))
        for (uint32_t k = 0; k + 1 < UNROLL_GLOBAL; k++) {
          if (nRest + k < nRestUnrolled) {
            yzUp[2 * (nRest + k)] = -1.e10f;
            yzUp[2 * (nRest + k) + 1] = -1.e10f;
            neighUp[nRest + k] = (calink)-1;
          }
        }
      }
    }

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

    int32_t linkUp = -1; // CALINK_INVAL as integer
    int32_t linkDn = -1; // CALINK_INVAL as integer
    float bestD = chi2Cut;

    for (int32_t k1 = binZmin; k1 <= binZmax; k1++) {
      int32_t iMin = lFirstHitInBin[lFirstHitInBinOffsetDn + k1 * nY + binYmin];
      int32_t iMax = lFirstHitInBin[lFirstHitInBinOffsetDn + k1 * nY + binYmax + 1];
      for (int32_t i = iMin; i < iMax; i++) {
        const GPUglobalref() cahit2& hitDataDn = pHitData[lHitNumberOffsetDn + i];
        float yDn = y0Dn + (hitDataDn.x) * stepYDn;
        float zDn = z0Dn + (hitDataDn.y) * stepZDn;

        if (yDn < minY || yDn > maxY || zDn < minZ || zDn > maxZ) {
          continue;
        }

        float yDnProjUp = s.mUpDx * (yDn - y);
        float zDnProjUp = s.mUpDx * (zDn - z);

        if constexpr (MAX_SHARED > 0) {
          const uint32_t maxSharedUp = GPUCA_PAR_NEIGHBOURS_FINDER_UNROLL_SHARED ? MAX_SHARED : CAMath::Min(nNeighUp, MAX_SHARED);
          GPUCA_UNROLL(U(MAX_SHARED), U(MAX_SHARED))
          for (uint32_t iUp = 0; iUp < maxSharedUp; iUp++) {
            const float dy = yDnProjUp - s.mA1[iUp][iThread];
            const float dz = zDnProjUp - s.mA2[iUp][iThread];
            const float d = dy * dy + dz * dz;
            if (d < bestD) {
              bestD = d;
              linkDn = i;
              linkUp = iUp;
            }
          }
        }

        if constexpr (MAX_GLOBAL > 0) {
          if (nNeighUp > MAX_SHARED) {
            for (uint32_t iUp = 0; iUp < nRestUnrolled; iUp += UNROLL_GLOBAL) {
              GPUCA_UNROLL(U(UNROLL_GLOBAL), U(UNROLL_GLOBAL))
              for (uint32_t k = 0; k < UNROLL_GLOBAL; k++) {
                const uint32_t jUp = iUp + k;
                const float dy = yDnProjUp - yzUp[2 * jUp];
                const float dz = zDnProjUp - yzUp[2 * jUp + 1];
                const float d = dy * dy + dz * dz;
                if (d < bestD) {
                  bestD = d;
                  linkDn = i;
                  linkUp = MAX_SHARED + jUp;
                }
              }
            }
          }
        }
      }
    }

    if (linkUp >= 0) {
      if constexpr (MAX_SHARED > 0 && MAX_GLOBAL > 0) {
        linkUp = ((uint32_t)linkUp >= MAX_SHARED) ? neighUp[linkUp - MAX_SHARED] : s.mB[linkUp][iThread];
      } else if constexpr (MAX_SHARED > 0) {
        linkUp = s.mB[linkUp][iThread];
      } else {
        linkUp = neighUp[linkUp];
      }
    }

    tracker.mData.mLinkUpData[lHitNumberOffset + ih] = linkUp;
    tracker.mData.mLinkDownData[lHitNumberOffset + ih] = linkDn;
  }
}
