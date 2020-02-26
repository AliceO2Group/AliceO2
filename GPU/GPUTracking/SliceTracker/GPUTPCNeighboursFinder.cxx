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
#include "GPUCommonMath.h"
using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCNeighboursFinder::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & GPUrestrict() s, processorType& GPUrestrict() tracker)
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
#endif
  if (iThread == 0) {
    s.mIRow = iBlock;
    if (s.mIRow < GPUCA_ROW_COUNT) {
#ifdef GPUCA_GPUCODE
      GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() row = s.mRow;
#else
      GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.mData.mRows[s.mIRow];
#endif
      s.mNHits = row.NHits();

      if ((s.mIRow >= 2) && (s.mIRow <= GPUCA_ROW_COUNT - 3)) {
        s.mIRowUp = s.mIRow + 2;
        s.mIRowDn = s.mIRow - 2;

        // references to the rows above and below

#ifdef GPUCA_GPUCODE
        GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() rowUp = s.mRowUp;
        GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() rowDn = s.mRowDown;
#else
        GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowUp = tracker.mData.mRows[s.mIRowUp];
        GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowDn = tracker.mData.mRows[s.mIRowDn];
#endif
        // the axis perpendicular to the rows
        const float xDn = rowDn.mX;
        const float x = row.mX;
        const float xUp = rowUp.mX;

        // number of hits in rows above and below
        s.mUpNHits = tracker.mData.mRows[s.mIRowUp].mNHits;
        s.mDnNHits = tracker.mData.mRows[s.mIRowDn].mNHits;

        // distance of the rows (absolute and relative)
        s.mUpDx = xUp - x;
        s.mDnDx = xDn - x;
        s.mUpTx = xUp / x;
        s.mDnTx = xDn / x;
        // UpTx/DnTx is used to move the HitArea such that central events are preferred (i.e. vertices
        // coming from y = 0, z = 0).

        // s.mGridUp = tracker.mData.mRows[ s.mIRowUp ].Grid();
        // s.mGridDn = tracker.mData.mRows[ s.mIRowDn ].Grid();
      }
    }
  }
  GPUbarrier();

  // local copies

  if ((s.mIRow <= 1) || (s.mIRow >= GPUCA_ROW_COUNT - 2)) {
#ifdef GPUCA_GPUCODE
    GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() row = s.mRow;
#else
    GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.mData.mRows[s.mIRow];
#endif
    long int lHitNumberOffset = row.mHitNumberOffset;
    for (int ih = iThread; ih < s.mNHits; ih += nThreads) {
      tracker.mData.mLinkUpData[lHitNumberOffset + ih] = CALINK_INVAL;
      tracker.mData.mLinkDownData[lHitNumberOffset + ih] = CALINK_INVAL;
    }
    return;
  }

  const float chi2Cut = 3.f * 3.f * 4 * (s.mUpDx * s.mUpDx + s.mDnDx * s.mDnDx);
// float chi2Cut = 3.*3.*(s.mUpDx*s.mUpDx + s.mDnDx*s.mDnDx ); //SG
#ifdef GPUCA_GPUCODE
  GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() row = s.mRow;
  GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() rowUp = s.mRowUp;
  GPUsharedref() const MEM_LOCAL(GPUTPCRow) & GPUrestrict() rowDn = s.mRowDown;
#else
  GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() row = tracker.mData.mRows[s.mIRow];
  GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowUp = tracker.mData.mRows[s.mIRowUp];
  GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & GPUrestrict() rowDn = tracker.mData.mRows[s.mIRowDn];
#endif
  const float y0 = row.mGrid.mYMin;
  const float z0 = row.mGrid.mZMin;
  const float stepY = row.mHstepY;
  const float stepZ = row.mHstepZ;

  const long int lHitNumberOffset = row.mHitNumberOffset;
  const long int lHitNumberOffsetUp = rowUp.mHitNumberOffset;
  const long int lHitNumberOffsetDn = rowDn.mHitNumberOffset;
  const int lFirstHitInBinOffsetUp = rowUp.mFirstHitInBinOffset;
  const int lFirstHitInBinOffsetDn = rowDn.mFirstHitInBinOffset;
  const calink* GPUrestrict() lFirstHitInBin = tracker.mData.mFirstHitInBin;
  const cahit2* GPUrestrict() pHitData = tracker.mData.mHitData;
  const float y0Up = rowUp.mGrid.mYMin;
  const float z0Up = rowUp.mGrid.mZMin;
  const float stepYUp = rowUp.mHstepY;
  const float stepZUp = rowUp.mHstepZ;
  const float y0Dn = rowDn.mGrid.mYMin;
  const float z0Dn = rowDn.mGrid.mZMin;
  const float stepYDn = rowDn.mHstepY;
  const float stepZDn = rowDn.mHstepZ;

  for (int ih = iThread; ih < s.mNHits; ih += nThreads) {
    int linkUp = -1;
    int linkDn = -1;

    if (s.mDnNHits > 0 && s.mUpNHits > 0) {
      const cahit2 hitData = tracker.mData.mHitData[lHitNumberOffset + ih];
      const float y = y0 + (hitData.x) * stepY;
      const float z = z0 + (hitData.y) * stepZ;

#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP < GPUCA_MAXN
      calink neighUp[GPUCA_MAXN - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP];
      float yzUp[GPUCA_MAXN - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP];
      float yzUp2[GPUCA_MAXN - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP];
#endif // GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0

      int nNeighUp = 0;
      float minZ, maxZ, minY, maxY;
      int binYmin, binYmax, binZmin, binZmax;
      int nY;

      const float kAngularMultiplier = tracker.mConstantMem->param.rec.SearchWindowDZDR;
      const float kAreaSize = tracker.mConstantMem->param.rec.NeighboursSearchArea;

      {
        const float yy = y * s.mUpTx;
        const float zz = kAngularMultiplier != 0.f ? z : (z * s.mUpTx);
        const float dy = kAreaSize;
        const float dz = kAngularMultiplier != 0.f ? (s.mUpDx * kAngularMultiplier) : kAreaSize;
        minZ = zz - dz;
        maxZ = zz + dz;
        minY = yy - dy;
        maxY = yy + dy;
        rowUp.Grid().GetBin(minY, minZ, &binYmin, &binZmin);
        rowUp.Grid().GetBin(maxY, maxZ, &binYmax, &binZmax);
        nY = rowUp.Grid().Ny();
      }

      bool dobreak = false;
      for (int k1 = binZmin; k1 <= binZmax; k1++) {
        int iMin = lFirstHitInBin[lFirstHitInBinOffsetUp + k1 * nY + binYmin];
        int iMax = lFirstHitInBin[lFirstHitInBinOffsetUp + k1 * nY + binYmax + 1];
        for (int i = iMin; i < iMax; i++) {
          const cahit2 hitDataUp = pHitData[lHitNumberOffsetUp + i];
          GPUTPCHit h;
          h.mY = y0Up + (hitDataUp.x) * stepYUp;
          h.mZ = z0Up + (hitDataUp.y) * stepZUp;
          if (h.mY < minY || h.mY > maxY || h.mZ < minZ || h.mZ > maxZ)
            continue;

#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP < GPUCA_MAXN
#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP == 0
          if (true) {
#else
          if ((unsigned int)nNeighUp >= GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP) {
#endif
            neighUp[nNeighUp - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP] = (calink)i;
            yzUp[nNeighUp - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP] = s.mDnDx * (h.Y() - y);
            yzUp2[nNeighUp - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP] = s.mDnDx * (h.Z() - z);
          } else
#endif
          {
#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
            s.mB[nNeighUp][iThread] = (calink)i;
            s.mA1[nNeighUp][iThread] = s.mDnDx * (h.Y() - y);
            s.mA2[nNeighUp][iThread] = s.mDnDx * (h.Z() - z);
#endif
          }
          if (++nNeighUp >= GPUCA_MAXN) {
            dobreak = true;
            break;
          }
        }
        if (dobreak)
          break;
      }

      if (nNeighUp > 0) {
        {
          const float yy = y * s.mDnTx;
          const float zz = kAngularMultiplier != 0.f ? z : (z * s.mDnTx);
          const float dy = kAreaSize;
          const float dz = kAngularMultiplier != 0.f ? (-s.mDnDx * kAngularMultiplier) : kAreaSize;
          minZ = zz - dz;
          maxZ = zz + dz;
          minY = yy - dy;
          maxY = yy + dy;
          rowDn.Grid().GetBin(minY, minZ, &binYmin, &binZmin);
          rowDn.Grid().GetBin(maxY, maxZ, &binYmax, &binZmax);
          nY = rowDn.Grid().Ny();
        }
        int bestDn = -1, bestUp = -1;
        float bestD = 1.e10f;

        int nNeighDn = 0;
        for (int k1 = binZmin; k1 <= binZmax; k1++) {
          int iMin = lFirstHitInBin[lFirstHitInBinOffsetDn + k1 * nY + binYmin];
          int iMax = lFirstHitInBin[lFirstHitInBinOffsetDn + k1 * nY + binYmax + 1];
          for (int i = iMin; i < iMax; i++) {
            const cahit2 hitDataDn = pHitData[lHitNumberOffsetDn + i];
            GPUTPCHit h;
            h.mY = y0Dn + (hitDataDn.x) * stepYDn;
            h.mZ = z0Dn + (hitDataDn.y) * stepZDn;
            if (h.mY < minY || h.mY > maxY || h.mZ < minZ || h.mZ > maxZ)
              continue;

            nNeighDn++;
            float2 yzdn = CAMath::MakeFloat2(s.mUpDx * (h.Y() - y), s.mUpDx * (h.Z() - z));

            for (int iUp = 0; iUp < nNeighUp; iUp++) {
#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0 && GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP < GPUCA_MAXN
              float2 yzup = iUp >= GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP ? CAMath::MakeFloat2(yzUp[iUp - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP], yzUp2[iUp - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP]) : CAMath::MakeFloat2(s.mA1[iUp][iThread], s.mA2[iUp][iThread]);
#elif GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP == GPUCA_MAXN
              const float2 yzup = CAMath::MakeFloat2(s.mA1[iUp][iThread], s.mA2[iUp][iThread]);
#else
              const float2 yzup = CAMath::MakeFloat2(yzUp[iUp], yzUp2[iUp]);
#endif
              const float dy = yzdn.x - yzup.x;
              const float dz = yzdn.y - yzup.y;
              const float d = dy * dy + dz * dz;
              if (d < bestD) {
                bestD = d;
                bestDn = i;
                bestUp = iUp;
              }
            }
          }
        }

        if (bestD <= chi2Cut) {
#if GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP > 0 && GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP < GPUCA_MAXN
          linkUp = bestUp >= GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP ? neighUp[bestUp - GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP] : s.mB[bestUp][iThread];
#elif GPUCA_NEIGHBOURS_FINDER_MAX_NNEIGHUP == GPUCA_MAXN
          linkUp = s.mB[bestUp][iThread];
#else
          linkUp = neighUp[bestUp];
#endif
          linkDn = bestDn;
        }
      }
    }

    tracker.mData.mLinkUpData[lHitNumberOffset + ih] = linkUp;
    tracker.mData.mLinkDownData[lHitNumberOffset + ih] = linkDn;
  }
}
