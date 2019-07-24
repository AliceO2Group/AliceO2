// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTrackletConstructor.cxx
/// \author Sergey Gorbunov, David Rohr

#define GPUCA_CADEBUG 0

#include "GPUTPCDef.h"
#include "GPUTPCGrid.h"
#include "GPUTPCHit.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCTracker.h"
#include "GPUTPCTracklet.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUCommonMath.h"

using namespace GPUCA_NAMESPACE::gpu;

MEM_CLASS_PRE2()
GPUd() void GPUTPCTrackletConstructor::InitTracklet(MEM_LG2(GPUTPCTrackParam) & tParam)
{
  // Initialize Tracklet Parameters using default values
  tParam.InitParam();
}

MEM_CLASS_PRE2()
GPUd() bool GPUTPCTrackletConstructor::CheckCov(MEM_LG2(GPUTPCTrackParam) & tParam)
{
  bool ok = 1;
  const float* c = tParam.Cov();
  for (int i = 0; i < 15; i++) {
    ok = ok && CAMath::Finite(c[i]);
  }
  for (int i = 0; i < 5; i++) {
    ok = ok && CAMath::Finite(tParam.Par()[i]);
  }
  ok = ok && (tParam.X() > 50);
  if (c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0) {
    ok = 0;
  }
  return (ok);
}

MEM_CLASS_PRE23()
GPUd() void GPUTPCTrackletConstructor::StoreTracklet(int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) & s, GPUTPCThreadMemory& r, GPUconstantref() MEM_LG2(GPUTPCTracker) & tracker, MEM_LG3(GPUTPCTrackParam) & tParam)
{
  // reconstruction of tracklets, tracklet store step
  if (r.mNHits && (r.mNHits < GPUCA_TRACKLET_SELECTOR_MIN_HITS(tParam.QPt()) || !CheckCov(tParam) || CAMath::Abs(tParam.GetQPt()) > tracker.Param().rec.MaxTrackQPt)) {
    r.mNHits = 0;
  }

  /*GPUInfo("Tracklet %d: Hits %3d NDF %3d Chi %8.4f Sign %f Cov: %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f", r.mItr, r.mNHits, tParam.GetNDF(), tParam.GetChi2(), tParam.GetSignCosPhi(),
          tParam.Cov()[0], tParam.Cov()[1], tParam.Cov()[2], tParam.Cov()[3], tParam.Cov()[4], tParam.Cov()[5], tParam.Cov()[6], tParam.Cov()[7], tParam.Cov()[8], tParam.Cov()[9],
          tParam.Cov()[10], tParam.Cov()[11], tParam.Cov()[12], tParam.Cov()[13], tParam.Cov()[14]);*/

  GPUglobalref() MEM_GLOBAL(GPUTPCTracklet)& tracklet = tracker.Tracklets()[r.mItr];

  tracklet.SetNHits(r.mNHits);
  CADEBUG(GPUInfo("    DONE %d hits", r.mNHits));

  if (r.mNHits > 0) {
    tracklet.SetFirstRow(r.mFirstRow);
    tracklet.SetLastRow(r.mLastRow);
    tracklet.SetParam(tParam.GetParam());
    int w = tracker.CalculateHitWeight(r.mNHits, tParam.GetChi2(), r.mItr);
    tracklet.SetHitWeight(w);
    for (int iRow = r.mFirstRow; iRow <= r.mLastRow; iRow++) {
      calink ih = CA_GET_ROW_HIT(iRow);
      if (ih != CALINK_INVAL) {
        CA_MAKE_SHARED_REF(GPUTPCRow, row, tracker.Row(iRow), s.mRows[iRow]);
        tracker.MaximizeHitWeight(row, ih, w);
      }
    }
  }
}

MEM_CLASS_PRE2()
GPUd() void GPUTPCTrackletConstructor::UpdateTracklet(int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) & s, GPUTPCThreadMemory& r, GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, MEM_LG2(GPUTPCTrackParam) & tParam, int iRow)
{
// reconstruction of tracklets, tracklets update step
#ifndef GPUCA_EXTERN_ROW_HITS
  GPUTPCTracklet& tracklet = tracker.Tracklets()[r.mItr];
#endif // GPUCA_EXTERN_ROW_HITS

  CA_MAKE_SHARED_REF(GPUTPCRow, row, tracker.Row(iRow), s.mRows[iRow]);

  float y0 = row.Grid().YMin();
  float stepY = row.HstepY();
  float z0 = row.Grid().ZMin() - tParam.ZOffset();
  float stepZ = row.HstepZ();

  if (r.mStage == 0) { // fitting part
    do {
      if (iRow < r.mStartRow || r.mCurrIH == CALINK_INVAL) {
        break;
      }
      if ((iRow - r.mStartRow) & 1) {
        CA_SET_ROW_HIT(iRow, CALINK_INVAL);
        break; // SG!!! - jump over the row
      }

      cahit2 hh = CA_TEXTURE_FETCH(cahit22, gAliTexRefu2, tracker.HitData(row), r.mCurrIH);

      int oldIH = r.mCurrIH;
      r.mCurrIH = CA_TEXTURE_FETCH(calink, gAliTexRefs, tracker.HitLinkUpData(row), r.mCurrIH);

      float x = row.X();
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;

      if (iRow == r.mStartRow) {
        tParam.SetX(x);
        tParam.SetY(y);
        r.mLastY = y;
        if (tracker.Param().ContinuousTracking) {
          tParam.SetZ(0.f);
          r.mLastZ = 0.f;
          tParam.SetZOffset(z);
        } else {
          tParam.SetZ(z);
          r.mLastZ = z;
          tParam.SetZOffset(0.f);
        }
        CADEBUG(
          printf("Tracklet %5d: FIT INIT  ROW %3d X %8.3f -", r.mItr, iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      } else {
        float err2Y, err2Z;
        float dx = x - tParam.X();
        float dy, dz;
        if (r.mNHits >= 10) {
          dy = y - tParam.Y();
          dz = z - tParam.Z();
        } else {
          dy = y - r.mLastY;
          dz = z - r.mLastZ;
        }
        r.mLastY = y;
        r.mLastZ = z;

        float ri = 1.f / CAMath::Sqrt(dx * dx + dy * dy);
        if (iRow == r.mStartRow + 2) { // SG!!! important - thanks to Matthias
          tParam.SetSinPhi(dy * ri);
          tParam.SetSignCosPhi(dx);
          tParam.SetDzDs(dz * ri);
          // std::cout << "Init. errors... " << r.mItr << std::endl;
          tracker.GetErrors2(iRow, tParam, err2Y, err2Z);
          // std::cout << "Init. errors = " << err2Y << " " << err2Z << std::endl;
          tParam.SetCov(0, err2Y);
          tParam.SetCov(2, err2Z);
        }
        float sinPhi, cosPhi;
        if (r.mNHits >= 10 && CAMath::Abs(tParam.SinPhi()) < GPUCA_MAX_SIN_PHI_LOW) {
          sinPhi = tParam.SinPhi();
          cosPhi = CAMath::Sqrt(1 - sinPhi * sinPhi);
        } else {
          sinPhi = dy * ri;
          cosPhi = dx * ri;
        }
        CADEBUG(
          printf("%14s: FIT TRACK ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
        if (!tParam.TransportToX(x, sinPhi, cosPhi, tracker.Param().ConstBz, CALINK_INVAL)) {
          CA_SET_ROW_HIT(iRow, CALINK_INVAL);
          break;
        }
        CADEBUG(
          printf("%15s hits %3d: FIT PROP  ROW %3d X %8.3f -", "", r.mNHits, iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
        tracker.GetErrors2(iRow, tParam.GetZ(), sinPhi, tParam.GetDzDs(), err2Y, err2Z);

        if (r.mNHits >= 10) {
          const float kFactor = tracker.Param().rec.HitPickUpFactor * tracker.Param().rec.HitPickUpFactor * 3.5f * 3.5f;
          float sy2 = kFactor * (tParam.GetErr2Y() + err2Y);
          float sz2 = kFactor * (tParam.GetErr2Z() + err2Z);
          if (sy2 > 2.f) {
            sy2 = 2.f;
          }
          if (sz2 > 2.f) {
            sz2 = 2.f;
          }
          dy = y - tParam.Y();
          dz = z - tParam.Z();
          if (dy * dy > sy2 || dz * dz > sz2) {
            if (++r.mNMissed >= GPUCA_TRACKLET_CONSTRUCTOR_MAX_ROW_GAP_SEED) {
              r.mCurrIH = CALINK_INVAL;
            }
            CA_SET_ROW_HIT(iRow, CALINK_INVAL);
            break;
          }
        }

        if (!tParam.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI_LOW)) {
          CA_SET_ROW_HIT(iRow, CALINK_INVAL);
          break;
        }
        CADEBUG(
          printf("%14s: FIT FILT  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      }
      CA_SET_ROW_HIT(iRow, oldIH);
      r.mNHitsEndRow = ++r.mNHits;
      r.mLastRow = iRow;
      r.mEndRow = iRow;
      r.mNMissed = 0;
      break;
    } while (0);

    /*QQQQprintf("Extrapolate Row %d X %f Y %f Z %f SinPhi %f DzDs %f QPt %f", iRow, tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt());
            for (int i = 0;i < 15;i++) printf(" C%d=%6.2f", i, tParam.GetCov(i));
            printf("\n");*/

    if (r.mCurrIH == CALINK_INVAL) {
      r.mStage = 1;
      r.mLastY = tParam.Y(); // Store last spatial position here to start inward following from here
      r.mLastZ = tParam.Z();
      if (CAMath::Abs(tParam.SinPhi()) > GPUCA_MAX_SIN_PHI) {
        r.mGo = 0;
      }
    }
  } else { // forward/backward searching part
    do {
      if (r.mStage == 2 && iRow > r.mEndRow) {
        break;
      }
      if (r.mNMissed > GPUCA_TRACKLET_CONSTRUCTOR_MAX_ROW_GAP) {
        r.mGo = 0;
        break;
      }

      r.mNMissed++;

      float x = row.X();
      float err2Y, err2Z;
      CADEBUG(
        printf("%14s: SEA TRACK ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      if (!tParam.TransportToX(x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().ConstBz, GPUCA_MAX_SIN_PHI_LOW)) {
        r.mGo = 0;
        CA_SET_ROW_HIT(iRow, CALINK_INVAL);
        break;
      }
      CADEBUG(
        printf("%14s: SEA PROP  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      if (row.NHits() < 1) {
        CA_SET_ROW_HIT(iRow, CALINK_INVAL);
        break;
      }

#ifndef GPUCA_TEXTURE_FETCH_CONSTRUCTOR
      GPUglobalref() const cahit2* hits = tracker.HitData(row);
      GPUglobalref() const calink* firsthit = tracker.FirstHitInBin(row);
#endif //! GPUCA_TEXTURE_FETCH_CONSTRUCTOR
      float fY = tParam.GetY();
      float fZ = tParam.GetZ();
      calink best = CALINK_INVAL;

      { // search for the closest hit
        tracker.GetErrors2(iRow, *((MEM_LG2(GPUTPCTrackParam)*)&tParam), err2Y, err2Z);
        const float kFactor = tracker.Param().rec.HitPickUpFactor * tracker.Param().rec.HitPickUpFactor * 3.5f * 3.5f;
        float sy2 = kFactor * (tParam.GetErr2Y() + err2Y);
        float sz2 = kFactor * (tParam.GetErr2Z() + err2Z);
        if (sy2 > 2.f) {
          sy2 = 2.f;
        }
        if (sz2 > 2.f) {
          sz2 = 2.f;
        }

        int bin, ny, nz;
        row.Grid().GetBinArea(fY, fZ + tParam.ZOffset(), 1.5f, 1.5f, bin, ny, nz);
        float ds = 1e6f;

        for (int k = 0; k <= nz; k++) {
          int nBinsY = row.Grid().Ny();
          int mybin = bin + k * nBinsY;
          unsigned int hitFst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin);
          unsigned int hitLst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin + ny + 1);
          for (unsigned int ih = hitFst; ih < hitLst; ih++) {
            cahit2 hh = CA_TEXTURE_FETCH(cahit2, gAliTexRefu2, hits, ih);
            float y = y0 + hh.x * stepY;
            float z = z0 + hh.y * stepZ;
            float dy = y - fY;
            float dz = z - fZ;
            if (dy * dy < sy2 && dz * dz < sz2) {
              float dds = GPUCA_Y_FACTOR * CAMath::Abs(dy) + CAMath::Abs(dz);
              if (dds < ds) {
                ds = dds;
                best = ih;
              }
            }
          }
        }
      } // end of search for the closest hit

      if (best == CALINK_INVAL) {
        CA_SET_ROW_HIT(iRow, CALINK_INVAL);
        break;
      }

      cahit2 hh = CA_TEXTURE_FETCH(cahit2, gAliTexRefu2, hits, best);
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;

      CADEBUG(GPUInfo("%14s: SEA Hit %5d, Res %f %f", "", best, tParam.Y() - y, tParam.Z() - z));

      calink oldHit = (r.mStage == 2 && iRow >= r.mStartRow) ? CA_GET_ROW_HIT(iRow) : CALINK_INVAL;
      if (oldHit != best && !tParam.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI_LOW, oldHit != CALINK_INVAL)) {
        CA_SET_ROW_HIT(iRow, CALINK_INVAL);
        break;
      }
      CA_SET_ROW_HIT(iRow, best);
      r.mNHits++;
      r.mNMissed = 0;
      CADEBUG(
        printf("%5s hits %3d: SEA FILT  ROW %3d X %8.3f -", "", r.mNHits, iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      if (r.mStage == 1) {
        r.mLastRow = iRow;
      } else {
        r.mFirstRow = iRow;
      }
    } while (0);
  }
}

GPUd() void GPUTPCTrackletConstructor::DoTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() GPUTPCTrackletConstructor::MEM_LOCAL(GPUTPCSharedMemory) & s, GPUTPCThreadMemory& r)
{
  int iRow = 0, iRowEnd = GPUCA_ROW_COUNT;
  MEM_PLAIN(GPUTPCTrackParam)
  tParam;
#ifndef GPUCA_EXTERN_ROW_HITS
  GPUTPCTracklet& tracklet = tracker.Tracklets()[r.mItr];
#endif // GPUCA_EXTERN_ROW_HITS
  if (r.mGo) {
    GPUTPCHitId id = tracker.TrackletStartHits()[r.mItr];

    r.mStartRow = r.mEndRow = r.mFirstRow = r.mLastRow = id.RowIndex();
    r.mCurrIH = id.HitIndex();
    r.mNMissed = 0;
    iRow = r.mStartRow;
    GPUTPCTrackletConstructor::InitTracklet(tParam);
  }
  r.mStage = 0;
  r.mNHits = 0;
  // if (tracker.Param().ISlice() != 35 && tracker.Param().ISlice() != 34 || r.mItr == CALINK_INVAL) {StoreTracklet( 0, 0, 0, 0, s, r, tracker, tParam );return;}

  for (int k = 0; k < 2; k++) {
    for (; iRow != iRowEnd; iRow += r.mStage == 2 ? -1 : 1) {
      if (!r.mGo) {
        break;
      }
      UpdateTracklet(0, 0, 0, 0, s, r, tracker, tParam, iRow);
    }
    if (!r.mGo && r.mStage == 2) {
      for (; iRow >= r.mStartRow; iRow--) {
        CA_SET_ROW_HIT(iRow, CALINK_INVAL);
      }
    }
    if (r.mStage == 2) {
      StoreTracklet(0, 0, 0, 0, s, r, tracker, tParam);
    } else {
      r.mNMissed = 0;
      if ((r.mGo = (tParam.TransportToX(tracker.Row(r.mEndRow).X(), tracker.Param().ConstBz, GPUCA_MAX_SIN_PHI) && tParam.Filter(r.mLastY, r.mLastZ, tParam.Err2Y() * 0.5f, tParam.Err2Z() * 0.5f, GPUCA_MAX_SIN_PHI_LOW, true)))) {
        CADEBUG(
          printf("%14s: SEA BACK  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
        float err2Y, err2Z;
        tracker.GetErrors2(r.mEndRow, tParam, err2Y, err2Z);
        if (tParam.GetCov(0) < err2Y) {
          tParam.SetCov(0, err2Y);
        }
        if (tParam.GetCov(2) < err2Z) {
          tParam.SetCov(2, err2Z);
        }
        CADEBUG(
          printf("%14s: SEA ADJUS ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      }
      r.mNHits -= r.mNHitsEndRow;
      r.mStage = 2;
      iRow = r.mEndRow;
      iRowEnd = -1;
    }
  }
}

template <>
GPUd() void GPUTPCTrackletConstructor::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) & sMem, processorType& tracker)
{
  if (get_local_id(0) == 0) {
    sMem.mNTracklets = *tracker.NTracklets();
  }

#ifdef GPUCA_GPUCODE
  for (unsigned int i = get_local_id(0); i < GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(GPUTPCRow)) / sizeof(int); i += get_local_size(0)) {
    reinterpret_cast<GPUsharedref() int*>(&sMem.mRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
  }

#endif
  GPUbarrier();

  GPUTPCThreadMemory rMem;
  for (rMem.mItr = get_global_id(0); rMem.mItr < sMem.mNTracklets; rMem.mItr += get_global_size(0)) {
    rMem.mGo = 1;
    DoTracklet(tracker, sMem, rMem);
  }
}

template <>
GPUd() void GPUTPCTrackletConstructor::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) & sMem, processorType& tracker0)
{
#ifdef GPUCA_GPUCODE
  GPUconstantref() MEM_GLOBAL(GPUTPCTracker)* pTracker = &tracker0;

  int mySlice = get_group_id(0) % GPUCA_NSLICES;
  int currentSlice = -1;

  if (get_local_id(0) == 0) {
    sMem.mNextTrackletFirstRun = 1;
  }

  for (unsigned int iSlice = 0; iSlice < GPUCA_NSLICES; iSlice++) {
    GPUconstantref() MEM_GLOBAL(GPUTPCTracker)& tracker = pTracker[mySlice];

    GPUTPCThreadMemory rMem;

    while ((rMem.mItr = FetchTracklet(tracker, sMem)) != -2) {
      if (rMem.mItr >= 0 && get_local_id(0) < GPUCA_THREAD_COUNT_CONSTRUCTOR) {
        rMem.mItr += get_local_id(0);
      } else {
        rMem.mItr = -1;
      }

      if (mySlice != currentSlice) {
        if (get_local_id(0) == 0) {
          sMem.mNTracklets = *tracker.NTracklets();
        }

        for (unsigned int i = get_local_id(0); i < GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(GPUTPCRow)) / sizeof(int); i += get_local_size(0)) {
          reinterpret_cast<GPUsharedref() int*>(&sMem.mRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
        }
        GPUbarrier();
        currentSlice = mySlice;
      }

      if (rMem.mItr >= 0 && rMem.mItr < sMem.mNTracklets) {
        rMem.mGo = true;
        DoTracklet(tracker, sMem, rMem);
      }
    }
    if (++mySlice >= GPUCA_NSLICES) {
      mySlice = 0;
    }
  }
#else
  throw std::logic_error("Not supported on CPU");
#endif
}

#ifdef GPUCA_GPUCODE

GPUdi() int GPUTPCTrackletConstructor::FetchTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) & sMem)
{
  const int nativeslice = get_group_id(0) % GPUCA_NSLICES;
  const unsigned int nTracklets = *tracker.NTracklets();
  GPUbarrier();
  if (get_local_id(0) == 0) {
    if (sMem.mNextTrackletFirstRun == 1) {
      sMem.mNextTrackletFirst = (get_group_id(0) - nativeslice) / GPUCA_NSLICES * GPUCA_THREAD_COUNT_CONSTRUCTOR;
      sMem.mNextTrackletFirstRun = 0;
    } else {
      if (tracker.GPUParameters()->nextTracklet < nTracklets) {
        const unsigned int firstTracklet = CAMath::AtomicAdd(&tracker.GPUParameters()->nextTracklet, GPUCA_THREAD_COUNT_CONSTRUCTOR);
        if (firstTracklet < nTracklets) {
          sMem.mNextTrackletFirst = firstTracklet;
        } else {
          sMem.mNextTrackletFirst = -2;
        }
      } else {
        sMem.mNextTrackletFirst = -2;
      }
    }
  }
  GPUbarrier();
  return (sMem.mNextTrackletFirst);
}

#else // GPUCA_GPUCODE

int GPUTPCTrackletConstructor::GPUTPCTrackletConstructorGlobalTracking(GPUTPCTracker& tracker, GPUTPCTrackParam& tParam, int row, int increment, int iTracklet)
{
  GPUTPCThreadMemory rMem;
  GPUshared() GPUTPCSharedMemory sMem;
  sMem.mNTracklets = *tracker.NTracklets();
  rMem.mItr = iTracklet;
  rMem.mStage = 3;
  rMem.mNHits = rMem.mNMissed = 0;
  rMem.mGo = 1;
  while (rMem.mGo && row >= 0 && row < GPUCA_ROW_COUNT) {
    UpdateTracklet(1, 1, 0, 0, sMem, rMem, tracker, tParam, row);
    row += increment;
  }
  if (!CheckCov(tParam)) {
    rMem.mNHits = 0;
  }
  return (rMem.mNHits);
}

#endif // GPUCA_GPUCODE
