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
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "GPUTPCGlobalTracking.h"
#endif
#include "GPUParam.inc"
#include "GPUCommonMath.h"

using namespace GPUCA_NAMESPACE::gpu;

MEM_CLASS_PRE2()
GPUdii() void GPUTPCTrackletConstructor::InitTracklet(MEM_LG2(GPUTPCTrackParam) & GPUrestrict() tParam)
{
  // Initialize Tracklet Parameters using default values
  tParam.InitParam();
}

MEM_CLASS_PRE2()
GPUd() bool GPUTPCTrackletConstructor::CheckCov(MEM_LG2(GPUTPCTrackParam) & GPUrestrict() tParam)
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
GPUd() void GPUTPCTrackletConstructor::StoreTracklet(int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & s, GPUTPCThreadMemory& GPUrestrict() r, GPUconstantref() MEM_LG2(GPUTPCTracker) & GPUrestrict() tracker, MEM_LG3(GPUTPCTrackParam) & GPUrestrict() tParam, calink* rowHits)
{
  // reconstruction of tracklets, tracklet store step
  if (r.mNHits == 0 || (r.mNHits < GPUCA_TRACKLET_SELECTOR_MIN_HITS_B5(tParam.QPt() * tracker.Param().qptB5Scaler) || !CheckCov(tParam) || CAMath::Abs(tParam.GetQPt() * tracker.Param().qptB5Scaler) > tracker.Param().rec.maxTrackQPtB5)) {
    return;
  }

  /*GPUInfo("Tracklet %d: Hits %3d NDF %3d Chi %8.4f Sign %f Cov: %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f", r.mISH, r.mNHits, tParam.GetNDF(), tParam.GetChi2(), tParam.GetSignCosPhi(),
          tParam.Cov()[0], tParam.Cov()[1], tParam.Cov()[2], tParam.Cov()[3], tParam.Cov()[4], tParam.Cov()[5], tParam.Cov()[6], tParam.Cov()[7], tParam.Cov()[8], tParam.Cov()[9],
          tParam.Cov()[10], tParam.Cov()[11], tParam.Cov()[12], tParam.Cov()[13], tParam.Cov()[14]);*/

  const unsigned int nHits = r.mLastRow + 1 - r.mFirstRow;
  unsigned int hitout = CAMath::AtomicAdd(tracker.NRowHits(), nHits);
  if (hitout + nHits > tracker.NMaxRowHits()) {
    tracker.raiseError(GPUErrors::ERROR_TRACKLET_HIT_OVERFLOW, tracker.ISlice(), hitout + nHits, tracker.NMaxRowHits());
    CAMath::AtomicExch(tracker.NRowHits(), tracker.NMaxRowHits());
    return;
  }
  unsigned int itrout = CAMath::AtomicAdd(tracker.NTracklets(), 1u);
  if (itrout >= tracker.NMaxTracklets()) {
    tracker.raiseError(GPUErrors::ERROR_TRACKLET_OVERFLOW, tracker.ISlice(), itrout, tracker.NMaxTracklets());
    CAMath::AtomicExch(tracker.NTracklets(), tracker.NMaxTracklets());
    return;
  }

  GPUglobalref() MEM_GLOBAL(GPUTPCTracklet) & GPUrestrict() tracklet = tracker.Tracklets()[itrout];

  tracklet.SetNHits(r.mNHits);
  CADEBUG(printf("    DONE %d hits\n", r.mNHits));

  tracklet.SetFirstRow(r.mFirstRow);
  tracklet.SetLastRow(r.mLastRow);
  tracklet.SetFirstHit(hitout);
  tracklet.SetParam(tParam.GetParam());
  int w = tracker.CalculateHitWeight(r.mNHits, tParam.GetChi2(), r.mISH);
  tracklet.SetHitWeight(w);
#ifdef __HIPCC__ // Todo: fixme!
  for (int iRow = r.mFirstRow - 1; ++iRow <= r.mLastRow; /*iRow++*/) {
#else
  for (int iRow = r.mFirstRow; iRow <= r.mLastRow; iRow++) {
#endif
    calink ih = rowHits[iRow];
    tracker.TrackletRowHits()[hitout + (iRow - r.mFirstRow)] = ih;
    if (ih != CALINK_INVAL) {
      CA_MAKE_SHARED_REF(GPUTPCRow, row, tracker.Row(iRow), s.mRows[iRow]);
      tracker.MaximizeHitWeight(row, ih, w);
    }
  }
}

MEM_CLASS_PRE2_TEMPLATE(class T)
GPUdic(2, 1) void GPUTPCTrackletConstructor::UpdateTracklet(int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/, GPUsharedref() T& s, GPUTPCThreadMemory& GPUrestrict() r, GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & GPUrestrict() tracker, MEM_LG2(GPUTPCTrackParam) & GPUrestrict() tParam, int iRow, calink& rowHit)
{
  // reconstruction of tracklets, tracklets update step
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
        rowHit = CALINK_INVAL;
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
        if (tracker.Param().par.continuousTracking) {
          float refZ = ((z > 0) ? tracker.Param().rec.tpc.defaultZOffsetOverR : -tracker.Param().rec.tpc.defaultZOffsetOverR) * x;
          tParam.SetZ(refZ);
          r.mLastZ = refZ;
          tParam.SetZOffset(z - refZ);
        } else {
          tParam.SetZ(z);
          r.mLastZ = z;
          tParam.SetZOffset(0.f);
        }
        CADEBUG(
          printf("Tracklet %5d: FIT INIT  ROW %3d X %8.3f -", r.mISH, iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
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
          // std::cout << "Init. errors... " << r.mISH << std::endl;
          tracker.GetErrors2Seeding(iRow, tParam, err2Y, err2Z);
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
        if (!tParam.TransportToX(x, sinPhi, cosPhi, tracker.Param().constBz, GPUCA_MAX_SIN_PHI)) {
          rowHit = CALINK_INVAL;
          break;
        }
        CADEBUG(
          printf("%5s hits %3d: FIT PROP  ROW %3d X %8.3f -", "", r.mNHits, iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
        tracker.GetErrors2Seeding(iRow, tParam.GetZ(), sinPhi, tParam.GetDzDs(), err2Y, err2Z);

        if (r.mNHits >= 10) {
          const float kFactor = tracker.Param().rec.tpc.hitPickUpFactor * tracker.Param().rec.tpc.hitPickUpFactor * 3.5f * 3.5f;
          float sy2 = kFactor * (tParam.Err2Y() + err2Y);
          float sz2 = kFactor * (tParam.Err2Z() + err2Z);
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
            rowHit = CALINK_INVAL;
            break;
          }
        }

        if (!tParam.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI_LOW)) {
          rowHit = CALINK_INVAL;
          break;
        }
        CADEBUG(
          printf("%14s: FIT FILT  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      }
      rowHit = oldIH;
      r.mNHitsEndRow = ++r.mNHits;
      r.mLastRow = iRow;
      r.mEndRow = iRow;
      r.mNMissed = 0;
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
      if (!tParam.TransportToX(x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().constBz, GPUCA_MAX_SIN_PHI_LOW)) {
        r.mGo = 0;
        rowHit = CALINK_INVAL;
        break;
      }
      CADEBUG(
        printf("%14s: SEA PROP  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
      if (row.NHits() < 1) {
        rowHit = CALINK_INVAL;
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
        tracker.GetErrors2Seeding(iRow, *((MEM_LG2(GPUTPCTrackParam)*)&tParam), err2Y, err2Z);
        const float kFactor = tracker.Param().rec.tpc.hitPickUpFactor * tracker.Param().rec.tpc.hitPickUpFactor * 3.5f * 3.5f;
        float sy2 = kFactor * (tParam.Err2Y() + err2Y);
        float sz2 = kFactor * (tParam.Err2Z() + err2Z);
        if (sy2 > 2.f) {
          sy2 = 2.f;
        }
        if (sz2 > 2.f) {
          sz2 = 2.f;
        }

        int bin, ny, nz;
        row.Grid().GetBinArea(fY, fZ + tParam.ZOffset(), 1.5f, 1.5f, bin, ny, nz);
        float ds = 1e6f;

#ifdef __HIPCC__ // Todo: fixme!
        for (int k = -1; ++k <= nz; /*k++*/) {
#else
        for (int k = 0; k <= nz; k++) {
#endif
          int nBinsY = row.Grid().Ny();
          int mybin = bin + k * nBinsY;
          unsigned int hitFst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin);
          unsigned int hitLst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin + ny + 1);
#ifdef __HIPCC__ // Todo: fixme!
          for (unsigned int ih = hitFst - 1; ++ih < hitLst; /*ih++*/) {
#else
          for (unsigned int ih = hitFst; ih < hitLst; ih++) {
#endif
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
        rowHit = CALINK_INVAL;
        break;
      }

      cahit2 hh = CA_TEXTURE_FETCH(cahit2, gAliTexRefu2, hits, best);
      float y = y0 + hh.x * stepY;
      float z = z0 + hh.y * stepZ;

      CADEBUG(printf("%14s: SEA Hit %5d (%8.3f %8.3f), Res %f %f\n", "", best, y, z, tParam.Y() - y, tParam.Z() - z));

      calink oldHit = (r.mStage == 2 && iRow >= r.mStartRow) ? rowHit : CALINK_INVAL;
      if (oldHit != best && !tParam.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI_LOW, oldHit != CALINK_INVAL)) {
        rowHit = CALINK_INVAL;
        break;
      }
      rowHit = best;
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

GPUdic(2, 1) void GPUTPCTrackletConstructor::DoTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & GPUrestrict() tracker, GPUsharedref() GPUTPCTrackletConstructor::MEM_LOCAL(GPUSharedMemory) & s, GPUTPCThreadMemory& GPUrestrict() r)
{
  int iRow = 0, iRowEnd = GPUCA_ROW_COUNT;
  MEM_PLAIN(GPUTPCTrackParam)
  tParam;
  calink rowHits[GPUCA_ROW_COUNT];
  if (r.mGo) {
    GPUTPCHitId id = tracker.TrackletStartHits()[r.mISH];

    r.mStartRow = r.mEndRow = r.mFirstRow = r.mLastRow = id.RowIndex();
    r.mCurrIH = id.HitIndex();
    r.mNMissed = 0;
    iRow = r.mStartRow;
    GPUTPCTrackletConstructor::InitTracklet(tParam);
  }
  r.mStage = 0;
  r.mNHits = 0;

#ifdef __HIPCC__ // Todo: fixme!
  for (int k = -1; ++k < 2; /*k++*/) {
#else
  for (int k = 0; k < 2; k++) {
#endif
    for (; iRow != iRowEnd; iRow += r.mStage == 2 ? -1 : 1) {
      if (!r.mGo) {
        break;
      }
      UpdateTracklet(0, 0, 0, 0, s, r, tracker, tParam, iRow, rowHits[iRow]);
    }
    if (!r.mGo && r.mStage == 2) {
      for (; iRow >= r.mStartRow; iRow--) {
        rowHits[iRow] = CALINK_INVAL;
      }
    }
    if (r.mStage == 2) {
      StoreTracklet(0, 0, 0, 0, s, r, tracker, tParam, rowHits);
    } else {
      r.mNMissed = 0;
      if ((r.mGo = (tParam.TransportToX(tracker.Row(r.mEndRow).X(), tracker.Param().constBz, GPUCA_MAX_SIN_PHI) && tParam.Filter(r.mLastY, r.mLastZ, tParam.Err2Y() * 0.5f, tParam.Err2Z() * 0.5f, GPUCA_MAX_SIN_PHI_LOW, true)))) {
        CADEBUG(
          printf("%14s: SEA BACK  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) { printf(" %8.3f", tParam.Par()[i]); } printf(" -"); for (int i = 0; i < 15; i++) { printf(" %8.3f", tParam.Cov()[i]); } printf("\n"));
        float err2Y, err2Z;
        tracker.GetErrors2Seeding(r.mEndRow, tParam, err2Y, err2Z);
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
GPUdii() void GPUTPCTrackletConstructor::Thread<GPUTPCTrackletConstructor::singleSlice>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & sMem, processorType& GPUrestrict() tracker)
{
  if (get_local_id(0) == 0) {
    sMem.mNStartHits = *tracker.NStartHits();
  }
  CA_SHARED_CACHE(&sMem.mRows[0], tracker.SliceDataRows(), GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(GPUTPCRow)));
  GPUbarrier();

  GPUTPCThreadMemory rMem;
  for (rMem.mISH = get_global_id(0); rMem.mISH < sMem.mNStartHits; rMem.mISH += get_global_size(0)) {
    rMem.mGo = 1;
    DoTracklet(tracker, sMem, rMem);
  }
}

template <>
GPUdii() void GPUTPCTrackletConstructor::Thread<GPUTPCTrackletConstructor::allSlices>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & sMem, processorType& GPUrestrict() tracker0)
{
  GPUconstantref() MEM_GLOBAL(GPUTPCTracker) * GPUrestrict() pTracker = &tracker0;
#ifdef GPUCA_GPUCODE
  int mySlice = get_group_id(0) % GPUCA_NSLICES;
  int currentSlice = -1;

  if (get_local_id(0) == 0) {
    sMem.mNextStartHitFirstRun = 1;
  }
  GPUCA_UNROLL(, U())
  for (unsigned int iSlice = 0; iSlice < GPUCA_NSLICES; iSlice++) {
    GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & GPUrestrict() tracker = pTracker[mySlice];

    GPUTPCThreadMemory rMem;

    while ((rMem.mISH = FetchTracklet(tracker, sMem)) != -2) {
      if (rMem.mISH >= 0 && get_local_id(0) < GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCTrackletConstructor)) {
        rMem.mISH += get_local_id(0);
      } else {
        rMem.mISH = -1;
      }

      if (mySlice != currentSlice) {
        if (get_local_id(0) == 0) {
          sMem.mNStartHits = *tracker.NStartHits();
        }
        CA_SHARED_CACHE(&sMem.mRows[0], tracker.SliceDataRows(), GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(GPUTPCRow)));
        GPUbarrier();
        currentSlice = mySlice;
      }

      if (rMem.mISH >= 0 && rMem.mISH < sMem.mNStartHits) {
        rMem.mGo = true;
        DoTracklet(tracker, sMem, rMem);
      }
    }
    if (++mySlice >= GPUCA_NSLICES) {
      mySlice = 0;
    }
  }
#else
  for (int iSlice = 0; iSlice < GPUCA_NSLICES; iSlice++) {
    Thread<singleSlice>(nBlocks, nThreads, iBlock, iThread, sMem, pTracker[iSlice]);
  }
#endif
}

#ifdef GPUCA_GPUCODE

GPUd() int GPUTPCTrackletConstructor::FetchTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & GPUrestrict() tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & sMem)
{
  const unsigned int nStartHit = *tracker.NStartHits();
  GPUbarrier();
  if (get_local_id(0) == 0) {
    int firstStartHit = -2;
    if (sMem.mNextStartHitFirstRun == 1) {
      firstStartHit = (get_group_id(0) - tracker.ISlice()) / GPUCA_NSLICES * GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCTrackletConstructor);
      sMem.mNextStartHitFirstRun = 0;
    } else {
      if (tracker.GPUParameters()->nextStartHit < nStartHit) {
        firstStartHit = CAMath::AtomicAdd<unsigned int>(&tracker.GPUParameters()->nextStartHit, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCTrackletConstructor));
      }
    }
    sMem.mNextStartHitFirst = firstStartHit < (int)nStartHit ? firstStartHit : -2;
  }
  GPUbarrier();
  return (sMem.mNextStartHitFirst);
}

#endif // GPUCA_GPUCODE

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
template <>
GPUd() int GPUTPCTrackletConstructor::GPUTPCTrackletConstructorGlobalTracking<GPUTPCGlobalTracking::GPUSharedMemory>(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & GPUrestrict() tracker, GPUsharedref() GPUTPCGlobalTracking::GPUSharedMemory& sMem, MEM_LG(GPUTPCTrackParam) & GPUrestrict() tParam, int row, int increment, int iTracklet, calink* rowHits)
{
  GPUTPCThreadMemory rMem;
  rMem.mISH = iTracklet;
  rMem.mStage = 3;
  rMem.mNHits = rMem.mNMissed = 0;
  rMem.mGo = 1;
  while (rMem.mGo && row >= 0 && row < GPUCA_ROW_COUNT) {
    UpdateTracklet(1, 1, 0, 0, sMem, rMem, tracker, tParam, row, rowHits[row]);
    row += increment;
  }
  if (!CheckCov(tParam)) {
    rMem.mNHits = 0;
  }
  return (rMem.mNHits);
}
#endif
