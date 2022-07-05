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

/// \file GPUTPCGlobalTracking.cxx
/// \author David Rohr

#include "GPUTPCDef.h"
#include "GPUTPCGlobalTracking.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUTPCTrackLinearisation.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"
#include "GPUParam.inc"

using namespace GPUCA_NAMESPACE::gpu;

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)

GPUd() int GPUTPCGlobalTracking::PerformGlobalTrackingRun(GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, const GPUTPCTracker& GPUrestrict() sliceSource, int iTrack, int rowIndex, float angle, int direction)
{
  /*for (int j = 0;j < Tracks()[j].NHits();j++)
  {
    GPUInfo("Hit %3d: Row %3d: X %3.7lf Y %3.7lf", j, mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex(), Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()).X(),
    (float) Data().HitDataY(Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()), mTrackHits[Tracks()[iTrack].FirstHitID() + j].HitIndex()) * Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()).HstepY() + Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()).Grid().YMin());
  }*/

  GPUTPCTrackParam tParam;
  tParam.InitParam();
  tParam.SetCov(0, 0.05);
  tParam.SetCov(2, 0.05);
  tParam.SetCov(5, 0.001);
  tParam.SetCov(9, 0.001);
  tParam.SetCov(14, 0.05);
  tParam.SetParam(sliceSource.Tracks()[iTrack].Param());

  // GPUInfo("Parameters X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());
  if (!tParam.Rotate(angle, GPUCA_MAX_SIN_PHI)) {
    return 0;
  }
  // GPUInfo("Rotated X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());

  int maxRowGap = 10;
  GPUTPCTrackLinearisation t0(tParam);
  do {
    rowIndex += direction;
    if (!tParam.TransportToX(tracker.Row(rowIndex).X(), t0, tracker.Param().constBz, GPUCA_MAX_SIN_PHI)) {
      return 0; // Reuse t0 linearization until we are in the next sector
    }
    // GPUInfo("Transported X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f (MaxY %f)", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi(), Row(rowIndex).MaxY());
    if (--maxRowGap == 0) {
      return 0;
    }
  } while (CAMath::Abs(tParam.Y()) > tracker.Row(rowIndex).MaxY());

  float err2Y, err2Z;
  tracker.GetErrors2Seeding(rowIndex, tParam.Z(), tParam.SinPhi(), tParam.DzDs(), err2Y, err2Z);
  if (tParam.GetCov(0) < err2Y) {
    tParam.SetCov(0, err2Y);
  }
  if (tParam.GetCov(2) < err2Z) {
    tParam.SetCov(2, err2Z);
  }

  calink rowHits[GPUCA_ROW_COUNT];
  int nHits = GPUTPCTrackletConstructor::GPUTPCTrackletConstructorGlobalTracking(tracker, smem, tParam, rowIndex, direction, 0, rowHits);
  if (nHits >= GPUCA_GLOBAL_TRACKING_MIN_HITS) {
    // GPUInfo("%d hits found", nHits);
    unsigned int hitId = CAMath::AtomicAdd(&tracker.CommonMemory()->nTrackHits, (unsigned int)nHits);
    if (hitId + nHits > tracker.NMaxTrackHits()) {
      tracker.raiseError(GPUErrors::ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW, tracker.ISlice(), hitId + nHits, tracker.NMaxTrackHits());
      CAMath::AtomicExch(&tracker.CommonMemory()->nTrackHits, tracker.NMaxTrackHits());
      return 0;
    }
    unsigned int trackId = CAMath::AtomicAdd(&tracker.CommonMemory()->nTracks, 1u);
    if (trackId >= tracker.NMaxTracks()) { // >= since will increase by 1
      tracker.raiseError(GPUErrors::ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW, tracker.ISlice(), trackId, tracker.NMaxTracks());
      CAMath::AtomicExch(&tracker.CommonMemory()->nTracks, tracker.NMaxTracks());
      return 0;
    }

    if (direction == 1) {
      int i = 0;
      while (i < nHits) {
        const calink rowHit = rowHits[rowIndex];
        if (rowHit != CALINK_INVAL) {
          // GPUInfo("New track: entry %d, row %d, hitindex %d", i, rowIndex, mTrackletRowHits[rowIndex * tracker.CommonMemory()->nTracklets]);
          tracker.TrackHits()[hitId + i].Set(rowIndex, rowHit);
          // if (i == 0) tParam.TransportToX(Row(rowIndex).X(), Param().constBz(), GPUCA_MAX_SIN_PHI); //Use transport with new linearisation, we have changed the track in between - NOT needed, fitting will always start at outer end of global track!
          i++;
        }
        rowIndex++;
      }
    } else {
      int i = nHits - 1;
      while (i >= 0) {
        const calink rowHit = rowHits[rowIndex];
        if (rowHit != CALINK_INVAL) {
          // GPUInfo("New track: entry %d, row %d, hitindex %d", i, rowIndex, mTrackletRowHits[rowIndex * tracker.CommonMemory()->nTracklets]);
          tracker.TrackHits()[hitId + i].Set(rowIndex, rowHit);
          i--;
        }
        rowIndex--;
      }
    }
    GPUTPCTrack& GPUrestrict() track = tracker.Tracks()[trackId];
    track.SetParam(tParam.GetParam());
    track.SetNHits(nHits);
    track.SetFirstHitID(hitId);
    track.SetLocalTrackId((sliceSource.ISlice() << 24) | sliceSource.Tracks()[iTrack].LocalTrackId());
  }

  return (nHits >= GPUCA_GLOBAL_TRACKING_MIN_HITS);
}

GPUd() void GPUTPCGlobalTracking::PerformGlobalTracking(int nBlocks, int nThreads, int iBlock, int iThread, const GPUTPCTracker& tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, GPUTPCTracker& GPUrestrict() sliceTarget, bool right)
{
  for (int i = iBlock * nThreads + iThread; i < tracker.CommonMemory()->nLocalTracks; i += nThreads * nBlocks) {
    {
      const int tmpHit = tracker.Tracks()[i].FirstHitID();
      if (tracker.TrackHits()[tmpHit].RowIndex() >= GPUCA_GLOBAL_TRACKING_MIN_ROWS && tracker.TrackHits()[tmpHit].RowIndex() < GPUCA_GLOBAL_TRACKING_RANGE) {
        int rowIndex = tracker.TrackHits()[tmpHit].RowIndex();
        const GPUTPCRow& GPUrestrict() row = tracker.Row(rowIndex);
        float Y = (float)tracker.Data().HitDataY(row, tracker.TrackHits()[tmpHit].HitIndex()) * row.HstepY() + row.Grid().YMin();
        if (!right && Y < -row.MaxY() * GPUCA_GLOBAL_TRACKING_Y_RANGE_LOWER) {
          // GPUInfo("Track %d, lower row %d, left border (%f of %f)", i, mTrackHits[tmpHit].RowIndex(), Y, -row.MaxY());
          PerformGlobalTrackingRun(sliceTarget, smem, tracker, i, rowIndex, -tracker.Param().par.dAlpha, -1);
        }
        if (right && Y > row.MaxY() * GPUCA_GLOBAL_TRACKING_Y_RANGE_LOWER) {
          // GPUInfo("Track %d, lower row %d, right border (%f of %f)", i, mTrackHits[tmpHit].RowIndex(), Y, row.MaxY());
          PerformGlobalTrackingRun(sliceTarget, smem, tracker, i, rowIndex, tracker.Param().par.dAlpha, -1);
        }
      }
    }

    {
      const int tmpHit = tracker.Tracks()[i].FirstHitID() + tracker.Tracks()[i].NHits() - 1;
      if (tracker.TrackHits()[tmpHit].RowIndex() < GPUCA_ROW_COUNT - GPUCA_GLOBAL_TRACKING_MIN_ROWS && tracker.TrackHits()[tmpHit].RowIndex() >= GPUCA_ROW_COUNT - GPUCA_GLOBAL_TRACKING_RANGE) {
        int rowIndex = tracker.TrackHits()[tmpHit].RowIndex();
        const GPUTPCRow& GPUrestrict() row = tracker.Row(rowIndex);
        float Y = (float)tracker.Data().HitDataY(row, tracker.TrackHits()[tmpHit].HitIndex()) * row.HstepY() + row.Grid().YMin();
        if (!right && Y < -row.MaxY() * GPUCA_GLOBAL_TRACKING_Y_RANGE_UPPER) {
          // GPUInfo("Track %d, upper row %d, left border (%f of %f)", i, mTrackHits[tmpHit].RowIndex(), Y, -row.MaxY());
          PerformGlobalTrackingRun(sliceTarget, smem, tracker, i, rowIndex, -tracker.Param().par.dAlpha, 1);
        }
        if (right && Y > row.MaxY() * GPUCA_GLOBAL_TRACKING_Y_RANGE_UPPER) {
          // GPUInfo("Track %d, upper row %d, right border (%f of %f)", i, mTrackHits[tmpHit].RowIndex(), Y, row.MaxY());
          PerformGlobalTrackingRun(sliceTarget, smem, tracker, i, rowIndex, tracker.Param().par.dAlpha, 1);
        }
      }
    }
  }
}

template <>
GPUdii() void GPUTPCGlobalTracking::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& GPUrestrict() tracker)
{
  CA_SHARED_CACHE(&smem.mRows[0], tracker.SliceDataRows(), GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(GPUTPCRow)));
  GPUbarrier();

  if (tracker.NHitsTotal() == 0) {
    return;
  }
  const int iSlice = tracker.ISlice();
  int sliceLeft = (iSlice + (GPUDataTypes::NSLICES / 2 - 1)) % (GPUDataTypes::NSLICES / 2);
  int sliceRight = (iSlice + 1) % (GPUDataTypes::NSLICES / 2);
  if (iSlice >= (int)GPUDataTypes::NSLICES / 2) {
    sliceLeft += GPUDataTypes::NSLICES / 2;
    sliceRight += GPUDataTypes::NSLICES / 2;
  }
  PerformGlobalTracking(nBlocks, nThreads, iBlock, iThread, tracker.GetConstantMem()->tpcTrackers[sliceLeft], smem, tracker, true);
  PerformGlobalTracking(nBlocks, nThreads, iBlock, iThread, tracker.GetConstantMem()->tpcTrackers[sliceRight], smem, tracker, false);
}

GPUd() int GPUTPCGlobalTracking::GlobalTrackingSliceOrder(int iSlice)
{
  iSlice++;
  if (iSlice == GPUDataTypes::NSLICES / 2) {
    iSlice = 0;
  }
  if (iSlice == GPUDataTypes::NSLICES) {
    iSlice = GPUDataTypes::NSLICES / 2;
  }
  return iSlice;
}

GPUd() void GPUTPCGlobalTracking::GlobalTrackingSliceLeftRight(unsigned int iSlice, unsigned int& left, unsigned int& right)
{
  left = (iSlice + (GPUDataTypes::NSLICES / 2 - 1)) % (GPUDataTypes::NSLICES / 2);
  right = (iSlice + 1) % (GPUDataTypes::NSLICES / 2);
  if (iSlice >= (int)GPUDataTypes::NSLICES / 2) {
    left += GPUDataTypes::NSLICES / 2;
    right += GPUDataTypes::NSLICES / 2;
  }
}
#endif // !__OPENCL__ || __OPENCLCPP__

template <>
GPUdii() void GPUTPCGlobalTrackingCopyNumbers::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& GPUrestrict() tracker, int n)
{
  for (int i = get_global_id(0); i < n; i += get_global_size(0)) {
    GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & GPUrestrict() trk = (&tracker)[i];
    trk.CommonMemory()->nLocalTracks = trk.CommonMemory()->nTracks;
    trk.CommonMemory()->nLocalTrackHits = trk.CommonMemory()->nTrackHits;
  }
}
