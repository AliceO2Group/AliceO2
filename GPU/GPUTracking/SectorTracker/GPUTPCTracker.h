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

/// \file GPUTPCTracker.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKER_H
#define GPUTPCTRACKER_H

#include "GPUTPCDef.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#include "GPUTPCHitId.h"
#include "GPUTPCTrackingData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCTracklet.h"
#include "GPUProcessor.h"

namespace o2::gpu
{
struct GPUTPCClusterData;
struct GPUParam;
class GPUTPCTrack;
class GPUTPCTrackParam;
class GPUTPCRow;

class GPUTPCTracker : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE_DEVICE
  GPUTPCTracker() = default;
  ~GPUTPCTracker();
  GPUTPCTracker(const GPUTPCTracker&) = delete;
  GPUTPCTracker& operator=(const GPUTPCTracker&) = delete;

  void SetSector(int32_t iSector);
  void InitializeProcessor();
  void InitializeRows(const GPUParam* param) { mData.InitializeRows(*param); }

  int32_t CheckEmptySector();

  // Debugging Stuff
  void DumpTrackingData(std::ostream& out);         // Dump Input Sector Data
  void DumpLinks(std::ostream& out, int32_t phase); // Dump all links to file (for comparison after NeighboursFinder/Cleaner)
  void DumpStartHits(std::ostream& out);            // Same for Start Hits
  void DumpHitWeights(std::ostream& out);           //....
  void DumpTrackHits(std::ostream& out);            // Same for Track Hits
  void DumpTrackletHits(std::ostream& out);         // Same for Track Hits
#endif

  struct commonMemoryStruct {
    GPUAtomic(uint32_t) nStartHits = 0; // number of start hits
    GPUAtomic(uint32_t) nTracklets = 0; // number of tracklets
    GPUAtomic(uint32_t) nRowHits = 0;   // number of tracklet hits
    GPUAtomic(uint32_t) nTracks = 0;    // number of reconstructed tracks
    int32_t nLocalTracks = 0;           // number of reconstructed tracks before extrapolation tracking
    GPUAtomic(uint32_t) nTrackHits = 0; // number of track hits
    int32_t nLocalTrackHits = 0;        // see above
  };

  GPUhdi() const GPUTPCRow& Row(const GPUTPCHitId& HitId) const { return mData.Row(HitId.RowIndex()); }
  GPUhdni() GPUglobalref() commonMemoryStruct* CommonMemory() const
  {
    return (mCommonMem);
  }

  GPUdi() static void GetErrors2Seeding(const GPUParam& param, char sector, int32_t iRow, const GPUTPCTrackParam& t, float time, float& ErrY2, float& ErrZ2)
  {
    // param.GetClusterErrors2(sector, iRow, param.GetContinuousTracking() != 0. ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, 0.f, 0.f, ErrY2, ErrZ2);
    param.GetClusterErrorsSeeding2(sector, iRow, param.par.continuousTracking != 0.f ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, ErrY2, ErrZ2);
  }

  GPUdi() void GetErrors2Seeding(int32_t iRow, const GPUTPCTrackParam& t, float time, float& ErrY2, float& ErrZ2) const
  {
    // Param().GetClusterErrors2(mISector, iRow, Param().GetContinuousTracking() != 0. ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, 0.f, 0.f, ErrY2, ErrZ2);
    Param().GetClusterErrorsSeeding2(mISector, iRow, Param().par.continuousTracking != 0.f ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, ErrY2, ErrZ2);
  }
  GPUdi() void GetErrors2Seeding(int32_t iRow, float z, float sinPhi, float DzDs, float time, float& ErrY2, float& ErrZ2) const
  {
    // Param().GetClusterErrors2(mISector, iRow, Param().GetContinuousTracking() != 0. ? 125.f : z, sinPhi, DzDs, time, 0.f, 0.f, ErrY2, ErrZ2);
    Param().GetClusterErrorsSeeding2(mISector, iRow, Param().par.continuousTracking != 0.f ? 125.f : z, sinPhi, DzDs, time, ErrY2, ErrZ2);
  }

  void SetupCommonMemory();
  void* SetPointersDataLinks(void* mem);
  void* SetPointersDataWeights(void* mem);
  void* SetPointersDataScratch(void* mem);
  void* SetPointersDataRows(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersScratchHost(void* mem);
  void* SetPointersCommon(void* mem);
  void* SetPointersTracklets(void* mem);
  void* SetPointersOutput(void* mem);
  void RegisterMemoryAllocation();

  int16_t MemoryResLinks() const { return mMemoryResLinks; }
  int16_t MemoryResScratchHost() const { return mMemoryResScratchHost; }
  int16_t MemoryResCommon() const { return mMemoryResCommon; }
  int16_t MemoryResTracklets() const { return mMemoryResTracklets; }
  int16_t MemoryResOutput() const { return mMemoryResOutput; }
  int16_t MemoryResSectorScratch() const { return mMemoryResSectorScratch; }

  void SetMaxData(const GPUTrackingInOutPointers& io);
  void UpdateMaxData();

  GPUhd() int32_t ISector() const { return mISector; }

  GPUhd() GPUconstantref() const GPUTPCTrackingData& Data() const { return mData; }
  GPUhdi() GPUconstantref() GPUTPCTrackingData& Data()
  {
    return mData;
  }

  GPUhd() GPUglobalref() const GPUTPCRow& Row(int32_t rowIndex) const { return mData.Row(rowIndex); }

  GPUhd() uint32_t NHitsTotal() const { return mData.NumberOfHits(); }
  GPUhd() uint32_t NMaxTracklets() const { return mNMaxTracklets; }
  GPUhd() uint32_t NMaxRowHits() const { return mNMaxRowHits; }
  GPUhd() uint32_t NMaxTracks() const { return mNMaxTracks; }
  GPUhd() uint32_t NMaxTrackHits() const { return mNMaxTrackHits; }
  GPUhd() uint32_t NMaxStartHits() const { return mNMaxStartHits; }
  GPUhd() uint32_t NMaxRowStartHits() const { return mNMaxRowStartHits; }

  GPUd() void SetHitLinkUpData(const GPUTPCRow& row, int32_t hitIndex, calink v) { mData.SetHitLinkUpData(row, hitIndex, v); }
  GPUd() void SetHitLinkDownData(const GPUTPCRow& row, int32_t hitIndex, calink v) { mData.SetHitLinkDownData(row, hitIndex, v); }
  GPUd() calink HitLinkUpData(const GPUTPCRow& row, int32_t hitIndex) const { return mData.HitLinkUpData(row, hitIndex); }
  GPUd() calink HitLinkDownData(const GPUTPCRow& row, int32_t hitIndex) const { return mData.HitLinkDownData(row, hitIndex); }

  GPUd() GPUglobalref() const cahit2* HitData(const GPUTPCRow& row) const { return mData.HitData(row); }
  GPUd() GPUglobalref() const calink* HitLinkUpData(const GPUTPCRow& row) const { return mData.HitLinkUpData(row); }
  GPUd() GPUglobalref() const calink* HitLinkDownData(const GPUTPCRow& row) const { return mData.HitLinkDownData(row); }
  GPUd() GPUglobalref() const calink* FirstHitInBin(const GPUTPCRow& row) const { return mData.FirstHitInBin(row); }

  GPUd() int32_t FirstHitInBin(const GPUTPCRow& row, int32_t binIndex) const { return mData.FirstHitInBin(row, binIndex); }

  GPUd() cahit HitDataY(const GPUTPCRow& row, int32_t hitIndex) const { return mData.HitDataY(row, hitIndex); }
  GPUd() cahit HitDataZ(const GPUTPCRow& row, int32_t hitIndex) const { return mData.HitDataZ(row, hitIndex); }
  GPUd() cahit2 HitData(const GPUTPCRow& row, int32_t hitIndex) const { return mData.HitData(row, hitIndex); }

  GPUhd() int32_t HitInputID(const GPUTPCRow& row, int32_t hitIndex) const { return mData.ClusterDataIndex(row, hitIndex); }

  /**
   * The hit weight is used to determine whether a hit belongs to a certain tracklet or another one
   * competing for the same hit. The tracklet that has a higher weight wins. Comparison is done
   * using the the number of hits in the tracklet (the more hits it has the more it keeps). If
   * tracklets have the same number of hits then it doesn't matter who gets it, but it should be
   * only one. So a unique number (row index is good) is added in the least significant part of
   * the weight
   */
  GPUdi() static int32_t CalculateHitWeight(int32_t NHits, float chi2)
  {
    const float chi2_suppress = 6.f;
    float weight = (((float)NHits * (chi2_suppress - chi2 / 500.f)) * (1e9f / chi2_suppress / 160.f));
    if (weight < 0.f || weight > 2e9f) {
      return 0;
    }
    return ((int32_t)weight);
    // return( (NHits << 16) + num);
  }
  GPUd() void MaximizeHitWeight(const GPUTPCRow& row, int32_t hitIndex, int32_t weight) { mData.MaximizeHitWeight(row, hitIndex, weight); }
  GPUd() void SetHitWeight(const GPUTPCRow& row, int32_t hitIndex, int32_t weight) { mData.SetHitWeight(row, hitIndex, weight); }
  GPUd() int32_t HitWeight(const GPUTPCRow& row, int32_t hitIndex) const { return mData.HitWeight(row, hitIndex); }

  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NTracklets() const { return &mCommonMem->nTracklets; }
  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NRowHits() const { return &mCommonMem->nRowHits; }
  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NStartHits() const { return &mCommonMem->nStartHits; }

  GPUhd() GPUglobalref() const GPUTPCHitId& TrackletStartHit(int32_t i) const { return mTrackletStartHits[i]; }
  GPUhd() GPUglobalref() const GPUTPCHitId* TrackletStartHits() const { return mTrackletStartHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackletStartHits() { return mTrackletStartHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackletTmpStartHits() const { return mTrackletTmpStartHits; }
  GPUhd() GPUglobalref() const GPUTPCTracklet& Tracklet(int32_t i) const { return mTracklets[i]; }
  GPUhd() GPUglobalref() GPUTPCTracklet* Tracklets() const { return mTracklets; }
  GPUhd() GPUglobalref() calink* TrackletRowHits() const { return mTrackletRowHits; }

  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NTracks() const { return &mCommonMem->nTracks; }
  GPUhd() GPUglobalref() GPUTPCTrack* Tracks() const { return mTracks; }
  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NTrackHits() const { return &mCommonMem->nTrackHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackHits() const { return mTrackHits; }

  GPUhd() GPUglobalref() GPUTPCRow* TrackingDataRows() const { return (mData.Rows()); }
  GPUhd() GPUglobalref() int32_t* RowStartHitCountOffset() const { return (mRowStartHitCountOffset); }

  struct trackSortData {
    int32_t fTtrack; // Track ID
    float fSortVal;  // Value to sort for
  };

  void* LinkTmpMemory() { return mLinkTmpMemory; }

#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  char* mStageAtSync = nullptr; // Temporary performance variable: Pointer to array storing current stage for every thread at every sync point
#endif

 private:
  friend class GPUTPCNeighboursFinder;
  friend class GPUTPCStartHitsSorter;
  friend class GPUTPCStartHitsFinder;
  char* mLinkTmpMemory = nullptr; // tmp memory for hits after neighbours finder

  int32_t mISector = -1; // Number of sector

  GPUTPCTrackingData mData; // The TrackingData object. It is used to encapsulate the storage in memory from the access

  uint32_t mNMaxStartHits = 0;
  uint32_t mNMaxRowStartHits = 0;
  uint32_t mNMaxTracklets = 0;
  uint32_t mNMaxRowHits = 0;
  uint32_t mNMaxTracks = 0;
  uint32_t mNMaxTrackHits = 0;
  uint16_t mMemoryResLinks = (uint16_t)-1;
  uint16_t mMemoryResScratch = (uint16_t)-1;
  uint16_t mMemoryResScratchHost = (uint16_t)-1;
  uint16_t mMemoryResCommon = (uint16_t)-1;
  uint16_t mMemoryResTracklets = (uint16_t)-1;
  uint16_t mMemoryResOutput = (uint16_t)-1;
  uint16_t mMemoryResSectorScratch = (uint16_t)-1;

  // GPU Temp Arrays
  GPUglobalref() int32_t* mRowStartHitCountOffset = nullptr;   // Offset, length and new offset of start hits in row
  GPUglobalref() GPUTPCHitId* mTrackletTmpStartHits = nullptr; // Unsorted start hits
  GPUglobalref() char* mGPUTrackletTemp = nullptr;             // Temp Memory for GPU Tracklet Constructor

  // event
  GPUglobalref() commonMemoryStruct* mCommonMem = nullptr;  // common event memory
  GPUglobalref() GPUTPCHitId* mTrackletStartHits = nullptr; // start hits for the tracklets
  GPUglobalref() GPUTPCTracklet* mTracklets = nullptr;      // tracklets
  GPUglobalref() calink* mTrackletRowHits = nullptr;        // Hits for each Tracklet in each row
  GPUglobalref() GPUTPCTrack* mTracks = nullptr;            // reconstructed tracks
  GPUglobalref() GPUTPCHitId* mTrackHits = nullptr;         // array of track hit numbers

  static int32_t StarthitSortComparison(const void* a, const void* b);
};
} // namespace o2::gpu

#endif // GPUTPCTRACKER_H
