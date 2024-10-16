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
#include "GPUTPCSliceData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCTracklet.h"
#include "GPUProcessor.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCSliceOutput;
struct GPUTPCClusterData;
MEM_CLASS_PRE()
struct GPUParam;
MEM_CLASS_PRE()
class GPUTPCTrack;
MEM_CLASS_PRE()
class GPUTPCTrackParam;
MEM_CLASS_PRE()
class GPUTPCRow;

MEM_CLASS_PRE()
class GPUTPCTracker : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE_DEVICE
  GPUTPCTracker();
  ~GPUTPCTracker();
  GPUTPCTracker(const GPUTPCTracker&) CON_DELETE;
  GPUTPCTracker& operator=(const GPUTPCTracker&) CON_DELETE;

  MEM_CLASS_PRE2()
  void SetSlice(int32_t iSlice);
  MEM_CLASS_PRE2()
  void InitializeProcessor();
  MEM_CLASS_PRE2()
  void InitializeRows(const MEM_CONSTANT(GPUParam) * param) { mData.InitializeRows(*param); }

  int32_t CheckEmptySlice();
  void WriteOutputPrepare();
  void WriteOutput();

  // Debugging Stuff
  void DumpSliceData(std::ostream& out);        // Dump Input Slice Data
  void DumpLinks(std::ostream& out, int32_t phase); // Dump all links to file (for comparison after NeighboursFinder/Cleaner)
  void DumpStartHits(std::ostream& out);        // Same for Start Hits
  void DumpHitWeights(std::ostream& out);       //....
  void DumpTrackHits(std::ostream& out);        // Same for Track Hits
  void DumpTrackletHits(std::ostream& out);     // Same for Track Hits
  void DumpOutput(std::ostream& out);           // Similar for output
#endif

  struct StructGPUParameters {
    GPUAtomic(uint32_t) nextStartHit; // Next Tracklet to process
  };

  MEM_CLASS_PRE2()
  struct StructGPUParametersConst {
    GPUglobalref() char* gpumem; // Base pointer to GPU memory (Needed for OpenCL for verification)
  };

  struct commonMemoryStruct {
    commonMemoryStruct() : nStartHits(0), nTracklets(0), nRowHits(0), nTracks(0), nLocalTracks(0), nTrackHits(0), nLocalTrackHits(0), gpuParameters() {}
    GPUAtomic(uint32_t) nStartHits;     // number of start hits
    GPUAtomic(uint32_t) nTracklets;     // number of tracklets
    GPUAtomic(uint32_t) nRowHits;       // number of tracklet hits
    GPUAtomic(uint32_t) nTracks;        // number of reconstructed tracks
    int32_t nLocalTracks;               // number of reconstructed tracks before global tracking
    GPUAtomic(uint32_t) nTrackHits;     // number of track hits
    int32_t nLocalTrackHits;            // see above
    StructGPUParameters gpuParameters;  // GPU parameters
  };

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  GPUhdi() GPUglobalref() const GPUTPCClusterData* ClusterData() const
  {
    return mData.ClusterData();
  }
  GPUhdi() MakeType(const MEM_LG(GPUTPCRow) &) Row(const GPUTPCHitId& HitId) const { return mData.Row(HitId.RowIndex()); }
  GPUhdi() GPUglobalref() GPUTPCSliceOutput* Output() const { return mOutput; }
#endif
  GPUhdni() GPUglobalref() commonMemoryStruct* CommonMemory() const
  {
    return (mCommonMem);
  }

  MEM_CLASS_PRE2()
  GPUdi() static void GetErrors2Seeding(const MEM_CONSTANT(GPUParam) & param, char sector, int32_t iRow, const MEM_LG2(GPUTPCTrackParam) & t, float time, float& ErrY2, float& ErrZ2)
  {
    // param.GetClusterErrors2(sector, iRow, param.GetContinuousTracking() != 0. ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, 0.f, 0.f, ErrY2, ErrZ2);
    param.GetClusterErrorsSeeding2(sector, iRow, param.par.continuousTracking != 0.f ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, ErrY2, ErrZ2);
  }

  MEM_CLASS_PRE2()
  GPUdi() void GetErrors2Seeding(int32_t iRow, const MEM_LG2(GPUTPCTrackParam) & t, float time, float& ErrY2, float& ErrZ2) const
  {
    // Param().GetClusterErrors2(mISlice, iRow, Param().GetContinuousTracking() != 0. ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, 0.f, 0.f, ErrY2, ErrZ2);
    Param().GetClusterErrorsSeeding2(mISlice, iRow, Param().par.continuousTracking != 0.f ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), time, ErrY2, ErrZ2);
  }
  GPUdi() void GetErrors2Seeding(int32_t iRow, float z, float sinPhi, float DzDs, float time, float& ErrY2, float& ErrZ2) const
  {
    // Param().GetClusterErrors2(mISlice, iRow, Param().GetContinuousTracking() != 0. ? 125.f : z, sinPhi, DzDs, time, 0.f, 0.f, ErrY2, ErrZ2);
    Param().GetClusterErrorsSeeding2(mISlice, iRow, Param().par.continuousTracking != 0.f ? 125.f : z, sinPhi, DzDs, time, ErrY2, ErrZ2);
  }

  void SetupCommonMemory();
  bool SliceDataOnGPU();
  void* SetPointersDataInput(void* mem);
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
  int16_t MemoryResSliceScratch() const { return mMemoryResSliceScratch; }
  int16_t MemoryResSliceInput() const { return mMemoryResSliceInput; }

  void SetMaxData(const GPUTrackingInOutPointers& io);
  void UpdateMaxData();

  GPUhd() int32_t ISlice() const { return mISlice; }

  GPUhd() GPUconstantref() const MEM_LG(GPUTPCSliceData) & Data() const { return mData; }
  GPUhdi() GPUconstantref() MEM_LG(GPUTPCSliceData) & Data()
  {
    return mData;
  }

  GPUhd() GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & Row(int32_t rowIndex) const { return mData.Row(rowIndex); }

  GPUhd() uint32_t NHitsTotal() const { return mData.NumberOfHits(); }
  GPUhd() uint32_t NMaxTracklets() const { return mNMaxTracklets; }
  GPUhd() uint32_t NMaxRowHits() const { return mNMaxRowHits; }
  GPUhd() uint32_t NMaxTracks() const { return mNMaxTracks; }
  GPUhd() uint32_t NMaxTrackHits() const { return mNMaxTrackHits; }
  GPUhd() uint32_t NMaxStartHits() const { return mNMaxStartHits; }
  GPUhd() uint32_t NMaxRowStartHits() const { return mNMaxRowStartHits; }

  MEM_TEMPLATE()
  GPUd() void SetHitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex, calink v) { mData.SetHitLinkUpData(row, hitIndex, v); }
  MEM_TEMPLATE()
  GPUd() void SetHitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex, calink v) { mData.SetHitLinkDownData(row, hitIndex, v); }
  MEM_TEMPLATE()
  GPUd() calink HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.HitLinkUpData(row, hitIndex); }
  MEM_TEMPLATE()
  GPUd() calink HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.HitLinkDownData(row, hitIndex); }

  MEM_TEMPLATE()
  GPUd() GPUglobalref() const cahit2* HitData(const MEM_TYPE(GPUTPCRow) & row) const { return mData.HitData(row); }
  MEM_TEMPLATE()
  GPUd() GPUglobalref() const calink* HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row) const { return mData.HitLinkUpData(row); }
  MEM_TEMPLATE()
  GPUd() GPUglobalref() const calink* HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row) const { return mData.HitLinkDownData(row); }
  MEM_TEMPLATE()
  GPUd() GPUglobalref() const calink* FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row) const { return mData.FirstHitInBin(row); }

  MEM_TEMPLATE()
  GPUd() int32_t FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row, int32_t binIndex) const { return mData.FirstHitInBin(row, binIndex); }

  MEM_TEMPLATE()
  GPUd() cahit HitDataY(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.HitDataY(row, hitIndex); }
  MEM_TEMPLATE()
  GPUd() cahit HitDataZ(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.HitDataZ(row, hitIndex); }
  MEM_TEMPLATE()
  GPUd() cahit2 HitData(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.HitData(row, hitIndex); }

  MEM_TEMPLATE()
  GPUhd() int32_t HitInputID(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.ClusterDataIndex(row, hitIndex); }

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
  MEM_TEMPLATE()
  GPUd() void MaximizeHitWeight(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex, int32_t weight) { mData.MaximizeHitWeight(row, hitIndex, weight); }
  MEM_TEMPLATE()
  GPUd() void SetHitWeight(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex, int32_t weight) { mData.SetHitWeight(row, hitIndex, weight); }
  MEM_TEMPLATE()
  GPUd() int32_t HitWeight(const MEM_TYPE(GPUTPCRow) & row, int32_t hitIndex) const { return mData.HitWeight(row, hitIndex); }

  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NTracklets() const { return &mCommonMem->nTracklets; }
  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NRowHits() const { return &mCommonMem->nRowHits; }
  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NStartHits() const { return &mCommonMem->nStartHits; }

  GPUhd() GPUglobalref() const GPUTPCHitId& TrackletStartHit(int32_t i) const { return mTrackletStartHits[i]; }
  GPUhd() GPUglobalref() const GPUTPCHitId* TrackletStartHits() const { return mTrackletStartHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackletStartHits() { return mTrackletStartHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackletTmpStartHits() const { return mTrackletTmpStartHits; }
  MEM_CLASS_PRE2()
  GPUhd() GPUglobalref() const MEM_LG2(GPUTPCTracklet) & Tracklet(int32_t i) const { return mTracklets[i]; }
  GPUhd() GPUglobalref() MEM_GLOBAL(GPUTPCTracklet) * Tracklets() const { return mTracklets; }
  GPUhd() GPUglobalref() calink* TrackletRowHits() const { return mTrackletRowHits; }

  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NTracks() const { return &mCommonMem->nTracks; }
  GPUhd() GPUglobalref() MEM_GLOBAL(GPUTPCTrack) * Tracks() const { return mTracks; }
  GPUhd() GPUglobalref() GPUAtomic(uint32_t) * NTrackHits() const { return &mCommonMem->nTrackHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackHits() const { return mTrackHits; }

  GPUhd() GPUglobalref() MEM_GLOBAL(GPUTPCRow) * SliceDataRows() const { return (mData.Rows()); }
  GPUhd() GPUglobalref() int32_t* RowStartHitCountOffset() const { return (mRowStartHitCountOffset); }
  GPUhd() GPUglobalref() StructGPUParameters* GPUParameters() const { return (&mCommonMem->gpuParameters); }
  GPUhd() MakeType(MEM_LG(StructGPUParametersConst) *) GPUParametersConst()
  {
    return (&mGPUParametersConst);
  }
  GPUhd() MakeType(MEM_LG(const StructGPUParametersConst) *) GetGPUParametersConst() const { return (&mGPUParametersConst); }
  GPUhd() void SetGPUTextureBase(GPUglobalref() const void* val) { mData.SetGPUTextureBase(val); }

  struct trackSortData {
    int32_t fTtrack; // Track ID
    float fSortVal; // Value to sort for
  };

  void* LinkTmpMemory() { return mLinkTmpMemory; }

#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  char* mStageAtSync = nullptr; // Temporary performance variable: Pointer to array storing current stage for every thread at every sync point
#endif

 private:
  friend class GPUTPCNeighboursFinder;
  friend class GPUTPCStartHitsSorter;
  friend class GPUTPCStartHitsFinder;
  char* mLinkTmpMemory; // tmp memory for hits after neighbours finder

  int32_t mISlice; // Number of slice

  /** A pointer to the ClusterData object that the SliceData was created from. This can be used to
 * merge clusters from inside the SliceTracker code and recreate the SliceData. */
  MEM_LG(GPUTPCSliceData)
  mData; // The SliceData object. It is used to encapsulate the storage in memory from the access

  uint32_t mNMaxStartHits;
  uint32_t mNMaxRowStartHits;
  uint32_t mNMaxTracklets;
  uint32_t mNMaxRowHits;
  uint32_t mNMaxTracks;
  uint32_t mNMaxTrackHits;
  int16_t mMemoryResLinks;
  int16_t mMemoryResScratch;
  int16_t mMemoryResScratchHost;
  int16_t mMemoryResCommon;
  int16_t mMemoryResTracklets;
  int16_t mMemoryResOutput;
  int16_t mMemoryResSliceScratch;
  int16_t mMemoryResSliceInput;

  // GPU Temp Arrays
  GPUglobalref() int32_t* mRowStartHitCountOffset;   // Offset, length and new offset of start hits in row
  GPUglobalref() GPUTPCHitId* mTrackletTmpStartHits; // Unsorted start hits
  GPUglobalref() char* mGPUTrackletTemp;             // Temp Memory for GPU Tracklet Constructor

  MEM_LG(StructGPUParametersConst)
  mGPUParametersConst; // Parameters for GPU if this is a GPU tracker

  // event
  GPUglobalref() commonMemoryStruct* mCommonMem;          // common event memory
  GPUglobalref() GPUTPCHitId* mTrackletStartHits;         // start hits for the tracklets
  GPUglobalref() MEM_GLOBAL(GPUTPCTracklet) * mTracklets; // tracklets
  GPUglobalref() calink* mTrackletRowHits;                // Hits for each Tracklet in each row
  GPUglobalref() MEM_GLOBAL(GPUTPCTrack) * mTracks;       // reconstructed tracks
  GPUglobalref() GPUTPCHitId* mTrackHits;                 // array of track hit numbers

  // output
  GPUglobalref() GPUTPCSliceOutput* mOutput; // address of pointer pointing to SliceOutput Object
  void* mOutputMemory;                       // Pointer to output memory if stored internally

  static int32_t StarthitSortComparison(const void* a, const void* b);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKER_H
