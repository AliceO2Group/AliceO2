// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#endif

  struct StructGPUParameters {
    GPUAtomic(unsigned int) nextTracklet; // Next Tracklet to process
  };

  MEM_CLASS_PRE2()
  struct StructGPUParametersConst {
    GPUglobalref() char* gpumem; // Base pointer to GPU memory (Needed for OpenCL for verification)
  };

  struct commonMemoryStruct {
    commonMemoryStruct() : nTracklets(0), nTracks(0), nLocalTracks(0), nTrackHits(0), nLocalTrackHits(0), gpuParameters() {}
    GPUAtomic(unsigned int) nTracklets; // number of tracklets
    GPUAtomic(unsigned int) nTracks;    // number of reconstructed tracks
    int nLocalTracks;                   // number of reconstructed tracks before global tracking
    GPUAtomic(unsigned int) nTrackHits; // number of track hits
    int nLocalTrackHits;                // see above
    int kernelError;                    // Error code during kernel execution
    StructGPUParameters gpuParameters;  // GPU parameters
  };

  MEM_CLASS_PRE2()
  void SetSlice(int iSlice);
  MEM_CLASS_PRE2()
  void InitializeProcessor();
  MEM_CLASS_PRE2()
  void InitializeRows(const MEM_CONSTANT(GPUParam) * param) { mData.InitializeRows(*param); }

  int CheckEmptySlice();
  void WriteOutputPrepare();
  void WriteOutput();

// GPU Tracker Interface
#if !defined(GPUCA_GPUCODE_DEVICE)
  // Debugging Stuff
  void DumpSliceData(std::ostream& out);    // Dump Input Slice Data
  void DumpLinks(std::ostream& out);        // Dump all links to file (for comparison after NeighboursFinder/Cleaner)
  void DumpStartHits(std::ostream& out);    // Same for Start Hits
  void DumpHitWeights(std::ostream& out);   //....
  void DumpTrackHits(std::ostream& out);    // Same for Track Hits
  void DumpTrackletHits(std::ostream& out); // Same for Track Hits
  void DumpOutput(FILE* out);               // Similar for output

  int ReadEvent();

  GPUh() const GPUTPCClusterData* ClusterData() const { return mData.ClusterData(); }

  GPUh() MakeType(const MEM_LG(GPUTPCRow) &) Row(const GPUTPCHitId& HitId) const { return mData.Row(HitId.RowIndex()); }

  GPUhd() GPUTPCSliceOutput* Output() const { return mOutput; }
#endif
  GPUhdni() GPUglobalref() commonMemoryStruct* CommonMemory() const
  {
    return (mCommonMem);
  }

  MEM_CLASS_PRE2()
  GPUd() void GetErrors2(int iRow, const MEM_LG2(GPUTPCTrackParam) & t, float& ErrY2, float& ErrZ2) const
  {
    // mCAParam.GetClusterErrors2( iRow, mCAParam.GetContinuousTracking() != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
    mCAParam->GetClusterRMS2(iRow, mCAParam->ContinuousTracking != 0.f ? 125.f : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2);
  }
  GPUd() void GetErrors2(int iRow, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
  {
    // mCAParam.GetClusterErrors2( iRow, mCAParam.GetContinuousTracking() != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
    mCAParam->GetClusterRMS2(iRow, mCAParam->ContinuousTracking != 0.f ? 125.f : z, sinPhi, DzDs, ErrY2, ErrZ2);
  }

  void SetupCommonMemory();
  void* SetPointersDataInput(void* mem);
  void* SetPointersDataScratch(void* mem);
  void* SetPointersDataRows(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersScratchHost(void* mem);
  void* SetPointersCommon(void* mem);
  void* SetPointersTracklets(void* mem);
  void* SetPointersOutput(void* mem);
  void RegisterMemoryAllocation();

  short MemoryResLinksScratch() { return mMemoryResLinksScratch; }
  short MemoryResScratchHost() { return mMemoryResScratchHost; }
  short MemoryResCommon() { return mMemoryResCommon; }
  short MemoryResTracklets() { return mMemoryResTracklets; }
  short MemoryResOutput() { return mMemoryResOutput; }

  void SetMaxData();
  void UpdateMaxData();

  GPUhd() GPUconstantref() const MEM_CONSTANT(GPUParam) & Param() const { return *mCAParam; }
  GPUhd() GPUconstantref() const MEM_CONSTANT(GPUParam) * pParam() const { return mCAParam; }
  GPUhd() int ISlice() const { return mISlice; }

  GPUhd() GPUconstantref() const MEM_LG(GPUTPCSliceData) & Data() const { return mData; }
  GPUhd() GPUconstantref() MEM_LG(GPUTPCSliceData) & Data()
  {
    return mData;
  }

  GPUhd() GPUglobalref() const MEM_GLOBAL(GPUTPCRow) & Row(int rowIndex) const { return mData.Row(rowIndex); }

  GPUhd() unsigned int NHitsTotal() const { return mData.NumberOfHits(); }
  GPUhd() unsigned int NMaxTracklets() const { return mNMaxTracklets; }
  GPUhd() unsigned int NMaxTracks() const { return mNMaxTracks; }
  GPUhd() unsigned int NMaxTrackHits() const { return mNMaxTrackHits; }
  GPUhd() unsigned int NMaxStartHits() const { return mNMaxStartHits; }

  MEM_TEMPLATE()
  GPUd() void SetHitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, int hitIndex, calink v) { mData.SetHitLinkUpData(row, hitIndex, v); }
  MEM_TEMPLATE()
  GPUd() void SetHitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, int hitIndex, calink v) { mData.SetHitLinkDownData(row, hitIndex, v); }
  MEM_TEMPLATE()
  GPUd() calink HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.HitLinkUpData(row, hitIndex); }
  MEM_TEMPLATE()
  GPUd() calink HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.HitLinkDownData(row, hitIndex); }

  MEM_TEMPLATE()
  GPUd() GPUglobalref() const cahit2* HitData(const MEM_TYPE(GPUTPCRow) & row) const { return mData.HitData(row); }
  MEM_TEMPLATE()
  GPUd() GPUglobalref() const calink* HitLinkUpData(const MEM_TYPE(GPUTPCRow) & row) const { return mData.HitLinkUpData(row); }
  MEM_TEMPLATE()
  GPUd() GPUglobalref() const calink* HitLinkDownData(const MEM_TYPE(GPUTPCRow) & row) const { return mData.HitLinkDownData(row); }
  MEM_TEMPLATE()
  GPUd() GPUglobalref() const calink* FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row) const { return mData.FirstHitInBin(row); }

  MEM_TEMPLATE()
  GPUd() int FirstHitInBin(const MEM_TYPE(GPUTPCRow) & row, int binIndex) const { return mData.FirstHitInBin(row, binIndex); }

  MEM_TEMPLATE()
  GPUd() cahit HitDataY(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.HitDataY(row, hitIndex); }
  MEM_TEMPLATE()
  GPUd() cahit HitDataZ(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.HitDataZ(row, hitIndex); }
  MEM_TEMPLATE()
  GPUd() cahit2 HitData(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.HitData(row, hitIndex); }

  MEM_TEMPLATE()
  GPUhd() int HitInputID(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.ClusterDataIndex(row, hitIndex); }

  /**
 * The hit weight is used to determine whether a hit belongs to a certain tracklet or another one
 * competing for the same hit. The tracklet that has a higher weight wins. Comparison is done
 * using the the number of hits in the tracklet (the more hits it has the more it keeps). If
 * tracklets have the same number of hits then it doesn't matter who gets it, but it should be
 * only one. So a unique number (row index is good) is added in the least significant part of
 * the weight
 */
  GPUd() static int CalculateHitWeight(int NHits, float chi2, int)
  {
    const float chi2_suppress = 6.f;
    float weight = (((float)NHits * (chi2_suppress - chi2 / 500.f)) * (1e9f / chi2_suppress / 160.f));
    if (weight < 0.f || weight > 2e9f) {
      return 0;
    }
    return ((int)weight);
    // return( (NHits << 16) + num);
  }
  MEM_TEMPLATE()
  GPUd() void MaximizeHitWeight(const MEM_TYPE(GPUTPCRow) & row, int hitIndex, int weight) { mData.MaximizeHitWeight(row, hitIndex, weight); }
  MEM_TEMPLATE()
  GPUd() void SetHitWeight(const MEM_TYPE(GPUTPCRow) & row, int hitIndex, int weight) { mData.SetHitWeight(row, hitIndex, weight); }
  MEM_TEMPLATE()
  GPUd() int HitWeight(const MEM_TYPE(GPUTPCRow) & row, int hitIndex) const { return mData.HitWeight(row, hitIndex); }

  GPUhd() GPUglobalref() GPUAtomic(unsigned int) * NTracklets() const { return &mCommonMem->nTracklets; }

  GPUhd() const GPUTPCHitId& TrackletStartHit(int i) const { return mTrackletStartHits[i]; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackletStartHits() const { return mTrackletStartHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackletTmpStartHits() const { return mTrackletTmpStartHits; }
  MEM_CLASS_PRE2()
  GPUhd() const MEM_LG2(GPUTPCTracklet) & Tracklet(int i) const { return mTracklets[i]; }
  GPUhd() GPUglobalref() MEM_GLOBAL(GPUTPCTracklet) * Tracklets() const { return mTracklets; }
  GPUhd() GPUglobalref() calink* TrackletRowHits() const { return mTrackletRowHits; }

  GPUhd() GPUglobalref() GPUAtomic(unsigned int) * NTracks() const { return &mCommonMem->nTracks; }
  GPUhd() GPUglobalref() MEM_GLOBAL(GPUTPCTrack) * Tracks() const { return mTracks; }
  GPUhd() GPUglobalref() GPUAtomic(unsigned int) * NTrackHits() const { return &mCommonMem->nTrackHits; }
  GPUhd() GPUglobalref() GPUTPCHitId* TrackHits() const { return mTrackHits; }

  GPUhd() GPUglobalref() MEM_GLOBAL(GPUTPCRow) * SliceDataRows() const { return (mData.Rows()); }

  GPUhd() GPUglobalref() int* RowStartHitCountOffset() const { return (mRowStartHitCountOffset); }
  GPUhd() GPUglobalref() StructGPUParameters* GPUParameters() const { return (&mCommonMem->gpuParameters); }
  GPUhd() MakeType(MEM_LG(StructGPUParametersConst) *) GPUParametersConst()
  {
    return (&mGPUParametersConst);
  }
  GPUhd() MakeType(MEM_LG(const StructGPUParametersConst) *) GetGPUParametersConst() const { return (&mGPUParametersConst); }
  GPUhd() void SetGPUTextureBase(const void* val) { mData.SetGPUTextureBase(val); }

  struct trackSortData {
    int fTtrack;    // Track ID
    float fSortVal; // Value to sort for
  };

  void PerformGlobalTracking(GPUTPCTracker& sliceLeft, GPUTPCTracker& sliceRight);
  void PerformGlobalTracking(GPUTPCTracker& sliceTarget, bool right);

  void* LinkTmpMemory() { return mLinkTmpMemory; }

#if !defined(GPUCA_GPUCODE)
  GPUh() int PerformGlobalTrackingRun(GPUTPCTracker& sliceSource, int iTrack, int rowIndex, float angle, int direction);
#endif
#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  char* mStageAtSync = nullptr; // Temporary performance variable: Pointer to array storing current stage for every thread at every sync point
#endif

 private:
  char* mLinkTmpMemory; // tmp memory for hits after neighbours finder

  int mISlice; // Number of slice

  /** A pointer to the ClusterData object that the SliceData was created from. This can be used to
 * merge clusters from inside the SliceTracker code and recreate the SliceData. */
  MEM_LG(GPUTPCSliceData)
  mData; // The SliceData object. It is used to encapsulate the storage in memory from the access

  unsigned int mNMaxStartHits;
  unsigned int mNMaxTracklets;
  unsigned int mNMaxTracks;
  unsigned int mNMaxTrackHits;
  short mMemoryResLinksScratch;
  short mMemoryResScratch;
  short mMemoryResScratchHost;
  short mMemoryResCommon;
  short mMemoryResTracklets;
  short mMemoryResOutput;

  // GPU Temp Arrays
  GPUglobalref() int* mRowStartHitCountOffset;       // Offset, length and new offset of start hits in row
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

  static int StarthitSortComparison(const void* a, const void* b);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKER_H
