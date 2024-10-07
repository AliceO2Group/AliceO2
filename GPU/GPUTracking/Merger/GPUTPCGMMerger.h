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

/// \file GPUTPCGMMerger.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMMERGER_H
#define GPUTPCGMMERGER_H

#include "GPUParam.h"
#include "GPUTPCDef.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUCommonDef.h"
#include "GPUProcessor.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUGeneralKernels.h"

#if !defined(GPUCA_GPUCODE)
#include <cmath>
#include <iostream>
#endif // GPUCA_GPUCODE

namespace o2
{
namespace base
{
class MatLayerCylSet;
}
namespace tpc
{
struct ClusterNative;
}
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCSliceTrack;
class GPUTPCSliceOutput;
class GPUTPCGMTrackParam;
class GPUTPCTracker;
class GPUChainTracking;
class GPUTPCGMPolynomialField;
struct GPUTPCGMLoopData;
struct MergeLooperParam;

/**
 * @class GPUTPCGMMerger
 *
 */
class GPUTPCGMMerger : public GPUProcessor
{
 public:
  GPUTPCGMMerger();
  ~GPUTPCGMMerger() CON_DEFAULT;
  GPUTPCGMMerger(const GPUTPCGMMerger&) CON_DELETE;
  const GPUTPCGMMerger& operator=(const GPUTPCGMMerger&) const CON_DELETE;
  static CONSTEXPR const int32_t NSLICES = GPUCA_NSLICES; //* N slices

  struct memory {
    GPUAtomic(uint32_t) nRetryRefit;
    GPUAtomic(uint32_t) nLoopData;
    GPUAtomic(uint32_t) nUnpackedTracks;
    GPUAtomic(uint32_t) nOutputTracks;
    GPUAtomic(uint32_t) nOutputTrackClusters;
    GPUAtomic(uint32_t) nO2Tracks;
    GPUAtomic(uint32_t) nO2ClusRefs;
    const GPUTPCTrack* firstGlobalTracks[NSLICES];
    GPUAtomic(uint32_t) tmpCounter[2 * NSLICES];
    GPUAtomic(uint32_t) nLooperMatchCandidates;
  };

  struct trackCluster {
    uint32_t id;
    uint8_t row;
    uint8_t slice;
    uint8_t leg;
  };

  struct tmpSort {
    uint32_t x;
    float y;
  };

  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);
  void* SetPointersMerger(void* mem);
  void* SetPointersRefitScratch(void* mem);
  void* SetPointersRefitScratch2(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersOutputO2(void* mem);
  void* SetPointersOutputO2Clus(void* mem);
  void* SetPointersOutputO2MC(void* mem);
  void* SetPointersOutputO2Scratch(void* mem);
  void* SetPointersOutputState(void* mem);
  void* SetPointersMemory(void* mem);

  void SetSliceData(int32_t index, const GPUTPCSliceOutput* sliceData) { mkSlices[index] = sliceData; }

  GPUhdi() int32_t NOutputTracks() const { return mMemory->nOutputTracks; }
  GPUhdi() const GPUTPCGMMergedTrack* OutputTracks() const { return mOutputTracks; }
  GPUhdi() GPUTPCGMMergedTrack* OutputTracks() { return mOutputTracks; }
  GPUhdi() const GPUdEdxInfo* OutputTracksdEdx() const { return mOutputTracksdEdx; }
  GPUhdi() GPUdEdxInfo* OutputTracksdEdx() { return mOutputTracksdEdx; }
  GPUhdi() uint32_t NClusters() const { return mNClusters; }
  GPUhdi() uint32_t NMaxClusters() const { return mNMaxClusters; }
  GPUhdi() uint32_t NMaxTracks() const { return mNMaxTracks; }
  GPUhdi() uint32_t NMaxOutputTrackClusters() const { return mNMaxOutputTrackClusters; }
  GPUhdi() uint32_t NOutputTrackClusters() const { return mMemory->nOutputTrackClusters; }
  GPUhdi() const GPUTPCGMMergedTrackHit* Clusters() const { return mClusters; }
  GPUhdi() GPUTPCGMMergedTrackHit* Clusters() { return (mClusters); }
  GPUhdi() const GPUTPCGMMergedTrackHitXYZ* ClustersXYZ() const { return mClustersXYZ; }
  GPUhdi() GPUTPCGMMergedTrackHitXYZ* ClustersXYZ() { return (mClustersXYZ); }
  GPUhdi() GPUAtomic(uint32_t) * ClusterAttachment() const { return mClusterAttachment; }
  GPUhdi() uint32_t* TrackOrderAttach() const { return mTrackOrderAttach; }
  GPUhdi() uint32_t* TrackOrderProcess() const { return mTrackOrderProcess; }
  GPUhdi() uint32_t* RetryRefitIds() const { return mRetryRefitIds; }
  GPUhdi() uint8_t* ClusterStateExt() const { return mClusterStateExt; }
  GPUhdi() GPUTPCGMLoopData* LoopData() const { return mLoopData; }
  GPUhdi() memory* Memory() const { return mMemory; }
  GPUhdi() GPUAtomic(uint32_t) * TmpCounter() { return mMemory->tmpCounter; }
  GPUhdi() uint2* ClusRefTmp() { return mClusRefTmp; }
  GPUhdi() uint32_t* TrackSort() { return mTrackSort; }
  GPUhdi() tmpSort* TrackSortO2() { return mTrackSortO2; }
  GPUhdi() MergeLooperParam* LooperCandidates() { return mLooperCandidates; }
  GPUhdi() GPUAtomic(uint32_t) * SharedCount() { return mSharedCount; }
  GPUhdi() gputpcgmmergertypes::GPUTPCGMBorderRange* BorderRange(int32_t i) { return mBorderRange[i]; }
  GPUhdi() const gputpcgmmergertypes::GPUTPCGMBorderRange* BorderRange(int32_t i) const { return mBorderRange[i]; }
  GPUhdi() GPUTPCGMBorderTrack* BorderTracks(int32_t i) { return mBorder[i]; }
  GPUhdi() o2::tpc::TrackTPC* OutputTracksTPCO2() { return mOutputTracksTPCO2; }
  GPUhdi() uint32_t* OutputClusRefsTPCO2() { return mOutputClusRefsTPCO2; }
  GPUhdi() o2::MCCompLabel* OutputTracksTPCO2MC() { return mOutputTracksTPCO2MC; }
  GPUhdi() uint32_t NOutputTracksTPCO2() const { return mMemory->nO2Tracks; }
  GPUhdi() uint32_t NOutputClusRefsTPCO2() const { return mMemory->nO2ClusRefs; }
  GPUhdi() GPUTPCGMSliceTrack* SliceTrackInfos() { return mSliceTrackInfos; }
  GPUhdi() int32_t NMaxSingleSliceTracks() const { return mNMaxSingleSliceTracks; }
  GPUhdi() int32_t* TrackIDs() { return mTrackIDs; }
  GPUhdi() int32_t* TmpSortMemory() { return mTmpSortMemory; }

  GPUd() uint16_t MemoryResMemory() { return mMemoryResMemory; }
  GPUd() uint16_t MemoryResOutput() const { return mMemoryResOutput; }
  GPUd() uint16_t MemoryResOutputState() const { return mMemoryResOutputState; }
  GPUd() uint16_t MemoryResOutputO2() const { return mMemoryResOutputO2; }
  GPUd() uint16_t MemoryResOutputO2Clus() const { return mMemoryResOutputO2Clus; }
  GPUd() uint16_t MemoryResOutputO2MC() const { return mMemoryResOutputO2MC; }
  GPUd() uint16_t MemoryResOutputO2Scratch() const { return mMemoryResOutputO2Scratch; }

  GPUd() int32_t RefitSliceTrack(GPUTPCGMSliceTrack& sliceTrack, const GPUTPCTrack* inTrack, float alpha, int32_t slice);
  GPUd() void SetTrackClusterZT(GPUTPCGMSliceTrack& track, int32_t iSlice, const GPUTPCTrack* sliceTr);

  int32_t CheckSlices();
  GPUd() void RefitSliceTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSlice);
  GPUd() void UnpackSliceGlobal(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSlice);
  GPUd() void UnpackSaveNumber(int32_t id);
  GPUd() void UnpackResetIds(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSlice);
  GPUd() void MergeCE(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ClearTrackLinks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, bool output);
  GPUd() void MergeWithinSlicesPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void MergeSlicesPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t border0, int32_t border1, int8_t useOrigTrackParam);
  template <int32_t I>
  GPUd() void MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSlice, int8_t withinSlice, int8_t mergeMode);
  GPUd() void MergeBorderTracksSetup(int32_t& n1, int32_t& n2, GPUTPCGMBorderTrack*& b1, GPUTPCGMBorderTrack*& b2, int32_t& jSlice, int32_t iSlice, int8_t withinSlice, int8_t mergeMode) const;
  template <int32_t I>
  GPUd() void MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, gputpcgmmergertypes::GPUTPCGMBorderRange* range, int32_t N, int32_t cmpMax);
  GPUd() void SortTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void SortTracksQPt(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void SortTracksPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void PrepareClustersForFit0(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void PrepareClustersForFit1(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void PrepareClustersForFit2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void LinkGlobalTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void CollectMergedTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void Finalize0(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void Finalize1(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void Finalize2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsSetup(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsHookNeighbors(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsHookLinks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsMultiJump(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveMergeSlices(gputpcgmmergertypes::GPUResolveSharedMemory& smem, int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int8_t useOrigTrackParam, int8_t mergeAll);
  GPUd() void MergeLoopersInit(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void MergeLoopersSort(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void MergeLoopersMain(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);

#ifndef GPUCA_GPUCODE
  void DumpSliceTracks(std::ostream& out) const;
  void DumpMergeRanges(std::ostream& out, int32_t withinSlice, int32_t mergeMode) const;
  void DumpTrackLinks(std::ostream& out, bool output, const char* type) const;
  void DumpMergedWithinSlices(std::ostream& out) const;
  void DumpMergedBetweenSlices(std::ostream& out) const;
  void DumpCollected(std::ostream& out) const;
  void DumpMergeCE(std::ostream& out) const;
  void DumpFitPrepare(std::ostream& out) const;
  void DumpRefit(std::ostream& out) const;
  void DumpFinal(std::ostream& out) const;

  template <int32_t mergeType>
  void MergedTrackStreamerInternal(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t slice1, int32_t slice2, int32_t mergeMode, float weight, float frac) const;
  void MergedTrackStreamer(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t slice1, int32_t slice2, int32_t mergeMode, float weight, float frac) const;
  const GPUTPCGMBorderTrack& MergedTrackStreamerFindBorderTrack(const GPUTPCGMBorderTrack* tracks, int32_t N, int32_t trackId) const;
  void DebugRefitMergedTrack(const GPUTPCGMMergedTrack& track) const;
  std::vector<uint32_t> StreamerOccupancyBin(int32_t iSlice, int32_t iRow, float time) const;
  std::vector<float> StreamerUncorrectedZY(int32_t iSlice, int32_t iRow, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop) const;

  void DebugStreamerUpdate(int32_t iTrk, int32_t ihit, float xx, float yy, float zz, const GPUTPCGMMergedTrackHit& cluster, const o2::tpc::ClusterNative& clusterNative, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop, const gputpcgmmergertypes::InterpolationErrorHit& interpolation, int8_t rejectChi2, bool refit, int32_t retVal, float avgInvCharge, float posY, float posZ, int16_t clusterState, int32_t retValReject, float err2Y, float err2Z) const;
#endif

  GPUdi() int32_t SliceTrackInfoFirst(int32_t iSlice) const { return mSliceTrackInfoIndex[iSlice]; }
  GPUdi() int32_t SliceTrackInfoLast(int32_t iSlice) const { return mSliceTrackInfoIndex[iSlice + 1]; }
  GPUdi() int32_t SliceTrackInfoGlobalFirst(int32_t iSlice) const { return mSliceTrackInfoIndex[NSLICES + iSlice]; }
  GPUdi() int32_t SliceTrackInfoGlobalLast(int32_t iSlice) const { return mSliceTrackInfoIndex[NSLICES + iSlice + 1]; }
  GPUdi() int32_t SliceTrackInfoLocalTotal() const { return mSliceTrackInfoIndex[NSLICES]; }
  GPUdi() int32_t SliceTrackInfoTotal() const { return mSliceTrackInfoIndex[2 * NSLICES]; }

 private:
  GPUd() void MergeSlicesPrepareStep2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iBorder, GPUTPCGMBorderTrack** B, GPUAtomic(uint32_t) * nB, bool useOrigTrackParam = false);
  template <int32_t I>
  GPUd() void MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSlice1, GPUTPCGMBorderTrack* B1, int32_t N1, int32_t iSlice2, GPUTPCGMBorderTrack* B2, int32_t N2, int32_t mergeMode = 0);

  GPUd() void MergeCEFill(const GPUTPCGMSliceTrack* track, const GPUTPCGMMergedTrackHit& cls, const GPUTPCGMMergedTrackHitXYZ* clsXYZ, int32_t itr);

  void CheckMergedTracks();
#ifndef GPUCA_GPUCODE
  void PrintMergeGraph(const GPUTPCGMSliceTrack* trk, std::ostream& out) const;
  template <class T, class S>
  int64_t GetTrackLabelA(const S& trk) const;
  template <class S>
  int64_t GetTrackLabel(const S& trk) const;
#endif

  GPUdi() void setBlockRange(int32_t elems, int32_t nBlocks, int32_t iBlock, int32_t& start, int32_t& end);
  GPUdi() void hookEdge(int32_t u, int32_t v);

  int32_t mNextSliceInd[NSLICES];
  int32_t mPrevSliceInd[NSLICES];

  const GPUTPCSliceOutput* mkSlices[NSLICES]; //* array of input slice tracks

  int32_t* mTrackLinks;
  int32_t* mTrackCCRoots; // root of the connected component of this track

  uint32_t mNTotalSliceTracks;       // maximum number of incoming slice tracks
  uint32_t mNMaxTracks;              // maximum number of output tracks
  uint32_t mNMaxSingleSliceTracks;   // max N tracks in one slice
  uint32_t mNMaxOutputTrackClusters; // max number of clusters in output tracks (double-counting shared clusters)
  uint32_t mNMaxClusters;            // max total unique clusters (in event)
  uint32_t mNMaxLooperMatches;       // Maximum number of candidate pairs for looper matching

  uint16_t mMemoryResMemory;
  uint16_t mMemoryResOutput;
  uint16_t mMemoryResOutputState;
  uint16_t mMemoryResOutputO2;
  uint16_t mMemoryResOutputO2Clus;
  uint16_t mMemoryResOutputO2MC;
  uint16_t mMemoryResOutputO2Scratch;

  int32_t mNClusters;                   // Total number of incoming clusters (from slice tracks)
  GPUTPCGMMergedTrack* mOutputTracks;   //* array of output merged tracks
  GPUdEdxInfo* mOutputTracksdEdx;       //* dEdx information
  GPUTPCGMSliceTrack* mSliceTrackInfos; //* additional information for slice tracks
  int32_t* mSliceTrackInfoIndex;
  GPUTPCGMMergedTrackHit* mClusters;
  GPUTPCGMMergedTrackHitXYZ* mClustersXYZ;
  int32_t* mGlobalClusterIDs;
  GPUAtomic(uint32_t) * mClusterAttachment;
  o2::tpc::TrackTPC* mOutputTracksTPCO2;
  uint32_t* mOutputClusRefsTPCO2;
  o2::MCCompLabel* mOutputTracksTPCO2MC;
  MergeLooperParam* mLooperCandidates;

  uint32_t* mTrackOrderAttach;
  uint32_t* mTrackOrderProcess;
  uint8_t* mClusterStateExt;
  uint2* mClusRefTmp;
  int32_t* mTrackIDs;
  int32_t* mTmpSortMemory;
  uint32_t* mTrackSort;
  tmpSort* mTrackSortO2;
  GPUAtomic(uint32_t) * mSharedCount;     // Must be uint32_t unfortunately for atomic support
  GPUTPCGMBorderTrack* mBorderMemory;     // memory for border tracks
  GPUTPCGMBorderTrack* mBorder[2 * NSLICES];
  gputpcgmmergertypes::GPUTPCGMBorderRange* mBorderRangeMemory;    // memory for border tracks
  gputpcgmmergertypes::GPUTPCGMBorderRange* mBorderRange[NSLICES]; // memory for border tracks
  memory* mMemory;
  uint32_t* mRetryRefitIds;
  GPUTPCGMLoopData* mLoopData;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCGMMERGER_H
