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
#include "GPUTPCGMSectorTrack.h"
#include "GPUCommonDef.h"
#include "GPUProcessor.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUGeneralKernels.h"

#if !defined(GPUCA_GPUCODE)
#include <cmath>
#include <iostream>
#endif // GPUCA_GPUCODE

namespace o2::base
{
class MatLayerCylSet;
}
namespace o2::tpc
{
struct ClusterNative;
}

namespace o2::gpu
{
class GPUTPCSectorTrack;
class GPUTPCGMTrackParam;
class GPUTPCTracker;
class GPUChainTracking;
class GPUTPCGMPolynomialField;
struct GPUTPCGMLoopData;
namespace internal
{
struct MergeLooperParam;
} // namespace internal

/**
 * @class GPUTPCGMMerger
 *
 */
class GPUTPCGMMerger : public GPUProcessor
{
 public:
  GPUTPCGMMerger();
  ~GPUTPCGMMerger() = default;
  GPUTPCGMMerger(const GPUTPCGMMerger&) = delete;
  const GPUTPCGMMerger& operator=(const GPUTPCGMMerger&) const = delete;
  static constexpr const int32_t NSECTORS = GPUCA_NSECTORS; //* N sectors

  struct memory {
    GPUAtomic(uint32_t) nRetryRefit;
    GPUAtomic(uint32_t) nLoopData;
    GPUAtomic(uint32_t) nUnpackedTracks;
    GPUAtomic(uint32_t) nMergedTracks;
    GPUAtomic(uint32_t) nMergedTrackClusters;
    GPUAtomic(uint32_t) nO2Tracks;
    GPUAtomic(uint32_t) nO2ClusRefs;
    const GPUTPCTrack* firstExtrapolatedTracks[NSECTORS];
    GPUAtomic(uint32_t) tmpCounter[2 * NSECTORS];
    GPUAtomic(uint32_t) nLooperMatchCandidates;
  };

  struct trackCluster {
    uint32_t id;
    uint8_t row;
    uint8_t sector;
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
  void* SetPointersOutput(void* mem);
  void* SetPointersOutputO2(void* mem);
  void* SetPointersOutputO2Clus(void* mem);
  void* SetPointersOutputO2MC(void* mem);
  void* SetPointersOutputO2Scratch(void* mem);
  void* SetPointersOutputState(void* mem);
  void* SetPointersMemory(void* mem);

  GPUhdi() int32_t NMergedTracks() const { return mMemory->nMergedTracks; }
  GPUhdi() const GPUTPCGMMergedTrack* MergedTracks() const { return mMergedTracks; }
  GPUhdi() GPUTPCGMMergedTrack* MergedTracks() { return mMergedTracks; }
  GPUhdi() const GPUdEdxInfo* MergedTracksdEdx() const { return mMergedTracksdEdx; }
  GPUhdi() GPUdEdxInfo* MergedTracksdEdx() { return mMergedTracksdEdx; }
  GPUhdi() const GPUdEdxInfo* MergedTracksdEdxAlt() const { return mMergedTracksdEdxAlt; }
  GPUhdi() GPUdEdxInfo* MergedTracksdEdxAlt() { return mMergedTracksdEdxAlt; }
  GPUhdi() uint32_t NClusters() const { return mNClusters; }
  GPUhdi() uint32_t NMaxClusters() const { return mNMaxClusters; }
  GPUhdi() uint32_t NMaxTracks() const { return mNMaxTracks; }
  GPUhdi() uint32_t NMaxMergedTrackClusters() const { return mNMaxMergedTrackClusters; }
  GPUhdi() uint32_t NMergedTrackClusters() const { return mMemory->nMergedTrackClusters; }
  GPUhdi() const GPUTPCGMMergedTrackHit* Clusters() const { return mClusters; }
  GPUhdi() GPUTPCGMMergedTrackHit* Clusters() { return (mClusters); }
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
  GPUhdi() internal::MergeLooperParam* LooperCandidates() { return mLooperCandidates; }
  GPUhdi() GPUAtomic(uint32_t) * SharedCount() { return mSharedCount; }
  GPUhdi() gputpcgmmergertypes::GPUTPCGMBorderRange* BorderRange(int32_t i) { return mBorderRange[i]; }
  GPUhdi() const gputpcgmmergertypes::GPUTPCGMBorderRange* BorderRange(int32_t i) const { return mBorderRange[i]; }
  GPUhdi() GPUTPCGMBorderTrack* BorderTracks(int32_t i) { return mBorder[i]; }
  GPUhdi() o2::tpc::TrackTPC* OutputTracksTPCO2() { return mOutputTracksTPCO2; }
  GPUhdi() uint32_t* OutputClusRefsTPCO2() { return mOutputClusRefsTPCO2; }
  GPUhdi() o2::MCCompLabel* OutputTracksTPCO2MC() { return mOutputTracksTPCO2MC; }
  GPUhdi() uint32_t NOutputTracksTPCO2() const { return mMemory->nO2Tracks; }
  GPUhdi() uint32_t NOutputClusRefsTPCO2() const { return mMemory->nO2ClusRefs; }
  GPUhdi() GPUTPCGMSectorTrack* SectorTrackInfos() { return mSectorTrackInfos; }
  GPUhdi() int32_t NMaxSingleSectorTracks() const { return mNMaxSingleSectorTracks; }
  GPUhdi() int32_t* TrackIDs() { return mTrackIDs; }
  GPUhdi() int32_t* TmpSortMemory() { return mTmpSortMemory; }

  GPUd() uint16_t MemoryResMemory() { return mMemoryResMemory; }
  GPUd() uint16_t MemoryResOutput() const { return mMemoryResOutput; }
  GPUd() uint16_t MemoryResOutputState() const { return mMemoryResOutputState; }
  GPUd() uint16_t MemoryResOutputO2() const { return mMemoryResOutputO2; }
  GPUd() uint16_t MemoryResOutputO2Clus() const { return mMemoryResOutputO2Clus; }
  GPUd() uint16_t MemoryResOutputO2MC() const { return mMemoryResOutputO2MC; }
  GPUd() uint16_t MemoryResOutputO2Scratch() const { return mMemoryResOutputO2Scratch; }

  GPUd() int32_t RefitSectorTrack(GPUTPCGMSectorTrack& sectorTrack, const GPUTPCTrack* inTrack, float alpha, int32_t sector);
  GPUd() void SetTrackClusterT(GPUTPCGMSectorTrack& track, int32_t iSector, const GPUTPCTrack* sectorTr);

  int32_t CheckSectors();
  GPUd() void RefitSectorTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector);
  GPUd() void UnpackSectorGlobal(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector);
  GPUd() void UnpackSaveNumber(int32_t id);
  GPUd() void UnpackResetIds(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector);
  GPUd() void MergeCE(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ClearTrackLinks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, bool output);
  GPUd() void MergeWithinSectorsPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void MergeSectorsPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t border0, int32_t border1, int8_t useOrigTrackParam);
  template <int32_t I>
  GPUd() void MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector, int8_t withinSector, int8_t mergeMode);
  GPUd() void MergeBorderTracksSetup(int32_t& n1, int32_t& n2, GPUTPCGMBorderTrack*& b1, GPUTPCGMBorderTrack*& b2, int32_t& jSector, int32_t iSector, int8_t withinSector, int8_t mergeMode) const;
  template <int32_t I>
  GPUd() void MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, gputpcgmmergertypes::GPUTPCGMBorderRange* range, int32_t N, int32_t cmpMax);
  GPUd() void SortTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void SortTracksQPt(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void SortTracksPrepare(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void PrepareForFit0(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void PrepareForFit1(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void PrepareForFit2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void LinkExtrapolatedTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void CollectMergedTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void Finalize0(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void Finalize1(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void Finalize2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsSetup(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsHookNeighbors(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsHookLinks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveFindConnectedComponentsMultiJump(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void ResolveMergeSectors(gputpcgmmergertypes::GPUResolveSharedMemory& smem, int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int8_t useOrigTrackParam, int8_t mergeAll);
  GPUd() void MergeLoopersInit(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void MergeLoopersSort(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);
  GPUd() void MergeLoopersMain(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread);

#ifndef GPUCA_GPUCODE
  void DumpSectorTracks(std::ostream& out) const;
  void DumpMergeRanges(std::ostream& out, int32_t withinSector, int32_t mergeMode) const;
  void DumpTrackLinks(std::ostream& out, bool output, const char* type) const;
  void DumpMergedWithinSectors(std::ostream& out) const;
  void DumpMergedBetweenSectors(std::ostream& out) const;
  void DumpCollected(std::ostream& out) const;
  void DumpMergeCE(std::ostream& out) const;
  void DumpFitPrepare(std::ostream& out) const;
  void DumpRefit(std::ostream& out) const;
  void DumpFinal(std::ostream& out) const;
  void DumpLoopers(std::ostream& out) const;
  void DumpTrackParam(std::ostream& out) const;
  void DumpTrackClusters(std::ostream& out, bool non0StateOnly = false, bool noNDF0 = false) const;

  template <int32_t mergeType>
  void MergedTrackStreamerInternal(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t sector1, int32_t sector2, int32_t mergeMode, float weight, float frac) const;
  void MergedTrackStreamer(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t sector1, int32_t sector2, int32_t mergeMode, float weight, float frac) const;
  const GPUTPCGMBorderTrack& MergedTrackStreamerFindBorderTrack(const GPUTPCGMBorderTrack* tracks, int32_t N, int32_t trackId) const;
  void DebugRefitMergedTrack(const GPUTPCGMMergedTrack& track) const;
  std::vector<uint32_t> StreamerOccupancyBin(int32_t iSector, int32_t iRow, float time) const;
  std::vector<float> StreamerUncorrectedZY(int32_t iSector, int32_t iRow, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop) const;

  void DebugStreamerUpdate(int32_t iTrk, int32_t ihit, float xx, float yy, float zz, const GPUTPCGMMergedTrackHit& cluster, const o2::tpc::ClusterNative& clusterNative, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop, const gputpcgmmergertypes::InterpolationErrorHit& interpolation, int8_t rejectChi2, bool refit, int32_t retVal, float avgInvCharge, float posY, float posZ, int16_t clusterState, int32_t retValReject, float err2Y, float err2Z) const;
#endif

  GPUdi() int32_t SectorTrackInfoFirst(int32_t iSector) const { return mSectorTrackInfoIndex[iSector]; }
  GPUdi() int32_t SectorTrackInfoLast(int32_t iSector) const { return mSectorTrackInfoIndex[iSector + 1]; }
  GPUdi() int32_t SectorTrackInfoGlobalFirst(int32_t iSector) const { return mSectorTrackInfoIndex[NSECTORS + iSector]; }
  GPUdi() int32_t SectorTrackInfoGlobalLast(int32_t iSector) const { return mSectorTrackInfoIndex[NSECTORS + iSector + 1]; }
  GPUdi() int32_t SectorTrackInfoLocalTotal() const { return mSectorTrackInfoIndex[NSECTORS]; }
  GPUdi() int32_t SectorTrackInfoTotal() const { return mSectorTrackInfoIndex[2 * NSECTORS]; }

  void CheckMergeGraph();
  void CheckCollectedTracks();

 private:
  GPUd() void MergeSectorsPrepareStep2(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iBorder, GPUTPCGMBorderTrack** B, GPUAtomic(uint32_t) * nB, bool useOrigTrackParam = false);
  template <int32_t I>
  GPUd() void MergeBorderTracks(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t iSector1, const GPUTPCGMBorderTrack* B1, int32_t N1, int32_t iSector2, const GPUTPCGMBorderTrack* B2, int32_t N2, int32_t mergeMode = 0);

  GPUd() void MergeCEFill(const GPUTPCGMSectorTrack* track, const GPUTPCGMMergedTrackHit& cls, int32_t itr);

#ifndef GPUCA_GPUCODE
  void PrintMergeGraph(const GPUTPCGMSectorTrack* trk, std::ostream& out) const;
  template <class T, class S>
  int64_t GetTrackLabelA(const S& trk) const;
  template <class S>
  int64_t GetTrackLabel(const S& trk) const;
#endif

  GPUdi() void setBlockRange(int32_t elems, int32_t nBlocks, int32_t iBlock, int32_t& start, int32_t& end);
  GPUdi() void hookEdge(int32_t u, int32_t v);

  int32_t mNextSectorInd[NSECTORS];
  int32_t mPrevSectorInd[NSECTORS];

  int32_t* mTrackLinks = nullptr;
  int32_t* mTrackCCRoots; // root of the connected component of this track

  uint32_t mNTotalSectorTracks = 0;      // maximum number of incoming sector tracks
  uint32_t mNMaxTracks = 0;              // maximum number of output tracks
  uint32_t mNMaxSingleSectorTracks = 0;  // max N tracks in one sector
  uint32_t mNMaxMergedTrackClusters = 0; // max number of clusters in output tracks (double-counting shared clusters)
  uint32_t mNMaxClusters = 0;            // max total unique clusters (in event)
  uint32_t mNMaxLooperMatches = 0;       // Maximum number of candidate pairs for looper matching

  uint16_t mMemoryResMemory = (uint16_t)-1;
  uint16_t mMemoryResOutput = (uint16_t)-1;
  uint16_t mMemoryResOutputState = (uint16_t)-1;
  uint16_t mMemoryResOutputO2 = (uint16_t)-1;
  uint16_t mMemoryResOutputO2Clus = (uint16_t)-1;
  uint16_t mMemoryResOutputO2MC = (uint16_t)-1;
  uint16_t mMemoryResOutputO2Scratch = (uint16_t)-1;

  int32_t mNClusters = 0;                           // Total number of incoming clusters (from sector tracks)
  GPUTPCGMMergedTrack* mMergedTracks = nullptr;     //* array of output merged tracks
  GPUdEdxInfo* mMergedTracksdEdx = nullptr;         //* dEdx information
  GPUdEdxInfo* mMergedTracksdEdxAlt = nullptr;      //* dEdx alternative information
  GPUTPCGMSectorTrack* mSectorTrackInfos = nullptr; //* additional information for sector tracks
  int32_t* mSectorTrackInfoIndex = nullptr;
  GPUTPCGMMergedTrackHit* mClusters = nullptr;
  GPUAtomic(uint32_t) * mClusterAttachment = nullptr;
  o2::tpc::TrackTPC* mOutputTracksTPCO2 = nullptr;
  uint32_t* mOutputClusRefsTPCO2 = nullptr;
  o2::MCCompLabel* mOutputTracksTPCO2MC = nullptr;
  internal::MergeLooperParam* mLooperCandidates = nullptr;

  uint32_t* mTrackOrderAttach = nullptr;
  uint32_t* mTrackOrderProcess = nullptr;
  uint8_t* mClusterStateExt = nullptr;
  uint2* mClusRefTmp = nullptr;
  int32_t* mTrackIDs = nullptr;
  int32_t* mTmpSortMemory = nullptr;
  uint32_t* mTrackSort = nullptr;
  tmpSort* mTrackSortO2 = nullptr;
  GPUAtomic(uint32_t) * mSharedCount = nullptr; // Must be uint32_t unfortunately for atomic support
  GPUTPCGMBorderTrack* mBorderMemory = nullptr; // memory for border tracks
  GPUTPCGMBorderTrack* mBorder[2 * NSECTORS];
  gputpcgmmergertypes::GPUTPCGMBorderRange* mBorderRangeMemory = nullptr; // memory for border tracks
  gputpcgmmergertypes::GPUTPCGMBorderRange* mBorderRange[NSECTORS];       // memory for border tracks
  memory* mMemory = nullptr;
  uint32_t* mRetryRefitIds = nullptr;
  GPUTPCGMLoopData* mLoopData = nullptr;
};
} // namespace o2::gpu

#endif // GPUTPCGMMERGER_H
