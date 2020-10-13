// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  static CONSTEXPR int NSLICES = GPUCA_NSLICES; //* N slices

  struct memory {
    GPUAtomic(unsigned int) nRetryRefit;
    GPUAtomic(unsigned int) nLoopData;
    GPUAtomic(unsigned int) nUnpackedTracks;
    GPUAtomic(unsigned int) nOutputTracks;
    GPUAtomic(unsigned int) nOutputTrackClusters;
    const GPUTPCTrack* firstGlobalTracks[NSLICES];
    GPUAtomic(unsigned int) tmpCounter[2 * NSLICES];
  };

  struct trackCluster {
    unsigned int id;
    unsigned char row;
    unsigned char slice;
    unsigned char leg;
  };

  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);
  void* SetPointersMerger(void* mem);
  void* SetPointersRefitScratch(void* mem);
  void* SetPointersRefitScratch2(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersMemory(void* mem);

  void SetSliceData(int index, const GPUTPCSliceOutput* sliceData) { mkSlices[index] = sliceData; }

  GPUhd() int NOutputTracks() const { return mMemory->nOutputTracks; }
  GPUhd() const GPUTPCGMMergedTrack* OutputTracks() const { return mOutputTracks; }
  GPUhd() GPUTPCGMMergedTrack* OutputTracks()
  {
    return mOutputTracks;
  }

  GPUhd() unsigned int NClusters() const { return mNClusters; }
  GPUhd() unsigned int NMaxClusters() const { return mNMaxClusters; }
  GPUhd() unsigned int NMaxTracks() const { return mNMaxTracks; }
  GPUhd() unsigned int NMaxOutputTrackClusters() const { return mNMaxOutputTrackClusters; }
  GPUhd() unsigned int NOutputTrackClusters() const { return mMemory->nOutputTrackClusters; }
  GPUhd() const GPUTPCGMMergedTrackHit* Clusters() const { return mClusters; }
  GPUhd() GPUTPCGMMergedTrackHit* Clusters() { return (mClusters); }
  GPUhd() const GPUTPCGMMergedTrackHitXYZ* ClustersXYZ() const { return mClustersXYZ; }
  GPUhd() GPUTPCGMMergedTrackHitXYZ* ClustersXYZ() { return (mClustersXYZ); }
  GPUhdi() GPUAtomic(unsigned int) * ClusterAttachment() const { return mClusterAttachment; }
  GPUhdi() unsigned int* TrackOrderAttach() const { return mTrackOrderAttach; }
  GPUhdi() unsigned int* TrackOrderProcess() const { return mTrackOrderProcess; }
  GPUhdi() unsigned int* RetryRefitIds() const { return mRetryRefitIds; }
  GPUhdi() unsigned char* ClusterStateExt() const { return mClusterStateExt; }
  GPUhdi() GPUTPCGMLoopData* LoopData() const { return mLoopData; }
  GPUhdi() memory* Memory() const { return mMemory; }
  GPUhdi() GPUAtomic(unsigned int) * TmpCounter() { return mMemory->tmpCounter; }
  GPUhdi() uint4* TmpMem() { return mTmpMem; }
  GPUhdi() gputpcgmmergertypes::GPUTPCGMBorderRange* BorderRange(int i) { return mBorderRange[i]; }

  GPUd() unsigned short MemoryResMemory() { return mMemoryResMemory; }
  GPUd() unsigned short MemoryResOutput() const { return mMemoryResOutput; }

  GPUd() int RefitSliceTrack(GPUTPCGMSliceTrack& sliceTrack, const GPUTPCTrack* inTrack, float alpha, int slice);
  GPUd() void SetTrackClusterZT(GPUTPCGMSliceTrack& track, int iSlice, const GPUTPCTrack* sliceTr);

  int CheckSlices();
  GPUd() void RefitSliceTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice);
  GPUd() void UnpackSliceGlobal(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice);
  GPUd() void UnpackSaveNumber(int id);
  GPUd() void UnpackResetIds(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice);
  GPUd() void MergeCE(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void ClearTrackLinks(int nBlocks, int nThreads, int iBlock, int iThread, bool nOutput);
  GPUd() void MergeWithinSlicesPrepare(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void MergeSlicesPrepare(int nBlocks, int nThreads, int iBlock, int iThread, int border0, int border1, char useOrigTrackParam);
  template <int I>
  GPUd() void MergeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice, char withinSlice, char mergeMode);
  GPUd() void MergeBorderTracksSetup(int& n1, int& n2, GPUTPCGMBorderTrack*& b1, GPUTPCGMBorderTrack*& b2, int& jSlice, int iSlice, char withinSlice, char mergeMode);
  template <int I>
  GPUd() void MergeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, gputpcgmmergertypes::GPUTPCGMBorderRange* range, int N, int cmpMax);
  GPUd() void SortTracks(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void SortTracksQPt(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void SortTracksPrepare(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void PrepareClustersForFit0(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void PrepareClustersForFit1(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void PrepareClustersForFit2(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void LinkGlobalTracks(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void CollectMergedTracks(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void Finalize0(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void Finalize1(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void Finalize2(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void ResolveFindConnectedComponentsSetup(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void ResolveFindConnectedComponentsHookNeighbors(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void ResolveFindConnectedComponentsHookLinks(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void ResolveFindConnectedComponentsMultiJump(int nBlocks, int nThreads, int iBlock, int iThread);
  GPUd() void ResolveMergeSlices(gputpcgmmergertypes::GPUResolveSharedMemory& smem, int nBlocks, int nThreads, int iBlock, int iThread, char useOrigTrackParam, char mergeAll);

#ifndef GPUCA_GPUCODE
  void DumpSliceTracks(std::ostream& out);
  void DumpMergedWithinSlices(std::ostream& out);
  void DumpMergedBetweenSlices(std::ostream& out);
  void DumpCollected(std::ostream& out);
  void DumpMergeCE(std::ostream& out);
  void DumpFitPrepare(std::ostream& out);
  void DumpRefit(std::ostream& out);
  void DumpFinal(std::ostream& out);
#endif

 private:
  GPUd() void MakeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iBorder, GPUTPCGMBorderTrack** B, GPUAtomic(unsigned int) * nB, bool useOrigTrackParam = false);
  template <int I>
  GPUd() void MergeBorderTracks(int nBlocks, int nThreads, int iBlock, int iThread, int iSlice1, GPUTPCGMBorderTrack* B1, int N1, int iSlice2, GPUTPCGMBorderTrack* B2, int N2, int mergeMode = 0);

  GPUd() void MergeCEFill(const GPUTPCGMSliceTrack* track, const GPUTPCGMMergedTrackHit& cls, const GPUTPCGMMergedTrackHitXYZ* clsXYZ, int itr);

  void CheckMergedTracks();
#ifndef GPUCA_GPUCODE
  void PrintMergeGraph(const GPUTPCGMSliceTrack* trk, std::ostream& out);
  int GetTrackLabel(const GPUTPCGMBorderTrack& trk);
#endif

  GPUdi() int SliceTrackInfoFirst(int iSlice)
  {
    return mSliceTrackInfoIndex[iSlice];
  }
  GPUdi() int SliceTrackInfoLast(int iSlice) { return mSliceTrackInfoIndex[iSlice + 1]; }
  GPUdi() int SliceTrackInfoGlobalFirst(int iSlice) { return mSliceTrackInfoIndex[NSLICES + iSlice]; }
  GPUdi() int SliceTrackInfoGlobalLast(int iSlice) { return mSliceTrackInfoIndex[NSLICES + iSlice + 1]; }
  GPUdi() int SliceTrackInfoLocalTotal() { return mSliceTrackInfoIndex[NSLICES]; }
  GPUdi() int SliceTrackInfoTotal() { return mSliceTrackInfoIndex[2 * NSLICES]; }

  GPUdi() void setBlockRange(int elems, int nBlocks, int iBlock, int& start, int& end);
  GPUdi() void hookEdge(int u, int v);

  int mNextSliceInd[NSLICES];
  int mPrevSliceInd[NSLICES];

  const GPUTPCSliceOutput* mkSlices[NSLICES]; //* array of input slice tracks

  int* mTrackLinks;
  int* mTrackCCRoots; // root of the connected component of this track

  unsigned int mNMaxSliceTracks;         // maximum number of incoming slice tracks
  unsigned int mNMaxTracks;              // maximum number of output tracks
  unsigned int mNMaxSingleSliceTracks;   // max N tracks in one slice
  unsigned int mNMaxOutputTrackClusters; // max number of clusters in output tracks (double-counting shared clusters)
  unsigned int mNMaxClusters;            // max total unique clusters (in event)

  unsigned short mMemoryResMemory;
  unsigned short mMemoryResOutput;

  int mNClusters;                     // Total number of incoming clusters (from slice tracks)
  GPUTPCGMMergedTrack* mOutputTracks; //* array of output merged tracks

  GPUTPCGMSliceTrack* mSliceTrackInfos; //* additional information for slice tracks
  int* mSliceTrackInfoIndex;
  GPUTPCGMMergedTrackHit* mClusters;
  GPUTPCGMMergedTrackHitXYZ* mClustersXYZ;
  int* mGlobalClusterIDs;
  GPUAtomic(unsigned int) * mClusterAttachment;
  unsigned int* mTrackOrderAttach;
  unsigned int* mTrackOrderProcess;
  unsigned char* mClusterStateExt;
  uint4* mTmpMem;
  GPUTPCGMBorderTrack* mBorderMemory; // memory for border tracks
  GPUTPCGMBorderTrack* mBorder[2 * NSLICES];
  gputpcgmmergertypes::GPUTPCGMBorderRange* mBorderRangeMemory;    // memory for border tracks
  gputpcgmmergertypes::GPUTPCGMBorderRange* mBorderRange[NSLICES]; // memory for border tracks
  memory* mMemory;
  unsigned int* mRetryRefitIds;
  GPUTPCGMLoopData* mLoopData;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCGMMERGER_H
