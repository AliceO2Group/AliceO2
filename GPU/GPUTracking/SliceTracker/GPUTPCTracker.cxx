// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTracker.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCTracker.h"
#include "GPUTPCRow.h"
#include "GPUTPCTrack.h"
#include "GPUCommonMath.h"

#include "GPUTPCClusterData.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCTrackParam.h"
#include "GPUParam.inc"
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "GPUTPCConvertImpl.h"
#endif

#if !defined(GPUCA_GPUCODE)
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "GPUReconstruction.h"
#include "GPUMemorySizeScalers.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

#if !defined(GPUCA_GPUCODE)

GPUTPCTracker::GPUTPCTracker()
  : GPUProcessor(), mLinkTmpMemory(nullptr), mISlice(-1), mData(), mNMaxStartHits(0), mNMaxRowStartHits(0), mNMaxTracklets(0), mNMaxRowHits(0), mNMaxTracks(0), mNMaxTrackHits(0), mMemoryResLinks(-1), mMemoryResScratchHost(-1), mMemoryResCommon(-1), mMemoryResTracklets(-1), mMemoryResOutput(-1), mMemoryResSliceScratch(-1), mMemoryResSliceInput(-1), mRowStartHitCountOffset(nullptr), mTrackletTmpStartHits(nullptr), mGPUTrackletTemp(nullptr), mGPUParametersConst(), mCommonMem(nullptr), mTrackletStartHits(nullptr), mTracklets(nullptr), mTrackletRowHits(nullptr), mTracks(nullptr), mTrackHits(nullptr), mOutput(nullptr), mOutputMemory(nullptr)
{
}

GPUTPCTracker::~GPUTPCTracker()
{
  if (mOutputMemory) {
    free(mOutputMemory);
  }
}

// ----------------------------------------------------------------------------------
void GPUTPCTracker::SetSlice(int iSlice) { mISlice = iSlice; }
void GPUTPCTracker::InitializeProcessor()
{
  if (mISlice < 0) {
    throw std::runtime_error("Slice not set");
  }
  InitializeRows(&Param());
  SetupCommonMemory();
}

bool GPUTPCTracker::SliceDataOnGPU()
{
  return (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) && (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) && mRec->GetParam().rec.mergerReadFromTrackerDirectly && (mRec->GetConstantMem().ioPtrs.clustersNative || mRec->GetConstantMem().ioPtrs.tpcZS || mRec->GetConstantMem().ioPtrs.tpcPackedDigits);
}

void* GPUTPCTracker::SetPointersDataInput(void* mem) { return mData.SetPointersInput(mem, mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging, SliceDataOnGPU()); }
void* GPUTPCTracker::SetPointersDataLinks(void* mem) { return mData.SetPointersLinks(mem); }
void* GPUTPCTracker::SetPointersDataWeights(void* mem) { return mData.SetPointersWeights(mem); }
void* GPUTPCTracker::SetPointersDataScratch(void* mem) { return mData.SetPointersScratch(mem, mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging, SliceDataOnGPU()); }
void* GPUTPCTracker::SetPointersDataRows(void* mem) { return mData.SetPointersRows(mem); }

void* GPUTPCTracker::SetPointersScratch(void* mem)
{
  computePointerWithAlignment(mem, mTrackletStartHits, mNMaxStartHits);
  if (mRec->GetProcessingSettings().memoryAllocationStrategy != GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mem = SetPointersTracklets(mem);
  }
  if (mRec->IsGPU()) {
    computePointerWithAlignment(mem, mTrackletTmpStartHits, GPUCA_ROW_COUNT * mNMaxRowStartHits);
    computePointerWithAlignment(mem, mRowStartHitCountOffset, GPUCA_ROW_COUNT);
  }
  return mem;
}

void* GPUTPCTracker::SetPointersScratchHost(void* mem)
{
  if (mRec->GetProcessingSettings().keepDisplayMemory) {
    computePointerWithAlignment(mem, mLinkTmpMemory, mRec->Res(mMemoryResLinks).Size());
  }
  mem = mData.SetPointersClusterIds(mem, mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging);
  return mem;
}

void* GPUTPCTracker::SetPointersCommon(void* mem)
{
  computePointerWithAlignment(mem, mCommonMem, 1);
  return mem;
}

void GPUTPCTracker::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  bool reuseCondition = !mRec->GetProcessingSettings().keepDisplayMemory && mRec->GetProcessingSettings().trackletSelectorInPipeline && ((mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || mRec->GetProcessingSettings().ompKernels == 1 || mRec->GetProcessingSettings().ompThreads == 1);
  GPUMemoryReuse reLinks{reuseCondition, GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::TrackerDataLinks, (unsigned short)(mISlice % mRec->GetProcessingSettings().nStreams)};
  mMemoryResLinks = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataLinks, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCSliceLinks", reLinks);
  mMemoryResSliceScratch = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK | GPUMemoryResource::MEMORY_CUSTOM, "TPCSliceScratch");
  mMemoryResSliceInput = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataInput, GPUMemoryResource::MEMORY_INPUT | GPUMemoryResource::MEMORY_STACK | GPUMemoryResource::MEMORY_CUSTOM, "TPCSliceInput");
  GPUMemoryReuse reWeights{reuseCondition, GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::TrackerDataWeights, (unsigned short)(mISlice % mRec->GetProcessingSettings().nStreams)};
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataWeights, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCSliceWeights", reWeights);
  GPUMemoryReuse reScratch{reuseCondition, GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::TrackerScratch, (unsigned short)(mISlice % mRec->GetProcessingSettings().nStreams)};
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersScratch, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK, "TPCTrackerScratch", reScratch);
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersScratchHost, GPUMemoryResource::MEMORY_SCRATCH_HOST, "TPCTrackerHost");
  mMemoryResCommon = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersCommon, GPUMemoryResource::MEMORY_PERMANENT, "TPCTrackerCommon");
  mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersDataRows, GPUMemoryResource::MEMORY_PERMANENT, "TPCSliceRows");

  unsigned int type = mRec->GetProcessingSettings().fullMergerOnGPU ? GPUMemoryResource::MEMORY_SCRATCH : GPUMemoryResource::MEMORY_OUTPUT;
  if (mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) { // For individual scheme, we allocate tracklets separately, and change the type for the following allocations to custom
    type |= GPUMemoryResource::MEMORY_CUSTOM;
    mMemoryResTracklets = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersTracklets, type, "TPCTrackerTracklets");
  }
  mMemoryResOutput = mRec->RegisterMemoryAllocation(this, &GPUTPCTracker::SetPointersOutput, type, "TPCTrackerTracks");
}

GPUhd() void* GPUTPCTracker::SetPointersTracklets(void* mem)
{
  computePointerWithAlignment(mem, mTracklets, mNMaxTracklets);
  computePointerWithAlignment(mem, mTrackletRowHits, mNMaxRowHits);
  return mem;
}

GPUhd() void* GPUTPCTracker::SetPointersOutput(void* mem)
{
  computePointerWithAlignment(mem, mTracks, mNMaxTracks);
  computePointerWithAlignment(mem, mTrackHits, mNMaxTrackHits);
  return mem;
}

void GPUTPCTracker::SetMaxData(const GPUTrackingInOutPointers& io)
{
  if (mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mNMaxStartHits = mData.NumberOfHits();
  } else {
    mNMaxStartHits = mRec->MemoryScalers()->NTPCStartHits(mData.NumberOfHits());
  }
  mNMaxRowStartHits = mRec->MemoryScalers()->NTPCRowStartHits(mData.NumberOfHits());
  mNMaxTracklets = mRec->MemoryScalers()->NTPCTracklets(mData.NumberOfHits());
  mNMaxRowHits = mRec->MemoryScalers()->NTPCTrackletHits(mData.NumberOfHits());
  mNMaxTracks = mRec->MemoryScalers()->NTPCSectorTracks(mData.NumberOfHits());
  mNMaxTrackHits = mRec->MemoryScalers()->NTPCSectorTrackHits(mData.NumberOfHits());
#ifdef GPUCA_SORT_STARTHITS_GPU
  if (mRec->IsGPU()) {
    if (mNMaxStartHits > mNMaxRowStartHits * GPUCA_ROW_COUNT) {
      mNMaxStartHits = mNMaxRowStartHits * GPUCA_ROW_COUNT;
    }
  }
#endif
  mData.SetMaxData();
}

void GPUTPCTracker::UpdateMaxData()
{
  mNMaxTracklets = mCommonMem->nStartHits;
  mNMaxTracks = mNMaxTracklets * 2 + 50;
  mNMaxRowHits = mNMaxTracklets * GPUCA_ROW_COUNT;
}

void GPUTPCTracker::SetupCommonMemory() { new (mCommonMem) commonMemoryStruct; }

GPUh() int GPUTPCTracker::CheckEmptySlice()
{
  // Check if the Slice is empty, if so set the output apropriate and tell the reconstuct procesdure to terminate
  if (NHitsTotal() < 1) {
    mCommonMem->nTracks = mCommonMem->nTrackHits = 0;
    if (mOutput) {
      WriteOutputPrepare();
      mOutput->SetNTracks(0);
      mOutput->SetNTrackClusters(0);
    }
    return 1;
  }
  return 0;
}

GPUh() void GPUTPCTracker::WriteOutputPrepare() { GPUTPCSliceOutput::Allocate(mOutput, mCommonMem->nTracks, mCommonMem->nTrackHits, &mRec->OutputControl(), mOutputMemory); }

template <class T>
static inline bool SortComparison(const T& a, const T& b)
{
  return (a.fSortVal < b.fSortVal);
}

GPUh() void GPUTPCTracker::WriteOutput()
{
  mOutput->SetNTracks(0);
  mOutput->SetNLocalTracks(0);
  mOutput->SetNTrackClusters(0);

  if (mCommonMem->nTracks == 0) {
    return;
  }
  if (mCommonMem->nTracks > GPUCA_MAX_SLICE_NTRACK) {
    GPUError("Maximum number of tracks exceeded, cannot store");
    return;
  }

  int nStoredHits = 0;
  int nStoredTracks = 0;
  int nStoredLocalTracks = 0;

  GPUTPCTrack* out = mOutput->FirstTrack();

  trackSortData* trackOrder = new trackSortData[mCommonMem->nTracks];
  for (unsigned int i = 0; i < mCommonMem->nTracks; i++) {
    trackOrder[i].fTtrack = i;
    trackOrder[i].fSortVal = mTracks[trackOrder[i].fTtrack].NHits() / 1000.f + mTracks[trackOrder[i].fTtrack].Param().GetZ() * 100.f + mTracks[trackOrder[i].fTtrack].Param().GetY();
  }
  std::sort(trackOrder, trackOrder + mCommonMem->nLocalTracks, SortComparison<trackSortData>); // TODO: Check why this sorting affects the merging efficiency!
  std::sort(trackOrder + mCommonMem->nLocalTracks, trackOrder + mCommonMem->nTracks, SortComparison<trackSortData>);

  for (unsigned int iTrTmp = 0; iTrTmp < mCommonMem->nTracks; iTrTmp++) {
    const int iTr = trackOrder[iTrTmp].fTtrack;
    GPUTPCTrack& iTrack = mTracks[iTr];

    *out = iTrack;
    int nClu = 0;
    int iID = iTrack.FirstHitID();

    for (int ith = 0; ith < iTrack.NHits(); ith++) {
      const GPUTPCHitId& ic = mTrackHits[iID + ith];
      int iRow = ic.RowIndex();
      int ih = ic.HitIndex();

      const GPUTPCRow& row = mData.Row(iRow);
      int clusterIndex = mData.ClusterDataIndex(row, ih);
#ifdef GPUCA_ARRAY_BOUNDS_CHECKS
      if (ih >= row.NHits() || ih < 0) {
        GPUError("Array out of bounds access (Sector Row) (Hit %d / %d - NumC %d): Sector %d Row %d Index %d", ith, iTrack.NHits(), NHitsTotal(), mISlice, iRow, ih);
        fflush(stdout);
        continue;
      }
      if (clusterIndex >= NHitsTotal() || clusterIndex < 0) {
        GPUError("Array out of bounds access (Cluster Data) (Hit %d / %d - NumC %d): Sector %d Row %d Hit %d, Clusterdata Index %d", ith, iTrack.NHits(), NHitsTotal(), mISlice, iRow, ih, clusterIndex);
        fflush(stdout);
        continue;
      }
#endif

      float origX, origY, origZ;
      unsigned char flags;
      unsigned short amp;
      int id;
      if (Param().par.earlyTpcTransform) {
        origX = mData.ClusterData()[clusterIndex].x;
        origY = mData.ClusterData()[clusterIndex].y;
        origZ = mData.ClusterData()[clusterIndex].z;
        flags = mData.ClusterData()[clusterIndex].flags;
        amp = mData.ClusterData()[clusterIndex].amp;
        id = mData.ClusterData()[clusterIndex].id;
      } else {
        const ClusterNativeAccess& cls = *mConstantMem->ioPtrs.clustersNative;
        id = clusterIndex + cls.clusterOffset[mISlice][0];
        GPUTPCConvertImpl::convert(*mConstantMem, mISlice, iRow, cls.clustersLinear[id].getPad(), cls.clustersLinear[id].getTime(), origX, origY, origZ);
        flags = cls.clustersLinear[id].getFlags();
        amp = cls.clustersLinear[id].qTot;
      }
      GPUTPCSliceOutCluster c;
      c.Set(id, iRow, flags, amp, origX, origY, origZ);
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
      c.mPad = mData.ClusterData()[clusterIndex].pad;
      c.mTime = mData.ClusterData()[clusterIndex].time;
#endif
      out->SetOutTrackCluster(nClu, c);
      nClu++;
    }

    nStoredTracks++;
    if (iTr < mCommonMem->nLocalTracks) {
      nStoredLocalTracks++;
    }
    nStoredHits += nClu;
    out->SetNHits(nClu);
    out = out->NextTrack();
  }
  delete[] trackOrder;

  mOutput->SetNTracks(nStoredTracks);
  mOutput->SetNLocalTracks(nStoredLocalTracks);
  mOutput->SetNTrackClusters(nStoredHits);
  if (Param().par.debugLevel >= 3) {
    GPUInfo("Slice %d, Output: Tracks %d, local tracks %d, hits %d", mISlice, nStoredTracks, nStoredLocalTracks, nStoredHits);
  }
}

#endif
