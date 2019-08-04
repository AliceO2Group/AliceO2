// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTracking.h
/// \author David Rohr

#ifndef GPUCHAINTRACKING_H
#define GPUCHAINTRACKING_H

#include "GPUChain.h"
#include "GPUReconstructionHelpers.h"
#include "GPUDataTypes.h"
#include <atomic>
#include <array>

namespace o2
{
namespace trd
{
class TRDGeometryFlat;
}
} // namespace o2

namespace o2
{
namespace tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
} // namespace tpc
} // namespace o2

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
class GPUTRDTracker;
class GPUTPCGPUTracker;
class GPUDisplay;
class GPUQA;
class GPUTPCClusterStatistics;
class GPUTRDGeometry;
class TPCFastTransform;

class GPUChainTracking : public GPUChain, GPUReconstructionHelpers::helperDelegateBase
{
  friend class GPUReconstruction;

 public:
  ~GPUChainTracking() override;
  void RegisterPermanentMemoryAndProcessors() override;
  void RegisterGPUProcessors() override;
  int Init() override;
  int PrepareEvent() override;
  int Finalize() override;
  int RunChain() override;
  void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) override;

  // Structures for input and output data
  GPUTrackingInOutPointers mIOPtrs;

  struct InOutMemory {
    InOutMemory();
    ~InOutMemory();
    InOutMemory(InOutMemory&&);
    InOutMemory& operator=(InOutMemory&&);

    std::unique_ptr<GPUTPCClusterData[]> clusterData[NSLICES];
    std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSLICES];
    std::unique_ptr<o2::tpc::ClusterNative[]> clustersNative;
    std::unique_ptr<GPUTPCTrack[]> sliceOutTracks[NSLICES];
    std::unique_ptr<GPUTPCHitId[]> sliceOutClusters[NSLICES];
    std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
    std::unique_ptr<GPUTPCMCInfo[]> mcInfosTPC;
    std::unique_ptr<GPUTPCGMMergedTrack[]> mergedTracks;
    std::unique_ptr<GPUTPCGMMergedTrackHit[]> mergedTrackHits;
    std::unique_ptr<GPUTRDTrackletWord[]> trdTracklets;
    std::unique_ptr<GPUTRDTrackletLabels[]> trdTrackletsMC;
    std::unique_ptr<GPUTRDTrack[]> trdTracks;
  } mIOMem;

  // Read / Dump / Clear Data
  void ClearIOPointers();
  void AllocateIOMemory();
  using GPUChain::DumpData;
  void DumpData(const char* filename);
  using GPUChain::ReadData;
  int ReadData(const char* filename);
  void DumpSettings(const char* dir = "") override;
  void ReadSettings(const char* dir = "") override;

  // Converter / loader functions
  int ConvertNativeToClusterData();
  void ConvertNativeToClusterDataLegacy();
  void ConvertRun2RawToNative();

  // Getters for external usage of tracker classes
  GPUTRDTracker* GetTRDTracker() { return &processors()->trdTracker; }
  GPUTPCTracker* GetTPCSliceTrackers() { return processors()->tpcTrackers; }
  const GPUTPCTracker* GetTPCSliceTrackers() const { return processors()->tpcTrackers; }
  const GPUTPCGMMerger& GetTPCMerger() const { return processors()->tpcMerger; }
  GPUTPCGMMerger& GetTPCMerger() { return processors()->tpcMerger; }
  GPUDisplay* GetEventDisplay() { return mEventDisplay.get(); }
  const GPUQA* GetQA() const { return mQA.get(); }
  GPUQA* GetQA() { return mQA.get(); }

  // Processing functions
  int RunTPCTrackingSlices();
  int RunTPCTrackingMerger();
  int RunTRDTracking();
  int DoTRDGPUTracking();
  int RunTPCCompression();

  // Getters / setters for parameters
  const TPCFastTransform* GetTPCTransform() const { return mTPCFastTransform; }
  const o2::base::MatLayerCylSet* GetMatLUT() const { return mMatLUT; }
  const GPUTRDGeometry* GetTRDGeometry() const { return (GPUTRDGeometry*)mTRDGeometry; }
  const o2::tpc::ClusterNativeAccess* GetClusterNativeAccess() const { return mClusterNativeAccess.get(); }
  void SetTPCFastTransform(std::unique_ptr<TPCFastTransform>&& tpcFastTransform);
  void SetMatLUT(std::unique_ptr<o2::base::MatLayerCylSet>&& lut);
  void SetTRDGeometry(std::unique_ptr<o2::trd::TRDGeometryFlat>&& geo);
  void SetTPCFastTransform(const TPCFastTransform* tpcFastTransform) { mTPCFastTransform = tpcFastTransform; }
  void SetMatLUT(const o2::base::MatLayerCylSet* lut) { mMatLUT = lut; }
  void SetTRDGeometry(const o2::trd::TRDGeometryFlat* geo) { mTRDGeometry = geo; }
  void LoadClusterErrors();

  const void* mConfigDisplay = nullptr; // Abstract pointer to Standalone Display Configuration Structure
  const void* mConfigQA = nullptr;      // Abstract pointer to Standalone QA Configuration Structure

 protected:
  struct GPUTrackingFlatObjects : public GPUProcessor {
    GPUChainTracking* mChainTracking = nullptr;
    TPCFastTransform* mTpcTransform = nullptr;
    char* mTpcTransformBuffer = nullptr;
    o2::base::MatLayerCylSet* mMatLUT = nullptr;
    char* mMatLUTBuffer = nullptr;
    o2::trd::TRDGeometryFlat* mTrdGeometry = nullptr;
    void* SetPointersFlatObjects(void* mem);
    short mMemoryResFlat = -1;
  };

  struct eventStruct // Must consist only of void* ptr that will hold the GPU event ptrs!
  {
    void* selector[NSLICES];
    void* stream[GPUCA_MAX_STREAMS];
    void* init;
    void* constructor;
  };

  GPUChainTracking(GPUReconstruction* rec, unsigned int maxTPCHits = GPUCA_MAX_CLUSTERS, unsigned int maxTRDTracklets = GPUCA_MAX_TRD_TRACKLETS);

  int ReadEvent(int iSlice, int threadId);
  void WriteOutput(int iSlice, int threadId);
  int GlobalTracking(int iSlice, int threadId);

  int PrepareProfile();
  int DoProfile();
  void PrintMemoryRelations();
  void PrintMemoryStatistics() override;

  bool ValidateSteps();

  // Pointers to tracker classes
  GPUTrackingFlatObjects mFlatObjectsShadow; // Host copy of flat objects that will be used on the GPU
  GPUTrackingFlatObjects mFlatObjectsDevice; // flat objects that will be used on the GPU

  // Display / QA
  std::unique_ptr<GPUDisplay> mEventDisplay;
  bool mDisplayRunning = false;
  std::unique_ptr<GPUQA> mQA;
  std::unique_ptr<GPUTPCClusterStatistics> mCompressionStatistics;
  bool mQAInitialized = false;

  // Ptr to reconstruction detector objects
  std::unique_ptr<o2::tpc::ClusterNativeAccess> mClusterNativeAccess; // Internal memory for clusterNativeAccess
  std::unique_ptr<TPCFastTransform> mTPCFastTransformU;               // Global TPC fast transformation object
  const TPCFastTransform* mTPCFastTransform = nullptr;                //
  std::unique_ptr<o2::base::MatLayerCylSet> mMatLUTU;                 // Material Lookup Table
  const o2::base::MatLayerCylSet* mMatLUT = nullptr;                  //
  std::unique_ptr<o2::trd::TRDGeometryFlat> mTRDGeometryU;            // TRD Geometry
  const o2::trd::TRDGeometryFlat* mTRDGeometry = nullptr;             //

  // Upper bounds for memory allocation
  unsigned int mMaxTPCHits;
  unsigned int mMaxTRDTracklets;

  // Timers and debug
  HighResTimer timerTPCtracking[NSLICES][10];
  std::ofstream mDebugFile;

  // Synchronization and Locks
  eventStruct* mEvents = nullptr;
#ifdef __ROOT__ // ROOT5 BUG: cint doesn't do volatile
#define volatile
#endif
  volatile int mSliceOutputReady = 0;
  volatile char mSliceLeftGlobalReady[NSLICES] = {0};
  volatile char mSliceRightGlobalReady[NSLICES] = {0};
#ifdef __ROOT__
#undef volatile
#endif
  std::array<char, NSLICES> mGlobalTrackingDone;
  std::array<char, NSLICES> mWriteOutputDone;

 private:
  int RunTPCTrackingSlices_internal();
  std::atomic_flag mLockAtomic = ATOMIC_FLAG_INIT;

  int HelperReadEvent(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par);
  int HelperOutput(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par);
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
