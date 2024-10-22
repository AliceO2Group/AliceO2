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

/// \file GPUChainTracking.h
/// \author David Rohr

#ifndef GPUCHAINTRACKING_H
#define GPUCHAINTRACKING_H

#include "GPUChain.h"
#include "GPUReconstructionHelpers.h"
#include "GPUDataTypes.h"
#include <atomic>
#include <mutex>
#include <functional>
#include <array>
#include <vector>
#include <utility>

namespace o2
{
namespace trd
{
class GeometryFlat;
}
} // namespace o2

namespace o2
{
namespace tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
class CalibdEdxContainer;
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
//class GPUTRDTrackerGPU;
class GPUTPCGPUTracker;
class GPUDisplayInterface;
class GPUQA;
class GPUTPCClusterStatistics;
class GPUTRDGeometry;
class TPCFastTransform;
class GPUTrackingInputProvider;
struct GPUChainTrackingFinalContext;
struct GPUTPCCFChainContext;
struct GPUNewCalibValues;
struct GPUTriggerOutputs;

class GPUChainTracking : public GPUChain, GPUReconstructionHelpers::helperDelegateBase
{
  friend class GPUReconstruction;

 public:
  ~GPUChainTracking() override;
  void RegisterPermanentMemoryAndProcessors() override;
  void RegisterGPUProcessors() override;
  int32_t Init() override;
  int32_t PrepareEvent() override;
  int32_t Finalize() override;
  int32_t RunChain() override;
  void MemorySize(size_t& gpuMem, size_t& pageLockedHostMem) override;
  int32_t CheckErrorCodes(bool cpuOnly = false, bool forceShowErrors = false, std::vector<std::array<uint32_t, 4>>* fillErrors = nullptr) override;
  bool SupportsDoublePipeline() override { return true; }
  int32_t FinalizePipelinedProcessing() override;
  void ClearErrorCodes(bool cpuOnly = false);
  int32_t DoQueuedUpdates(int32_t stream, bool updateSlave = true); // Forces doing queue calib updates, don't call when you are not sure you are allowed to do so!
  bool QARanForTF() const { return mFractionalQAEnabled; }

  // Structures for input and output data
  GPUTrackingInOutPointers& mIOPtrs;

  struct InOutMemory {
    InOutMemory();
    ~InOutMemory();
    InOutMemory(InOutMemory&&);
    InOutMemory& operator=(InOutMemory&&);

    std::unique_ptr<uint64_t[]> tpcZSpages;
    std::unique_ptr<char[]> tpcZSpagesChar;        // Same as above, but as char (needed for reading dumps, but deprecated, since alignment can be wrong) // TODO: Fix alignment
    std::unique_ptr<char[]> tpcCompressedClusters; // TODO: Fix alignment
    std::unique_ptr<GPUTrackingInOutZS> tpcZSmeta;
    std::unique_ptr<GPUTrackingInOutZS::GPUTrackingInOutZSMeta> tpcZSmeta2;
    std::unique_ptr<o2::tpc::Digit[]> tpcDigits[NSLICES];
    std::unique_ptr<GPUTrackingInOutDigits> digitMap;
    std::unique_ptr<GPUTPCClusterData[]> clusterData[NSLICES];
    std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSLICES];
    std::unique_ptr<o2::tpc::ClusterNative[]> clustersNative;
    std::unique_ptr<o2::tpc::ClusterNativeAccess> clusterNativeAccess;
    std::unique_ptr<GPUTPCTrack[]> sliceTracks[NSLICES];
    std::unique_ptr<GPUTPCHitId[]> sliceClusters[NSLICES];
    std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
    std::unique_ptr<GPUTPCMCInfo[]> mcInfosTPC;
    std::unique_ptr<GPUTPCMCInfoCol[]> mcInfosTPCCol;
    std::unique_ptr<GPUTPCGMMergedTrack[]> mergedTracks;
    std::unique_ptr<GPUTPCGMMergedTrackHit[]> mergedTrackHits;
    std::unique_ptr<GPUTPCGMMergedTrackHitXYZ[]> mergedTrackHitsXYZ;
    std::unique_ptr<GPUTRDTrackletWord[]> trdTracklets;
    std::unique_ptr<GPUTRDSpacePoint[]> trdSpacePoints;
    std::unique_ptr<float[]> trdTriggerTimes;
    std::unique_ptr<int32_t[]> trdTrackletIdxFirst;
    std::unique_ptr<uint8_t[]> trdTrigRecMask;
    std::unique_ptr<GPUTRDTrackGPU[]> trdTracks;
    std::unique_ptr<char[]> clusterNativeMC;
    std::unique_ptr<o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>> clusterNativeMCView;
    std::unique_ptr<char[]> tpcDigitsMC[NSLICES];
    std::unique_ptr<o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>[]> tpcDigitMCView;
    std::unique_ptr<GPUTPCDigitsMCInput> tpcDigitMCMap;
    std::unique_ptr<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>> clusterNativeMCBuffer;
    std::unique_ptr<GPUSettingsTF[]> settingsTF;
  } mIOMem;

  // Read / Dump / Clear Data
  void ClearIOPointers();
  void AllocateIOMemory();
  using GPUChain::DumpData;
  void DumpData(const char* filename);
  using GPUChain::ReadData;
  int32_t ReadData(const char* filename);
  void DumpSettings(const char* dir = "") override;
  void ReadSettings(const char* dir = "") override;

  // Converter / loader functions
  int32_t ConvertNativeToClusterData();
  void ConvertNativeToClusterDataLegacy();
  void ConvertRun2RawToNative();
  void ConvertZSEncoder(int32_t version);
  void ConvertZSFilter(bool zs12bit);

  // Getters for external usage of tracker classes
  GPUTRDTrackerGPU* GetTRDTrackerGPU() { return &processors()->trdTrackerGPU; }
  GPUTPCTracker* GetTPCSliceTrackers() { return processors()->tpcTrackers; }
  const GPUTPCTracker* GetTPCSliceTrackers() const { return processors()->tpcTrackers; }
  const GPUTPCGMMerger& GetTPCMerger() const { return processors()->tpcMerger; }
  GPUTPCGMMerger& GetTPCMerger() { return processors()->tpcMerger; }
  GPUDisplayInterface* GetEventDisplay() { return mEventDisplay.get(); }
  const GPUQA* GetQA() const { return mQAFromForeignChain ? mQAFromForeignChain->mQA.get() : mQA.get(); }
  GPUQA* GetQA() { return mQAFromForeignChain ? mQAFromForeignChain->mQA.get() : mQA.get(); }
  int32_t ForceInitQA();
  void SetQAFromForeignChain(GPUChainTracking* chain) { mQAFromForeignChain = chain; }
  const GPUSettingsDisplay* GetEventDisplayConfig() const { return mConfigDisplay; }
  const GPUSettingsQA* GetQAConfig() const { return mConfigQA; }

  // Processing functions
  int32_t RunTPCClusterizer(bool synchronizeOutput = true);
  int32_t ForwardTPCDigits();
  int32_t RunTPCTrackingSlices();
  int32_t RunTPCTrackingMerger(bool synchronizeOutput = true);
  template <int32_t I>
  int32_t RunTRDTracking();
  template <int32_t I, class T = GPUTRDTracker>
  int32_t DoTRDGPUTracking(T* externalInstance = nullptr);
  int32_t RunTPCCompression();
  int32_t RunTPCDecompression();
  int32_t RunRefit();

  // Getters / setters for parameters
  const CorrectionMapsHelper* GetTPCTransformHelper() const { return processors()->calibObjects.fastTransformHelper; }
  const TPCPadGainCalib* GetTPCPadGainCalib() const { return processors()->calibObjects.tpcPadGain; }
  const TPCZSLinkMapping* GetTPCZSLinkMapping() const { return processors()->calibObjects.tpcZSLinkMapping; }
  const o2::tpc::CalibdEdxContainer* GetdEdxCalibContainer() const { return processors()->calibObjects.dEdxCalibContainer; }
  const o2::base::MatLayerCylSet* GetMatLUT() const { return processors()->calibObjects.matLUT; }
  const GPUTRDGeometry* GetTRDGeometry() const { return (GPUTRDGeometry*)processors()->calibObjects.trdGeometry; }
  const o2::base::Propagator* GetO2Propagator() const { return processors()->calibObjects.o2Propagator; }
  const o2::base::Propagator* GetDeviceO2Propagator();
  void SetTPCFastTransform(std::unique_ptr<TPCFastTransform>&& tpcFastTransform, std::unique_ptr<CorrectionMapsHelper>&& tpcTransformHelper);
  void SetMatLUT(std::unique_ptr<o2::base::MatLayerCylSet>&& lut);
  void SetTRDGeometry(std::unique_ptr<o2::trd::GeometryFlat>&& geo);
  void SetMatLUT(const o2::base::MatLayerCylSet* lut) { processors()->calibObjects.matLUT = lut; }
  void SetTRDGeometry(const o2::trd::GeometryFlat* geo) { processors()->calibObjects.trdGeometry = geo; }
  void SetO2Propagator(const o2::base::Propagator* prop);
  void SetCalibObjects(const GPUCalibObjectsConst& obj) { processors()->calibObjects = obj; }
  void SetCalibObjects(const GPUCalibObjects& obj) { memcpy((void*)&processors()->calibObjects, (const void*)&obj, sizeof(obj)); }
  void SetUpdateCalibObjects(const GPUCalibObjectsConst& obj, const GPUNewCalibValues& vals);
  void LoadClusterErrors();
  void SetSubOutputControl(int32_t i, GPUOutputControl* v) { mSubOutputControls[i] = v; }
  void SetFinalInputCallback(std::function<void()> v) { mWaitForFinalInputs = v; }

  const GPUSettingsDisplay* mConfigDisplay = nullptr; // Abstract pointer to Standalone Display Configuration Structure
  const GPUSettingsQA* mConfigQA = nullptr;           // Abstract pointer to Standalone QA Configuration Structure
  bool mFractionalQAEnabled = false;

 protected:
  struct GPUTrackingFlatObjects : public GPUProcessor {
    GPUChainTracking* mChainTracking = nullptr;
    GPUCalibObjects mCalibObjects;
    char* mTpcTransformBuffer = nullptr;
    char* mTpcTransformRefBuffer = nullptr;
    char* mTpcTransformMShapeBuffer = nullptr;
    char* mdEdxSplinesBuffer = nullptr;
    char* mMatLUTBuffer = nullptr;
    int16_t mMemoryResFlat = -1;
    void* SetPointersFlatObjects(void* mem);
  };
  void UpdateGPUCalibObjects(int32_t stream, const GPUCalibObjectsConst* ptrMask = nullptr);
  void UpdateGPUCalibObjectsPtrs(int32_t stream);

  struct eventStruct // Must consist only of void* ptr that will hold the GPU event ptrs!
  {
    deviceEvent slice[NSLICES];
    deviceEvent stream[GPUCA_MAX_STREAMS];
    deviceEvent init;
    deviceEvent single;
  };

  struct outputQueueEntry {
    void* dst;
    void* src;
    size_t size;
    RecoStep step;
  };

  GPUChainTracking(GPUReconstruction* rec, uint32_t maxTPCHits = GPUCA_MAX_CLUSTERS, uint32_t maxTRDTracklets = GPUCA_MAX_TRD_TRACKLETS);

  int32_t ReadEvent(uint32_t iSlice, int32_t threadId);
  void WriteOutput(int32_t iSlice, int32_t threadId);
  int32_t GlobalTracking(uint32_t iSlice, int32_t threadId, bool synchronizeOutput = true);

  int32_t PrepareProfile();
  int32_t DoProfile();
  void PrintMemoryRelations();
  void PrintMemoryStatistics() override;
  void PrepareDebugOutput();
  void PrintDebugOutput();
  void PrintOutputStat();

  bool ValidateSteps();
  bool ValidateSettings();

  // Pointers to tracker classes
  GPUTrackingFlatObjects mFlatObjectsShadow; // Host copy of flat objects that will be used on the GPU
  GPUTrackingFlatObjects mFlatObjectsDevice; // flat objects that will be used on the GPU
  std::unique_ptr<GPUTrackingInputProvider> mInputsHost;
  std::unique_ptr<GPUTrackingInputProvider> mInputsShadow;

  // Display / QA
  bool mDisplayRunning = false;
  std::unique_ptr<GPUDisplayInterface> mEventDisplay;
  GPUChainTracking* mQAFromForeignChain = nullptr;
  std::unique_ptr<GPUQA> mQA;
  std::unique_ptr<GPUTPCClusterStatistics> mCompressionStatistics;

  // Ptr to detector / calibration objects
  std::unique_ptr<TPCFastTransform> mTPCFastTransformU;              // Global TPC fast transformation object
  std::unique_ptr<TPCFastTransform> mTPCFastTransformRefU;           // Global TPC fast transformation ref object
  std::unique_ptr<TPCFastTransform> mTPCFastTransformMShapeU;        // Global TPC fast transformation for M-shape object
  std::unique_ptr<CorrectionMapsHelper> mTPCFastTransformHelperU;    // Global TPC fast transformation helper object
  std::unique_ptr<TPCPadGainCalib> mTPCPadGainCalibU;                // TPC gain calibration and cluster finder parameters
  std::unique_ptr<TPCZSLinkMapping> mTPCZSLinkMappingU;              // TPC Mapping data required by ZS Link decoder
  std::unique_ptr<o2::tpc::CalibdEdxContainer> mdEdxCalibContainerU; // TPC dEdx calibration container
  std::unique_ptr<o2::base::MatLayerCylSet> mMatLUTU;                // Material Lookup Table
  std::unique_ptr<o2::trd::GeometryFlat> mTRDGeometryU;              // TRD Geometry

  // Ptrs to internal buffers
  std::unique_ptr<o2::tpc::ClusterNativeAccess> mClusterNativeAccess;
  std::array<GPUOutputControl*, GPUTrackingOutputs::count()> mSubOutputControls = {nullptr};
  std::unique_ptr<GPUTriggerOutputs> mTriggerBuffer;

  // (Ptrs to) configuration objects
  std::unique_ptr<GPUTPCCFChainContext> mCFContext;
  bool mTPCSliceScratchOnStack = false;
  std::unique_ptr<GPUCalibObjectsConst> mNewCalibObjects;
  bool mUpdateNewCalibObjects = false;
  std::unique_ptr<GPUNewCalibValues> mNewCalibValues;

  // Upper bounds for memory allocation
  uint32_t mMaxTPCHits = 0;
  uint32_t mMaxTRDTracklets = 0;

  // Debug
  std::unique_ptr<std::ofstream> mDebugFile;

  // Synchronization and Locks
  eventStruct* mEvents = nullptr;
  VOLATILE int32_t mSliceSelectorReady = 0;
  std::array<int8_t, NSLICES> mWriteOutputDone;

  std::vector<outputQueueEntry> mOutputQueue;

 private:
  int32_t RunChainFinalize();
  void SanityCheck();
  int32_t RunTPCTrackingSlices_internal();
  int32_t RunTPCClusterizer_prepare(bool restorePointers);
#ifdef GPUCA_TPC_GEOMETRY_O2
  std::pair<uint32_t, uint32_t> RunTPCClusterizer_transferZS(int32_t iSlice, const CfFragment& fragment, int32_t lane);
  void RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int32_t stage, bool doGPU, int32_t lane);
  std::pair<uint32_t, uint32_t> TPCClusterizerDecodeZSCount(uint32_t iSlice, const CfFragment& fragment);
  std::pair<uint32_t, uint32_t> TPCClusterizerDecodeZSCountUpdate(uint32_t iSlice, const CfFragment& fragment);
  void TPCClusterizerEnsureZSOffsets(uint32_t iSlice, const CfFragment& fragment);
#endif
  void RunTPCTrackingMerger_MergeBorderTracks(int8_t withinSlice, int8_t mergeMode, GPUReconstruction::krnlDeviceType deviceType);
  void RunTPCTrackingMerger_Resolve(int8_t useOrigTrackParam, int8_t mergeAll, GPUReconstruction::krnlDeviceType deviceType);
  void RunTPCClusterFilter(o2::tpc::ClusterNativeAccess* clusters, std::function<o2::tpc::ClusterNative*(size_t)> allocator, bool applyClusterCuts);

  std::atomic_flag mLockAtomicOutputBuffer = ATOMIC_FLAG_INIT;
  std::mutex mMutexUpdateCalib;
  std::unique_ptr<GPUChainTrackingFinalContext> mPipelineFinalizationCtx;
  GPUChainTrackingFinalContext* mPipelineNotifyCtx = nullptr;
  std::function<void()> mWaitForFinalInputs;

  int32_t HelperReadEvent(int32_t iSlice, int32_t threadId, GPUReconstructionHelpers::helperParam* par);
  int32_t HelperOutput(int32_t iSlice, int32_t threadId, GPUReconstructionHelpers::helperParam* par);

  int32_t OutputStream() const { return mRec->NStreams() - 2; }
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
