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
#include "GPUDataTypesIO.h"
#include "GPUDataTypesConfig.h"
#include "GPUCommonAlignedAlloc.h"
#include <atomic>
#include <mutex>
#include <functional>
#include <array>
#include <vector>
#include <utility>

namespace o2::dataformats
{
template <typename TruthElement>
class ConstMCTruthContainer;
} // namespace o2::dataformats

namespace o2::trd
{
class GeometryFlat;
} // namespace o2::trd

namespace o2::tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
class CalibdEdxContainer;
} // namespace o2::tpc

namespace o2::base
{
class MatLayerCylSet;
template <typename>
class PropagatorImpl;
using Propagator = PropagatorImpl<float>;
} // namespace o2::base

namespace o2::gpu
{
// class GPUTRDTrackerGPU;
class GPUTPCGPUTracker;
class GPUDisplayInterface;
class GPUQA;
class GPUTPCClusterStatistics;
class GPUTRDGeometry;
class GPUTRDRecoParam;
class TPCFastTransformPOD;
class GPUTrackingInputProvider;
struct GPUChainTrackingFinalContext;
struct GPUTPCCFChainContext;
struct GPUNewCalibValues;
struct GPUTriggerOutputs;
struct CfFragment;
class GPUTPCClusterFinder;
struct GPUSettingsProcessing;
struct GPUSettingsRec;

class GPUChainTracking : public GPUChain
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
  static void ApplySyncSettings(GPUSettingsProcessing& proc, GPUSettingsRec& rec, gpudatatypes::RecoStepField& steps, bool syncMode, int32_t dEdxMode = -2);

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
    std::unique_ptr<o2::tpc::Digit[]> tpcDigits[NSECTORS];
    std::unique_ptr<GPUTrackingInOutDigits> digitMap;
    std::unique_ptr<GPUTPCClusterData[]> clusterData[NSECTORS];
    std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSECTORS];
    std::unique_ptr<o2::tpc::ClusterNative[]> clustersNative;
    std::unique_ptr<o2::tpc::ClusterNativeAccess> clusterNativeAccess;
    std::unique_ptr<GPUTPCTrack[]> sectorTracks[NSECTORS];
    std::unique_ptr<GPUTPCHitId[]> sectorClusters[NSECTORS];
    std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
    std::unique_ptr<GPUTPCMCInfo[]> mcInfosTPC;
    std::unique_ptr<GPUTPCMCInfoCol[]> mcInfosTPCCol;
    std::unique_ptr<GPUTPCGMMergedTrack[]> mergedTracks;
    std::unique_ptr<GPUTPCGMMergedTrackHit[]> mergedTrackHits;
    std::unique_ptr<GPUTRDTrackletWord[]> trdTracklets;
    std::unique_ptr<GPUTRDSpacePoint[]> trdSpacePoints;
    std::unique_ptr<float[]> trdTriggerTimes;
    std::unique_ptr<int32_t[]> trdTrackletIdxFirst;
    std::unique_ptr<uint8_t[]> trdTrigRecMask;
    std::unique_ptr<GPUTRDTrackGPU[]> trdTracks;
    std::unique_ptr<char[]> clusterNativeMC;
    std::unique_ptr<o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>> clusterNativeMCView;
    std::unique_ptr<char[]> tpcDigitsMC[NSECTORS];
    std::unique_ptr<o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>[]> tpcDigitMCView;
    std::unique_ptr<GPUTPCDigitsMCInput> tpcDigitMCMap;
    std::unique_ptr<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>> clusterNativeMCBuffer;
    std::unique_ptr<GPUSettingsTF[]> settingsTF;
  } mIOMem;

  // Read / Dump / Clear Data
  void ClearIOPointers();
  void AllocateIOMemory();
  using GPUChain::DumpData;
  void DumpData(const char* filename, const GPUTrackingInOutPointers* ioPtrs = nullptr);
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
  int32_t RunTPCTrackingSectors();
  int32_t RunTPCTrackingMerger(bool synchronizeOutput = true);
  int32_t RunTRDTracking();
  template <int32_t I, class T = GPUTRDTracker>
  int32_t DoTRDGPUTracking(T* externalInstance = nullptr);
  int32_t RunTPCCompression();
  int32_t RunTPCDecompression();
  int32_t RunRefit();

  // Getters / setters for parameters
  const TPCFastTransformPOD* GetTPCTransform() const;
  const TPCPadGainCalib* GetTPCPadGainCalib() const;
  const TPCZSLinkMapping* GetTPCZSLinkMapping() const;
  const o2::tpc::CalibdEdxContainer* GetdEdxCalibContainer() const;
  const o2::base::MatLayerCylSet* GetMatLUT() const;
  const GPUTRDGeometry* GetTRDGeometry() const;
  const GPUTRDRecoParam* GetTRDRecoParam() const;
  const o2::base::Propagator* GetO2Propagator() const;
  const o2::base::Propagator* GetDeviceO2Propagator();
  void SetTPCFastTransform(aligned_unique_buffer_ptr<TPCFastTransformPOD>&& tpcFastTransform);
  void SetMatLUT(std::unique_ptr<o2::base::MatLayerCylSet>&& lut);
  void SetTRDGeometry(std::unique_ptr<o2::trd::GeometryFlat>&& geo);
  void SetTRDRecoParam(std::unique_ptr<GPUTRDRecoParam>&& par);
  void SetMatLUT(const o2::base::MatLayerCylSet* lut);
  void SetTRDGeometry(const o2::trd::GeometryFlat* geo);
  void SetTRDRecoParam(const GPUTRDRecoParam* par);
  void SetO2Propagator(const o2::base::Propagator* prop);
  void SetCalibObjects(const GPUCalibObjectsConst& obj);
  void SetCalibObjects(const GPUCalibObjects& obj);
  void SetUpdateCalibObjects(const GPUCalibObjectsConst& obj, const GPUNewCalibValues& vals);
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
    char* mdEdxSplinesBuffer = nullptr;
    char* mMatLUTBuffer = nullptr;
    int16_t mMemoryResFlat = -1;
    void* SetPointersFlatObjects(void* mem);
  };
  void UpdateGPUCalibObjects(int32_t stream, const GPUCalibObjectsConst* ptrMask = nullptr);
  void UpdateGPUCalibObjectsPtrs(int32_t stream);

  struct eventStruct // Must consist only of void* ptr that will hold the GPU event ptrs!
  {
    deviceEvent sector[NSECTORS];
    deviceEvent stream[constants::GPU_MAX_STREAMS];
    deviceEvent init;
    deviceEvent single;
  };

  struct outputQueueEntry {
    void* dst;
    void* src;
    size_t size;
    RecoStep step;
  };

  GPUChainTracking(GPUReconstruction* rec, uint32_t maxTPCHits = constants::GPU_MEM_MAX_TPC_CLUSTERS, uint32_t maxTRDTracklets = constants::GPU_MEM_MAX_TRD_TRACKLETS);

  int32_t ExtrapolationTracking(uint32_t iSector, bool blocking);

  int32_t PrepareProfile();
  int32_t DoProfile();
  void PrintMemoryRelations();
  void PrintMemoryStatistics() override;
  void PrepareKernelDebugOutput();
  void PrintKernelDebugOutput();
  void PrintOutputStat();
  static void DumpClusters(std::ostream& out, const o2::tpc::ClusterNativeAccess* clusters);
  static void DebugSortCompressedClusters(o2::tpc::CompressedClustersFlat* cls);
  void DoDebugRawDump();

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
  aligned_unique_buffer_ptr<TPCFastTransformPOD> mTPCFastTransformU; // Global TPC fast transformation object
  std::unique_ptr<TPCPadGainCalib> mTPCPadGainCalibU;                // TPC gain calibration and cluster finder parameters
  std::unique_ptr<TPCZSLinkMapping> mTPCZSLinkMappingU;              // TPC Mapping data required by ZS Link decoder
  std::unique_ptr<o2::tpc::CalibdEdxContainer> mdEdxCalibContainerU; // TPC dEdx calibration container
  std::unique_ptr<o2::base::MatLayerCylSet> mMatLUTU;                // Material Lookup Table
  std::unique_ptr<o2::trd::GeometryFlat> mTRDGeometryU;              // TRD Geometry
  std::unique_ptr<GPUTRDRecoParam> mTRDRecoParamU;                   // TRD RecoParam

  // Ptrs to internal buffers
  std::unique_ptr<o2::tpc::ClusterNativeAccess> mClusterNativeAccess, mClusterNativeAccessReduced;
  std::array<GPUOutputControl*, GPUTrackingOutputs::count()> mSubOutputControls = {nullptr};
  std::unique_ptr<GPUTriggerOutputs> mTriggerBuffer;

  // (Ptrs to) configuration objects
  std::unique_ptr<GPUTPCCFChainContext> mCFContext;
  bool mTPCSectorScratchOnStack = false;
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
  std::array<int8_t, NSECTORS> mExtrapolationTrackingDone;

  std::vector<outputQueueEntry> mOutputQueue;

 private:
  int32_t RunChainFinalize();
  void OutputSanityCheck();
  int32_t RunTPCTrackingSectors_internal();
  int32_t RunTPCClusterizer_prepare(bool restorePointers);
#ifndef GPUCA_RUN2
  std::pair<uint32_t, uint32_t> RunTPCClusterizer_transferZS(int32_t iSector, const CfFragment& fragment, int32_t lane);
  void RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int32_t stage, bool doGPU, int32_t lane);
  std::pair<uint32_t, uint32_t> TPCClusterizerDecodeZSCount(uint32_t iSector, const CfFragment& fragment);
  std::pair<uint32_t, uint32_t> TPCClusterizerDecodeZSCountUpdate(uint32_t iSector, const CfFragment& fragment);
  void TPCClusterizerEnsureZSOffsets(uint32_t iSector, const CfFragment& fragment);
#endif
  void RunTPCTrackingMerger_MergeBorderTracks(int8_t withinSector, int8_t mergeMode, GPUReconstruction::krnlDeviceType deviceType);
  void RunTPCTrackingMerger_Resolve(int8_t useOrigTrackParam, int8_t mergeAll, GPUReconstruction::krnlDeviceType deviceType);
  void RunTPCClusterFilter(o2::tpc::ClusterNativeAccess* clusters, std::function<o2::tpc::ClusterNative*(size_t)> allocator, bool applyClusterCuts);
  bool NeedTPCClustersOnGPU();
  void WriteReducedClusters();
  void SortClusters(bool buildNativeGPU, bool propagateMCLabels, o2::tpc::ClusterNativeAccess* clusterAccess, o2::tpc::ClusterNative* clusters);
  template <int32_t I>
  int32_t RunTRDTrackingInternal();
  uint32_t StreamForSector(uint32_t sector) const;

  std::mutex mMutexUpdateCalib;
  std::unique_ptr<GPUChainTrackingFinalContext> mPipelineFinalizationCtx;
  GPUChainTrackingFinalContext* mPipelineNotifyCtx = nullptr;
  std::function<void()> mWaitForFinalInputs;

  int32_t OutputStream() const { return mRec->NStreams() - 2; }
};
} // namespace o2::gpu

#endif
