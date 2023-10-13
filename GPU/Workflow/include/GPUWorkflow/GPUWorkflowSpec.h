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

/// @file   GPURecoWorkflowSpec.h
/// @author Matthias Richter
/// @since  2018-04-18
/// @brief  Processor spec for running TPC CA tracking

#ifndef O2_GPU_WORKFLOW_SPEC_H
#define O2_GPU_WORKFLOW_SPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InitContext.h"
#include "Framework/CompletionPolicy.h"
#include "Algorithm/Parser.h"
#include <string>
#include <array>
#include <vector>
#include <mutex>
#include <functional>
#include <queue>

class TStopwatch;
namespace fair::mq
{
struct RegionInfo;
enum class State : int;
} // namespace fair::mq
namespace o2
{
namespace base
{
struct GRPGeomRequest;
} // namespace base
namespace tpc
{
class CalibdEdxContainer;
struct ClusterGroupHeader;
class VDriftHelper;
class CorrectionMapsLoader;
} // namespace tpc

namespace trd
{
class GeometryFlat;
} // namespace trd

namespace its
{
class Tracker;
class Vertexer;
class TimeFrame;
} // namespace its

namespace itsmft
{
class TopologyDictionary;
} // namespace itsmft

namespace dataformats
{
class MeanVertexObject;
} // namespace dataformats

namespace gpu
{
struct GPUO2InterfaceConfiguration;
class GPUDisplayFrontendInterface;
class CorrectionMapsHelper;
class TPCFastTransform;
struct GPUSettingsTF;
class GPUO2Interface;
struct TPCPadGainCalib;
struct TPCZSLinkMapping;
struct GPUSettingsO2;
class GPUO2InterfaceQA;
struct GPUTrackingInOutPointers;
struct GPUTrackingInOutZS;
struct GPUInterfaceOutputs;
struct GPUInterfaceInputUpdate;
namespace gpurecoworkflow_internals
{
struct GPURecoWorkflowSpec_TPCZSBuffers;
struct GPURecoWorkflowSpec_PipelineInternals;
struct GPURecoWorkflow_QueueObject;
} // namespace gpurecoworkflow_internals

class GPURecoWorkflowSpec : public o2::framework::Task
{
 public:
  using CompletionPolicyData = std::vector<framework::InputSpec>;

  struct Config {
    int itsTriggerType = 0;
    int lumiScaleMode = 0;
    int enableDoublePipeline = 0;
    bool decompressTPC = false;
    bool decompressTPCFromROOT = false;
    bool caClusterer = false;
    bool zsDecoder = false;
    bool zsOnTheFly = false;
    bool outputTracks = false;
    bool outputCompClusters = false;
    bool outputCompClustersFlat = false;
    bool outputCAClusters = false;
    bool outputQA = false;
    bool outputSharedClusterMap = false;
    bool processMC = false;
    bool sendClustersPerSector = false;
    bool askDISTSTF = true;
    bool runTPCTracking = false;
    bool runTRDTracking = false;
    bool readTRDtracklets = false;
    bool requireCTPLumi = false;
    bool outputErrorQA = false;
    bool runITSTracking = false;
    bool itsOverrBeamEst = false;
    bool tpcTriggerHandling = false;
  };

  GPURecoWorkflowSpec(CompletionPolicyData* policyData, Config const& specconfig, std::vector<int> const& tpcsectors, unsigned long tpcSectorMask, std::shared_ptr<o2::base::GRPGeomRequest>& ggr, std::function<bool(o2::framework::DataProcessingHeader::StartTime)>** gPolicyOrder = nullptr);
  ~GPURecoWorkflowSpec() override;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void stop() final;
  o2::framework::Inputs inputs();
  o2::framework::Outputs outputs();
  o2::framework::Options options();

  void deinitialize();

 private:
  struct calibObjectStruct {
    std::unique_ptr<TPCFastTransform> mFastTransform;
    std::unique_ptr<TPCFastTransform> mFastTransformRef;
    std::unique_ptr<o2::tpc::CorrectionMapsLoader> mFastTransformHelper;
    std::unique_ptr<TPCPadGainCalib> mTPCPadGainCalib;
    std::unique_ptr<o2::tpc::CalibdEdxContainer> mdEdxCalibContainer;
  };

  /// initialize TPC options from command line
  void initFunctionTPCCalib(o2::framework::InitContext& ic);
  void initFunctionITS(o2::framework::InitContext& ic);
  /// storing new calib objects in buffer
  void finaliseCCDBTPC(o2::framework::ConcreteDataMatcher& matcher, void* obj);
  void finaliseCCDBITS(o2::framework::ConcreteDataMatcher& matcher, void* obj);
  /// asking for newer calib objects
  template <class T>
  bool fetchCalibsCCDBTPC(o2::framework::ProcessingContext& pc, T& newCalibObjects, calibObjectStruct& oldCalibObjects);
  bool fetchCalibsCCDBITS(o2::framework::ProcessingContext& pc);
  /// delete old calib objects no longer needed
  void cleanOldCalibsTPCPtrs(calibObjectStruct& oldCalibObjects);

  void doCalibUpdates(o2::framework::ProcessingContext& pc, calibObjectStruct& oldCalibObjects);

  void doTrackTuneTPC(GPUTrackingInOutPointers& ptrs, char* buffout);

  template <class D, class E, class F, class G, class H, class I, class J, class K>
  void processInputs(o2::framework::ProcessingContext&, D&, E&, F&, G&, bool&, H&, I&, J&, K&);

  int runMain(o2::framework::ProcessingContext* pc, GPUTrackingInOutPointers* ptrs, GPUInterfaceOutputs* outputRegions, int threadIndex = 0, GPUInterfaceInputUpdate* inputUpdateCallback = nullptr);
  int runITSTracking(o2::framework::ProcessingContext& pc);

  int handlePipeline(o2::framework::ProcessingContext& pc, GPUTrackingInOutPointers& ptrs, gpurecoworkflow_internals::GPURecoWorkflowSpec_TPCZSBuffers& tpcZSmeta, o2::gpu::GPUTrackingInOutZS& tpcZS, std::unique_ptr<gpurecoworkflow_internals::GPURecoWorkflow_QueueObject>& context);
  void RunReceiveThread();
  void RunWorkerThread(int id);
  void ExitPipeline();
  void handlePipelineEndOfStream(o2::framework::EndOfStreamContext& ec);
  void initPipeline(o2::framework::InitContext& ic);
  void enqueuePipelinedJob(GPUTrackingInOutPointers* ptrs, GPUInterfaceOutputs* outputRegions, gpurecoworkflow_internals::GPURecoWorkflow_QueueObject* context, bool inputFinal);
  void finalizeInputPipelinedJob(GPUTrackingInOutPointers* ptrs, GPUInterfaceOutputs* outputRegions, gpurecoworkflow_internals::GPURecoWorkflow_QueueObject* context);
  void receiveFMQStateCallback(fair::mq::State);

  CompletionPolicyData* mPolicyData;
  std::function<bool(o2::framework::DataProcessingHeader::StartTime)> mPolicyOrder;
  std::unique_ptr<GPUO2Interface> mGPUReco;
  std::unique_ptr<GPUDisplayFrontendInterface> mDisplayFrontend;

  calibObjectStruct mCalibObjects;
  std::unique_ptr<o2::tpc::CalibdEdxContainer> mdEdxCalibContainerBufferNew;
  std::unique_ptr<TPCPadGainCalib> mTPCPadGainCalibBufferNew;
  std::queue<calibObjectStruct> mOldCalibObjects;
  std::unique_ptr<TPCZSLinkMapping> mTPCZSLinkMapping;
  std::unique_ptr<o2::tpc::VDriftHelper> mTPCVDriftHelper;
  std::unique_ptr<o2::trd::GeometryFlat> mTRDGeometry;
  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
  std::unique_ptr<GPUSettingsO2> mConfParam;
  std::unique_ptr<TStopwatch> mTimer;
  std::vector<std::array<unsigned int, 4>> mErrorQA;
  int mQATaskMask = 0;
  std::unique_ptr<GPUO2InterfaceQA> mQA;
  std::vector<int> mClusterOutputIds;
  std::vector<int> mTPCSectors;
  std::unique_ptr<o2::its::Tracker> mITSTracker;
  std::unique_ptr<o2::its::Vertexer> mITSVertexer;
  std::unique_ptr<gpurecoworkflow_internals::GPURecoWorkflowSpec_PipelineInternals> mPipeline;
  o2::its::TimeFrame* mITSTimeFrame = nullptr;
  std::vector<fair::mq::RegionInfo> mRegionInfos;
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr;
  const o2::dataformats::MeanVertexObject* mMeanVertex;
  unsigned long mTPCSectorMask = 0;
  int mVerbosity = 0;
  unsigned int mNTFs = 0;
  unsigned int mNDebugDumps = 0;
  unsigned int mNextThreadIndex = 0;
  bool mUpdateGainMapCCDB = true;
  std::unique_ptr<o2::gpu::GPUSettingsTF> mTFSettings;
  Config mSpecConfig;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGR;
  bool mGRPGeomUpdated = false;
  bool mAutoContinuousMaxTimeBin = false;
  bool mAutoSolenoidBz = false;
  bool mMatLUTCreated = false;
  bool mITSGeometryCreated = false;
  bool mTRDGeometryCreated = false;
  bool mPropagatorInstanceCreated = false;
  bool mITSRunVertexer = false;
  bool mITSCosmicsProcessing = false;
  std::string mITSMode = "sync";
};

} // end namespace gpu
} // end namespace o2

#endif // O2_GPU_WORKFLOW_SPEC_H
