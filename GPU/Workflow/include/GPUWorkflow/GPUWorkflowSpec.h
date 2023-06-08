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
#include "Framework/ProcessingContext.h"
#include "Framework/CompletionPolicy.h"
#include "Algorithm/Parser.h"
#include <string>
#include <array>
#include <vector>

class TStopwatch;
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

class GPURecoWorkflowSpec : public o2::framework::Task
{
 public:
  using CompletionPolicyData = std::vector<framework::InputSpec>;

  struct Config {
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
    bool runTRDTracking = false;
    bool readTRDtracklets = false;
    bool requireCTPLumi = false;
    bool outputErrorQA = false;
  };

  GPURecoWorkflowSpec(CompletionPolicyData* policyData, Config const& specconfig, std::vector<int> const& tpcsectors, unsigned long tpcSectorMask, std::shared_ptr<o2::base::GRPGeomRequest>& ggr);
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
  /// initialize TPC options from command line
  void initFunctionTPC(o2::framework::InitContext& ic);
  /// storing new calib objects in buffer
  void finaliseCCDBTPC(o2::framework::ConcreteDataMatcher& matcher, void* obj);
  /// asking for newer calib objects
  template <class T>
  bool fetchCalibsCCDBTPC(o2::framework::ProcessingContext& pc, T& newCalibObjects);
  /// storing the new calib objects by overwritting the old calibs
  void storeUpdatedCalibsTPCPtrs();

  void doCalibUpdates(o2::framework::ProcessingContext& pc);

  CompletionPolicyData* mPolicyData;
  std::unique_ptr<o2::algorithm::ForwardParser<o2::tpc::ClusterGroupHeader>> mParser;
  std::unique_ptr<GPUO2Interface> mTracker;
  std::unique_ptr<GPUDisplayFrontendInterface> mDisplayFrontend;
  std::unique_ptr<TPCFastTransform> mFastTransform;
  std::unique_ptr<TPCFastTransform> mFastTransformRef;
  std::unique_ptr<TPCFastTransform> mFastTransformNew;
  std::unique_ptr<TPCFastTransform> mFastTransformRefNew;
  std::unique_ptr<o2::tpc::CorrectionMapsLoader> mFastTransformHelper;
  std::unique_ptr<o2::tpc::CorrectionMapsLoader> mFastTransformHelperNew;

  std::unique_ptr<TPCPadGainCalib> mTPCPadGainCalib;
  std::unique_ptr<TPCPadGainCalib> mTPCPadGainCalibBufferNew;
  std::unique_ptr<TPCZSLinkMapping> mTPCZSLinkMapping;
  std::unique_ptr<o2::tpc::CalibdEdxContainer> mdEdxCalibContainer;
  std::unique_ptr<o2::tpc::CalibdEdxContainer> mdEdxCalibContainerBufferNew;
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
  unsigned long mTPCSectorMask = 0;
  int mVerbosity = 0;
  unsigned int mNTFs = 0;
  bool mUpdateGainMapCCDB = true;
  std::unique_ptr<o2::gpu::GPUSettingsTF> mTFSettings;
  Config mSpecConfig;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGR;
  bool mGRPGeomUpdated = false;
  bool mAutoContinuousMaxTimeBin = false;
  bool mAutoSolenoidBz = false;
  bool mGeometryCreated = false;
  bool mPropagatorInstanceCreated = false;
  bool mMustUpdateFastTransform = false;
};

} // end namespace gpu
} // end namespace o2

#endif // O2_GPU_WORKFLOW_SPEC_H
