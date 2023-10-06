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

/// @file   GPUWorkflowSpec.cxx
/// @author Matthias Richter, David Rohr
/// @since  2018-04-18
/// @brief  Processor spec for running TPC CA tracking

#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/DataRefUtils.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/SerializationMethods.h"
#include "Framework/Logger.h"
#include "Framework/CallbackService.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/RawDeviceService.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/RawDataTypes.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "TPCReconstruction/TPCTrackingDigitsPreCheck.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCFastTransform.h"
#include "DPLUtils/DPLRawParser.h"
#include "DPLUtils/DPLRawPageSequencer.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonUtils/NameConf.h"
#include "TPCBase/RDHUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceQA.h"
#include "GPUO2Interface.h"
#include "CalibdEdxContainer.h"
#include "GPUNewCalibValues.h"
#include "TPCPadGainCalib.h"
#include "TPCZSLinkMapping.h"
#include "display/GPUDisplayInterface.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/VDriftHelper.h"
#include "CorrectionMapsHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Algorithm/Parser.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/GeometryFlat.h"
#include "ITSBase/GeometryTGeo.h"
#include "CommonUtils/DebugStreamer.h"
#include "GPUReconstructionConvert.h"
#include "DetectorsRaw/RDHUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/Vertexer.h"
#include "GPUWorkflowInternal.h"
// #include "Framework/ThreadPool.h"

#include <TStopwatch.h>
#include <TObjArray.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH1D.h>
#include <TGraphAsymmErrors.h>

#include <filesystem>
#include <memory>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;
using namespace o2::gpu::gpurecoworkflow_internals;

namespace o2::gpu
{

GPURecoWorkflowSpec::GPURecoWorkflowSpec(GPURecoWorkflowSpec::CompletionPolicyData* policyData, Config const& specconfig, std::vector<int> const& tpcsectors, unsigned long tpcSectorMask, std::shared_ptr<o2::base::GRPGeomRequest>& ggr, std::function<bool(o2::framework::DataProcessingHeader::StartTime)>** gPolicyOrder) : o2::framework::Task(), mPolicyData(policyData), mTPCSectorMask(tpcSectorMask), mTPCSectors(tpcsectors), mSpecConfig(specconfig), mGGR(ggr)
{
  if (mSpecConfig.outputCAClusters && !mSpecConfig.caClusterer && !mSpecConfig.decompressTPC) {
    throw std::runtime_error("inconsistent configuration: cluster output is only possible if CA clusterer is activated");
  }

  mConfig.reset(new GPUO2InterfaceConfiguration);
  mConfParam.reset(new GPUSettingsO2);
  mTFSettings.reset(new GPUSettingsTF);
  mTimer.reset(new TStopwatch);
  mPipeline.reset(new GPURecoWorkflowSpec_PipelineInternals);

  if (mSpecConfig.enableDoublePipeline == 1 && gPolicyOrder) {
    *gPolicyOrder = &mPolicyOrder;
  }
}

GPURecoWorkflowSpec::~GPURecoWorkflowSpec() = default;

void GPURecoWorkflowSpec::init(InitContext& ic)
{
  GRPGeomHelper::instance().setRequest(mGGR);
  GPUO2InterfaceConfiguration& config = *mConfig.get();

  // Create configuration object and fill settings
  mConfig->configGRP.solenoidBz = 0;
  mTFSettings->hasSimStartOrbit = 1;
  auto& hbfu = o2::raw::HBFUtils::Instance();
  mTFSettings->simStartOrbit = hbfu.getFirstIRofTF(o2::InteractionRecord(0, hbfu.orbitFirstSampled)).orbit;

  *mConfParam = mConfig->ReadConfigurableParam();
  if (mConfParam->display) {
    mDisplayFrontend.reset(GPUDisplayFrontendInterface::getFrontend(mConfig->configDisplay.displayFrontend.c_str()));
    mConfig->configProcessing.eventDisplay = mDisplayFrontend.get();
    if (mConfig->configProcessing.eventDisplay != nullptr) {
      LOG(info) << "Event display enabled";
    } else {
      throw std::runtime_error("GPU Event Display frontend could not be created!");
    }
  }
  if (mSpecConfig.enableDoublePipeline) {
    mConfig->configProcessing.doublePipeline = 1;
  }

  mAutoSolenoidBz = mConfParam->solenoidBz == -1e6f;
  mAutoContinuousMaxTimeBin = mConfig->configGRP.continuousMaxTimeBin == -1;
  if (mAutoContinuousMaxTimeBin) {
    mConfig->configGRP.continuousMaxTimeBin = (256 * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
  }
  if (mConfig->configProcessing.deviceNum == -2) {
    int myId = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
    int idMax = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
    mConfig->configProcessing.deviceNum = myId;
    LOG(info) << "GPU device number selected from pipeline id: " << myId << " / " << idMax;
  }
  if (mConfig->configProcessing.debugLevel >= 3 && mVerbosity == 0) {
    mVerbosity = 1;
  }
  mConfig->configProcessing.runMC = mSpecConfig.processMC;
  if (mSpecConfig.outputQA) {
    if (!mSpecConfig.processMC && !mConfig->configQA.clusterRejectionHistograms) {
      throw std::runtime_error("Need MC information to create QA plots");
    }
    if (!mSpecConfig.processMC) {
      mConfig->configQA.noMC = true;
    }
    mConfig->configQA.shipToQC = true;
    if (!mConfig->configProcessing.runQA) {
      mConfig->configQA.enableLocalOutput = false;
      mQATaskMask = (mSpecConfig.processMC ? 15 : 0) | (mConfig->configQA.clusterRejectionHistograms ? 32 : 0);
      mConfig->configProcessing.runQA = -mQATaskMask;
    }
  }
  mConfig->configReconstruction.tpc.nWaysOuter = true;
  mConfig->configInterface.outputToExternalBuffers = true;
  if (mConfParam->synchronousProcessing) {
    mConfig->configReconstruction.useMatLUT = false;
  }

  // Configure the "GPU workflow" i.e. which steps we run on the GPU (or CPU)
  if (mSpecConfig.outputTracks || mSpecConfig.outputCompClusters || mSpecConfig.outputCompClustersFlat) {
    mConfig->configWorkflow.steps.set(GPUDataTypes::RecoStep::TPCConversion,
                                      GPUDataTypes::RecoStep::TPCSliceTracking,
                                      GPUDataTypes::RecoStep::TPCMerging);
    mConfig->configWorkflow.outputs.set(GPUDataTypes::InOutType::TPCMergedTracks);
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCdEdx, mConfParam->rundEdx == -1 ? !mConfParam->synchronousProcessing : mConfParam->rundEdx);
  }
  if (mSpecConfig.outputCompClusters || mSpecConfig.outputCompClustersFlat) {
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCCompression, true);
    mConfig->configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, true);
  }
  mConfig->configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCClusters);
  if (mSpecConfig.caClusterer) { // Override some settings if we have raw data as input
    mConfig->configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCRaw);
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCClusterFinding, true);
    mConfig->configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCClusters, true);
  }
  if (mSpecConfig.decompressTPC) {
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCCompression, false);
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCDecompression, true);
    mConfig->configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCCompressedClusters);
    mConfig->configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCClusters, true);
    mConfig->configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, false);
    if (mTPCSectorMask != 0xFFFFFFFFF) {
      throw std::invalid_argument("Cannot run TPC decompression with a sector mask");
    }
  }
  if (mSpecConfig.runTRDTracking) {
    mConfig->configWorkflow.inputs.setBits(GPUDataTypes::InOutType::TRDTracklets, true);
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TRDTracking, true);
  }
  if (mSpecConfig.runITSTracking) {
    mConfig->configWorkflow.inputs.setBits(GPUDataTypes::InOutType::ITSClusters, true);
    mConfig->configWorkflow.outputs.setBits(GPUDataTypes::InOutType::ITSTracks, true);
    mConfig->configWorkflow.steps.setBits(GPUDataTypes::RecoStep::ITSTracking, true);
  }
  if (mSpecConfig.outputSharedClusterMap) {
    mConfig->configProcessing.outputSharedClusterMap = true;
  }
  mConfig->configProcessing.createO2Output = mSpecConfig.outputTracks ? 2 : 0; // Disable O2 TPC track format output if no track output requested
  mConfig->configProcessing.param.tpcTriggerHandling = mSpecConfig.tpcTriggerHandling;

  if (mConfParam->transformationFile.size() || mConfParam->transformationSCFile.size()) {
    LOG(fatal) << "Deprecated configurable param options GPU_global.transformationFile or transformationSCFile used\n"
               << "Instead, link the corresponding file as <somedir>/TPC/Calib/CorrectionMap/snapshot.root and use it via\n"
               << "--condition-remap file://<somdir>=TPC/Calib/CorrectionMap option";
  }
  /* if (config.configProcessing.doublePipeline && ic.services().get<ThreadPool>().poolSize != 2) {
    throw std::runtime_error("double pipeline requires exactly 2 threads");
  } */
  if (config.configProcessing.doublePipeline && (mSpecConfig.readTRDtracklets || mSpecConfig.runITSTracking || !(mSpecConfig.zsOnTheFly || mSpecConfig.zsDecoder))) {
    LOG(fatal) << "GPU two-threaded pipeline works only with TPC-only processing, and with ZS input";
  }

  if (mSpecConfig.enableDoublePipeline != 2) {
    mGPUReco = std::make_unique<GPUO2Interface>();

    // initialize TPC calib objects
    initFunctionTPCCalib(ic);

    mConfig->configCalib.fastTransform = mCalibObjects.mFastTransformHelper->getCorrMap();
    mConfig->configCalib.fastTransformRef = mCalibObjects.mFastTransformHelper->getCorrMapRef();
    mConfig->configCalib.fastTransformHelper = mCalibObjects.mFastTransformHelper.get();
    if (mConfig->configCalib.fastTransform == nullptr) {
      throw std::invalid_argument("GPU workflow: initialization of the TPC transformation failed");
    }

    if (mConfParam->matLUTFile.size()) {
      LOGP(info, "Loading matlut file {}", mConfParam->matLUTFile.c_str());
      mConfig->configCalib.matLUT = o2::base::MatLayerCylSet::loadFromFile(mConfParam->matLUTFile.c_str());
      if (mConfig->configCalib.matLUT == nullptr) {
        LOGF(fatal, "Error loading matlut file");
      }
    } else {
      mConfig->configProcessing.lateO2MatLutProvisioningSize = 50 * 1024 * 1024;
    }

    if (mSpecConfig.readTRDtracklets) {
      mTRDGeometry = std::make_unique<o2::trd::GeometryFlat>();
      mConfig->configCalib.trdGeometry = mTRDGeometry.get();
    }

    mConfig->configProcessing.internalO2PropagatorGPUField = true;

    if (mConfParam->printSettings) {
      mConfig->PrintParam();
    }

    // Configuration is prepared, initialize the tracker.
    if (mGPUReco->Initialize(config) != 0) {
      throw std::invalid_argument("GPU Reconstruction initialization failed");
    }
    if (mSpecConfig.outputQA) {
      mQA = std::make_unique<GPUO2InterfaceQA>(mConfig.get());
    }
    if (mSpecConfig.outputErrorQA) {
      mGPUReco->setErrorCodeOutput(&mErrorQA);
    }

    // initialize ITS
    if (mSpecConfig.runITSTracking) {
      initFunctionITS(ic);
    }
  }

  if (mSpecConfig.enableDoublePipeline) {
    initPipeline(ic);
    if (mConfParam->dump >= 2) {
      LOG(fatal) << "Cannot use dump-only mode with multi-threaded pipeline";
    }
  }

  auto& callbacks = ic.services().get<CallbackService>();
  callbacks.set<CallbackService::Id::RegionInfoCallback>([this](fair::mq::RegionInfo const& info) {
    if (info.size == 0) {
      return;
    }
    if (mSpecConfig.enableDoublePipeline) {
      mRegionInfos.emplace_back(info);
    }
    if (mSpecConfig.enableDoublePipeline == 2) {
      return;
    }
    if (mConfParam->registerSelectedSegmentIds != -1 && info.managed && info.id != (unsigned int)mConfParam->registerSelectedSegmentIds) {
      return;
    }
    int fd = 0;
    if (mConfParam->mutexMemReg) {
      mode_t mask = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
      fd = open("/tmp/o2_gpu_memlock_mutex.lock", O_RDWR | O_CREAT | O_CLOEXEC, mask);
      if (fd == -1) {
        throw std::runtime_error("Error opening lock file");
      }
      fchmod(fd, mask);
      if (lockf(fd, F_LOCK, 0)) {
        throw std::runtime_error("Error locking file");
      }
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    if (mConfParam->benchmarkMemoryRegistration) {
      start = std::chrono::high_resolution_clock::now();
    }
    if (mGPUReco->registerMemoryForGPU(info.ptr, info.size)) {
      throw std::runtime_error("Error registering memory for GPU");
    }
    if (mConfParam->benchmarkMemoryRegistration) {
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      LOG(info) << "Memory registration time (0x" << info.ptr << ", " << info.size << " bytes): " << elapsed_seconds.count() << " s";
    }
    if (mConfParam->mutexMemReg) {
      if (lockf(fd, F_ULOCK, 0)) {
        throw std::runtime_error("Error unlocking file");
      }
      close(fd);
    }
  });

  mTimer->Stop();
  mTimer->Reset();
}

void GPURecoWorkflowSpec::stop()
{
  LOGF(info, "GPU Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer->CpuTime(), mTimer->RealTime(), mTimer->Counter() - 1);
}

void GPURecoWorkflowSpec::endOfStream(EndOfStreamContext& ec)
{
  handlePipelineEndOfStream(ec);
}

void GPURecoWorkflowSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mSpecConfig.enableDoublePipeline != 2) {
    finaliseCCDBTPC(matcher, obj);
    if (mSpecConfig.runITSTracking) {
      finaliseCCDBITS(matcher, obj);
    }
  }
  if (GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    mGRPGeomUpdated = true;
    return;
  }
}

template <class D, class E, class F, class G, class H, class I, class J, class K>
void GPURecoWorkflowSpec::processInputs(ProcessingContext& pc, D& tpcZSmeta, E& inputZS, F& tpcZS, G& tpcZSonTheFlySizes, bool& debugTFDump, H& compClustersDummy, I& compClustersFlatDummy, J& pCompClustersFlat, K& tmpEmptyCompClusters)
{
  if (mSpecConfig.enableDoublePipeline == 1) {
    return;
  }
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;

  if (mSpecConfig.zsOnTheFly || mSpecConfig.zsDecoder) {
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        tpcZSmeta.Pointers[i][j].clear();
        tpcZSmeta.Sizes[i][j].clear();
      }
    }
  }
  if (mSpecConfig.zsOnTheFly) {
    tpcZSonTheFlySizes = {0};
    // tpcZSonTheFlySizes: #zs pages per endpoint:
    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "ZSSIZES"}, Lifetime::Timeframe}};
    bool recv = false, recvsizes = false;
    for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
      if (recvsizes) {
        throw std::runtime_error("Received multiple ZSSIZES data");
      }
      tpcZSonTheFlySizes = pc.inputs().get<std::array<unsigned int, NEndpoints * NSectors>>(ref);
      recvsizes = true;
    }
    // zs pages
    std::vector<InputSpec> filter2 = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "TPCZS"}, Lifetime::Timeframe}};
    for (auto const& ref : InputRecordWalker(pc.inputs(), filter2)) {
      if (recv) {
        throw std::runtime_error("Received multiple TPCZS data");
      }
      inputZS = pc.inputs().get<gsl::span<o2::tpc::ZeroSuppressedContainer8kb>>(ref);
      recv = true;
    }
    if (!recv || !recvsizes) {
      throw std::runtime_error("TPC ZS on the fly data not received");
    }

    unsigned int offset = 0;
    for (unsigned int i = 0; i < NSectors; i++) {
      unsigned int pageSector = 0;
      for (unsigned int j = 0; j < NEndpoints; j++) {
        pageSector += tpcZSonTheFlySizes[i * NEndpoints + j];
        offset += tpcZSonTheFlySizes[i * NEndpoints + j];
      }
      if (mVerbosity >= 1) {
        LOG(info) << "GOT ZS on the fly pages FOR SECTOR " << i << " ->  pages: " << pageSector;
      }
    }
  }
  if (mSpecConfig.zsDecoder) {
    std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
    auto isSameRdh = [](const char* left, const char* right) -> bool {
      return o2::raw::RDHUtils::getFEEID(left) == o2::raw::RDHUtils::getFEEID(right) && o2::raw::RDHUtils::getDetectorField(left) == o2::raw::RDHUtils::getDetectorField(right);
    };
    auto checkForZSData = [](const char* ptr, uint32_t subSpec) -> bool {
      const auto rdhLink = o2::raw::RDHUtils::getLinkID(ptr);
      const auto detField = o2::raw::RDHUtils::getDetectorField(ptr);
      const auto feeID = o2::raw::RDHUtils::getFEEID(ptr);
      const auto feeLinkID = o2::tpc::rdh_utils::getLink(feeID);
      // This check is not what it is supposed to be, but some MC SYNTHETIC data was generated with rdhLinkId set to feeLinkId, so we add some extra logic so we can still decode it
      return detField == o2::tpc::raw_data_types::ZS && ((feeLinkID == o2::tpc::rdh_utils::UserLogicLinkID && (rdhLink == o2::tpc::rdh_utils::UserLogicLinkID || rdhLink == 0)) ||
                                                         (feeLinkID == o2::tpc::rdh_utils::ILBZSLinkID && (rdhLink == o2::tpc::rdh_utils::UserLogicLinkID || rdhLink == o2::tpc::rdh_utils::ILBZSLinkID || rdhLink == 0)) ||
                                                         (feeLinkID == o2::tpc::rdh_utils::DLBZSLinkID && (rdhLink == o2::tpc::rdh_utils::UserLogicLinkID || rdhLink == o2::tpc::rdh_utils::DLBZSLinkID || rdhLink == 0)));
    };
    auto insertPages = [&tpcZSmeta, checkForZSData](const char* ptr, size_t count, uint32_t subSpec) -> void {
      if (checkForZSData(ptr, subSpec)) {
        int rawcru = o2::tpc::rdh_utils::getCRU(ptr);
        int rawendpoint = o2::tpc::rdh_utils::getEndPoint(ptr);
        tpcZSmeta.Pointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(ptr);
        tpcZSmeta.Sizes[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(count);
      }
    };
    if (DPLRawPageSequencer(pc.inputs(), filter)(isSameRdh, insertPages, checkForZSData)) {
      debugTFDump = true;
      static unsigned int nErrors = 0;
      nErrors++;
      if (nErrors == 1 || (nErrors < 100 && nErrors % 10 == 0) || nErrors % 1000 == 0 || mNTFs % 1000 == 0) {
        LOG(error) << "DPLRawPageSequencer failed to process TPC raw data - data most likely not padded correctly - Using slow page scan instead (this alarm is downscaled from now on, so far " << nErrors << " of " << mNTFs << " TFs affected)";
      }
    }

    int totalCount = 0;
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        tpcZSmeta.Pointers2[i][j] = tpcZSmeta.Pointers[i][j].data();
        tpcZSmeta.Sizes2[i][j] = tpcZSmeta.Sizes[i][j].data();
        tpcZS.slice[i].zsPtr[j] = tpcZSmeta.Pointers2[i][j];
        tpcZS.slice[i].nZSPtr[j] = tpcZSmeta.Sizes2[i][j];
        tpcZS.slice[i].count[j] = tpcZSmeta.Pointers[i][j].size();
        totalCount += tpcZSmeta.Pointers[i][j].size();
      }
    }
  } else if (mSpecConfig.decompressTPC) {
    if (mSpecConfig.decompressTPCFromROOT) {
      compClustersDummy = *pc.inputs().get<o2::tpc::CompressedClustersROOT*>("input");
      compClustersFlatDummy.setForward(&compClustersDummy);
      pCompClustersFlat = &compClustersFlatDummy;
    } else {
      pCompClustersFlat = pc.inputs().get<o2::tpc::CompressedClustersFlat*>("input").get();
    }
    if (pCompClustersFlat == nullptr) {
      tmpEmptyCompClusters.reset(new char[sizeof(o2::tpc::CompressedClustersFlat)]);
      memset(tmpEmptyCompClusters.get(), 0, sizeof(o2::tpc::CompressedClustersFlat));
      pCompClustersFlat = (o2::tpc::CompressedClustersFlat*)tmpEmptyCompClusters.get();
    }
  } else if (!mSpecConfig.zsOnTheFly) {
    if (mVerbosity) {
      LOGF(info, "running tracking for sector(s) 0x%09x", mTPCSectorMask);
    }
  }
}

int GPURecoWorkflowSpec::runMain(o2::framework::ProcessingContext* pc, GPUTrackingInOutPointers* ptrs, GPUInterfaceOutputs* outputRegions, int threadIndex, GPUInterfaceInputUpdate* inputUpdateCallback)
{
  int retVal = 0;
  if (mConfParam->dump < 2) {
    retVal = mGPUReco->RunTracking(ptrs, outputRegions, threadIndex, inputUpdateCallback);

    if (retVal == 0 && mSpecConfig.runITSTracking) {
      retVal = runITSTracking(*pc);
    }
  }

  if (!mSpecConfig.enableDoublePipeline) { // TODO: Why is this needed for double-pipeline?
    mGPUReco->Clear(false, threadIndex);   // clean non-output memory used by GPU Reconstruction
  }
  return retVal;
}

void GPURecoWorkflowSpec::cleanOldCalibsTPCPtrs(calibObjectStruct& oldCalibObjects)
{
  if (mOldCalibObjects.size() > 0) {
    mOldCalibObjects.pop();
  }
  mOldCalibObjects.emplace(std::move(oldCalibObjects));
}

void GPURecoWorkflowSpec::run(ProcessingContext& pc)
{
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;

  auto cput = mTimer->CpuTime();
  auto realt = mTimer->RealTime();
  mTimer->Start(false);
  mNTFs++;

  std::vector<gsl::span<const char>> inputs;

  const o2::tpc::CompressedClustersFlat* pCompClustersFlat = nullptr;
  size_t compClustersFlatDummyMemory[(sizeof(o2::tpc::CompressedClustersFlat) + sizeof(size_t) - 1) / sizeof(size_t)];
  o2::tpc::CompressedClustersFlat& compClustersFlatDummy = reinterpret_cast<o2::tpc::CompressedClustersFlat&>(compClustersFlatDummyMemory);
  o2::tpc::CompressedClusters compClustersDummy;
  o2::gpu::GPUTrackingInOutZS tpcZS;
  GPURecoWorkflowSpec_TPCZSBuffers tpcZSmeta;
  std::array<unsigned int, NEndpoints * NSectors> tpcZSonTheFlySizes;
  gsl::span<const o2::tpc::ZeroSuppressedContainer8kb> inputZS;
  std::unique_ptr<char[]> tmpEmptyCompClusters;

  bool getWorkflowTPCInput_clusters = false, getWorkflowTPCInput_mc = false, getWorkflowTPCInput_digits = false;
  bool debugTFDump = false;

  if (mSpecConfig.processMC) {
    getWorkflowTPCInput_mc = true;
  }
  if (!mSpecConfig.decompressTPC && !mSpecConfig.caClusterer) {
    getWorkflowTPCInput_clusters = true;
  }
  if (!mSpecConfig.decompressTPC && mSpecConfig.caClusterer && ((!mSpecConfig.zsOnTheFly || mSpecConfig.processMC) && !mSpecConfig.zsDecoder)) {
    getWorkflowTPCInput_digits = true;
  }

  // ------------------------------ Handle inputs ------------------------------

  auto lockDecodeInput = std::make_unique<std::lock_guard<std::mutex>>(mPipeline->mutexDecodeInput);

  GRPGeomHelper::instance().checkUpdates(pc);
  if (GRPGeomHelper::instance().getGRPECS()->isDetReadOut(o2::detectors::DetID::TPC) && mConfParam->tpcTriggeredMode ^ !GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::TPC)) {
    LOG(fatal) << "configKeyValue tpcTriggeredMode does not match GRP isDetContinuousReadOut(TPC) setting";
  }

  GPUTrackingInOutPointers ptrs;
  processInputs(pc, tpcZSmeta, inputZS, tpcZS, tpcZSonTheFlySizes, debugTFDump, compClustersDummy, compClustersFlatDummy, pCompClustersFlat, tmpEmptyCompClusters);                  // Process non-digit / non-cluster inputs
  const auto& inputsClustersDigits = o2::tpc::getWorkflowTPCInput(pc, mVerbosity, getWorkflowTPCInput_mc, getWorkflowTPCInput_clusters, mTPCSectorMask, getWorkflowTPCInput_digits); // Process digit and cluster inputs

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  mTFSettings->tfStartOrbit = tinfo.firstTForbit;
  mTFSettings->hasTfStartOrbit = 1;
  mTFSettings->hasNHBFPerTF = 1;
  mTFSettings->nHBFPerTF = GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF();
  mTFSettings->hasRunStartOrbit = 0;
  if (mVerbosity) {
    LOG(info) << "TF firstTForbit " << mTFSettings->tfStartOrbit << " nHBF " << mTFSettings->nHBFPerTF << " runStartOrbit " << mTFSettings->runStartOrbit << " simStartOrbit " << mTFSettings->simStartOrbit;
  }
  ptrs.settingsTF = mTFSettings.get();

  if (mConfParam->checkFirstTfOrbit) {
    static uint32_t lastFirstTFOrbit = -1;
    static uint32_t lastTFCounter = -1;
    if (lastFirstTFOrbit != -1 && lastTFCounter != -1) {
      int diffOrbit = tinfo.firstTForbit - lastFirstTFOrbit;
      int diffCounter = tinfo.tfCounter - lastTFCounter;
      if (diffOrbit != diffCounter * mTFSettings->nHBFPerTF) {
        LOG(error) << "Time frame has mismatching firstTfOrbit - Last orbit/counter: " << lastFirstTFOrbit << " " << lastTFCounter << " - Current: " << tinfo.firstTForbit << " " << tinfo.tfCounter;
      }
    }
    lastFirstTFOrbit = tinfo.firstTForbit;
    lastTFCounter = tinfo.tfCounter;
  }

  o2::globaltracking::RecoContainer inputTracksTRD;
  decltype(o2::trd::getRecoInputContainer(pc, &ptrs, &inputTracksTRD)) trdInputContainer;
  if (mSpecConfig.readTRDtracklets) {
    o2::globaltracking::DataRequest dataRequestTRD;
    dataRequestTRD.requestTracks(o2::dataformats::GlobalTrackID::getSourcesMask(o2::dataformats::GlobalTrackID::NONE), false);
    inputTracksTRD.collectData(pc, dataRequestTRD);
    trdInputContainer = std::move(o2::trd::getRecoInputContainer(pc, &ptrs, &inputTracksTRD));
  }

  void* ptrEp[NSectors * NEndpoints] = {};
  bool doInputDigits = false, doInputDigitsMC = false;
  if (mSpecConfig.decompressTPC) {
    ptrs.tpcCompressedClusters = pCompClustersFlat;
  } else if (mSpecConfig.zsOnTheFly) {
    const unsigned long long int* buffer = reinterpret_cast<const unsigned long long int*>(&inputZS[0]);
    o2::gpu::GPUReconstructionConvert::RunZSEncoderCreateMeta(buffer, tpcZSonTheFlySizes.data(), *&ptrEp, &tpcZS);
    ptrs.tpcZS = &tpcZS;
    doInputDigits = doInputDigitsMC = mSpecConfig.processMC;
  } else if (mSpecConfig.zsDecoder) {
    ptrs.tpcZS = &tpcZS;
    if (mSpecConfig.processMC) {
      throw std::runtime_error("Cannot process MC information, none available");
    }
  } else if (mSpecConfig.caClusterer) {
    doInputDigits = true;
    doInputDigitsMC = mSpecConfig.processMC;
  } else {
    ptrs.clustersNative = &inputsClustersDigits->clusterIndex;
  }

  if (mTPCSectorMask != 0xFFFFFFFFF) {
    // Clean out the unused sectors, such that if they were present by chance, they are not processed, and if the values are uninitialized, we should not crash
    for (unsigned int i = 0; i < NSectors; i++) {
      if (!(mTPCSectorMask & (1ul << i))) {
        if (ptrs.tpcZS) {
          for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
            tpcZS.slice[i].zsPtr[j] = nullptr;
            tpcZS.slice[i].nZSPtr[j] = nullptr;
            tpcZS.slice[i].count[j] = 0;
          }
        }
      }
    }
  }

  GPUTrackingInOutDigits tpcDigitsMap;
  GPUTPCDigitsMCInput tpcDigitsMapMC;
  if (doInputDigits) {
    ptrs.tpcPackedDigits = &tpcDigitsMap;
    if (doInputDigitsMC) {
      tpcDigitsMap.tpcDigitsMC = &tpcDigitsMapMC;
    }
    for (unsigned int i = 0; i < NSectors; i++) {
      tpcDigitsMap.tpcDigits[i] = inputsClustersDigits->inputDigits[i].data();
      tpcDigitsMap.nTPCDigits[i] = inputsClustersDigits->inputDigits[i].size();
      if (doInputDigitsMC) {
        tpcDigitsMapMC.v[i] = inputsClustersDigits->inputDigitsMCPtrs[i];
      }
    }
  }

  o2::tpc::TPCSectorHeader clusterOutputSectorHeader{0};
  if (mClusterOutputIds.size() > 0) {
    clusterOutputSectorHeader.sectorBits = mTPCSectorMask;
    // subspecs [0, NSectors - 1] are used to identify sector data, we use NSectors to indicate the full TPC
    clusterOutputSectorHeader.activeSectors = mTPCSectorMask;
  }

  // ------------------------------ Prepare stage for double-pipeline before normal output preparation ------------------------------

  std::unique_ptr<GPURecoWorkflow_QueueObject> pipelineContext;
  if (mSpecConfig.enableDoublePipeline) {
    if (handlePipeline(pc, ptrs, tpcZSmeta, tpcZS, pipelineContext)) {
      return;
    }
  }

  // ------------------------------ Prepare outputs ------------------------------

  GPUInterfaceOutputs outputRegions;
  using outputDataType = char;
  using outputBufferUninitializedVector = std::decay_t<decltype(pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(Output{"", "", 0}))>;
  using outputBufferType = std::pair<std::optional<std::reference_wrapper<outputBufferUninitializedVector>>, outputDataType*>;
  std::vector<outputBufferType> outputBuffers(GPUInterfaceOutputs::count(), {std::nullopt, nullptr});

  auto setOutputAllocator = [this, &outputBuffers, &outputRegions, &pc](const char* name, bool condition, GPUOutputControl& region, auto&& outputSpec, size_t offset = 0) {
    if (condition) {
      auto& buffer = outputBuffers[outputRegions.getIndex(region)];
      if (mConfParam->allocateOutputOnTheFly) {
        region.allocator = [this, name, &buffer, &pc, outputSpec = std::move(outputSpec), offset](size_t size) -> void* {
          size += offset;
          if (mVerbosity) {
            LOG(info) << "ALLOCATING " << size << " bytes for " << std::get<DataOrigin>(outputSpec).template as<std::string>() << "/" << std::get<DataDescription>(outputSpec).template as<std::string>() << "/" << std::get<2>(outputSpec);
          }
          std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
          if (mVerbosity) {
            start = std::chrono::high_resolution_clock::now();
          }
          buffer.first.emplace(pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(std::make_from_tuple<Output>(outputSpec), size));
          if (mVerbosity) {
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            LOG(info) << "Allocation time for " << name << " (" << size << " bytes)"
                      << ": " << elapsed_seconds.count() << "s";
          }
          return (buffer.second = buffer.first->get().data()) + offset;
        };
      } else {
        buffer.first.emplace(pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(std::make_from_tuple<Output>(outputSpec), mConfParam->outputBufferSize));
        region.ptrBase = (buffer.second = buffer.first->get().data()) + offset;
        region.size = buffer.first->get().size() - offset;
      }
    }
  };

  auto downSizeBuffer = [](outputBufferType& buffer, size_t size) {
    if (!buffer.first) {
      return;
    }
    if (buffer.first->get().size() < size) {
      throw std::runtime_error("Invalid buffer size requested");
    }
    buffer.first->get().resize(size);
    if (size && buffer.first->get().data() != buffer.second) {
      throw std::runtime_error("Inconsistent buffer address after downsize");
    }
  };

  /*auto downSizeBufferByName = [&outputBuffers, &outputRegions, &downSizeBuffer](GPUOutputControl& region, size_t size) {
    auto& buffer = outputBuffers[outputRegions.getIndex(region)];
    downSizeBuffer(buffer, size);
  };*/

  auto downSizeBufferToSpan = [&outputBuffers, &outputRegions, &downSizeBuffer](GPUOutputControl& region, auto span) {
    auto& buffer = outputBuffers[outputRegions.getIndex(region)];
    if (!buffer.first) {
      return;
    }
    if (span.size() && buffer.second != (char*)span.data()) {
      throw std::runtime_error("Buffer does not match span");
    }
    downSizeBuffer(buffer, span.size() * sizeof(*span.data()));
  };

  setOutputAllocator("COMPCLUSTERSFLAT", mSpecConfig.outputCompClustersFlat, outputRegions.compressedClusters, std::make_tuple(gDataOriginTPC, (DataDescription) "COMPCLUSTERSFLAT", 0));
  setOutputAllocator("CLUSTERNATIVE", mClusterOutputIds.size() > 0, outputRegions.clustersNative, std::make_tuple(gDataOriginTPC, mSpecConfig.sendClustersPerSector ? (DataDescription) "CLUSTERNATIVETMP" : (DataDescription) "CLUSTERNATIVE", NSectors, Lifetime::Timeframe, clusterOutputSectorHeader), sizeof(o2::tpc::ClusterCountIndex));
  setOutputAllocator("CLSHAREDMAP", mSpecConfig.outputSharedClusterMap, outputRegions.sharedClusterMap, std::make_tuple(gDataOriginTPC, (DataDescription) "CLSHAREDMAP", 0));
  setOutputAllocator("TRACKS", mSpecConfig.outputTracks, outputRegions.tpcTracksO2, std::make_tuple(gDataOriginTPC, (DataDescription) "TRACKS", 0));
  setOutputAllocator("CLUSREFS", mSpecConfig.outputTracks, outputRegions.tpcTracksO2ClusRefs, std::make_tuple(gDataOriginTPC, (DataDescription) "CLUSREFS", 0));
  setOutputAllocator("TRACKSMCLBL", mSpecConfig.outputTracks && mSpecConfig.processMC, outputRegions.tpcTracksO2Labels, std::make_tuple(gDataOriginTPC, (DataDescription) "TRACKSMCLBL", 0));
  setOutputAllocator("TRIGGERWORDS", mSpecConfig.zsDecoder && mConfig->configProcessing.param.tpcTriggerHandling, outputRegions.tpcTriggerWords, std::make_tuple(gDataOriginTPC, (DataDescription) "TRIGGERWORDS", 0));
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clustersMCBuffer;
  if (mSpecConfig.processMC && mSpecConfig.caClusterer) {
    outputRegions.clusterLabels.allocator = [&clustersMCBuffer](size_t size) -> void* { return &clustersMCBuffer; };
  }

  // ------------------------------ Actual processing ------------------------------

  if ((int)(ptrs.tpcZS != nullptr) + (int)(ptrs.tpcPackedDigits != nullptr && (ptrs.tpcZS == nullptr || ptrs.tpcPackedDigits->tpcDigitsMC == nullptr)) + (int)(ptrs.clustersNative != nullptr) + (int)(ptrs.tpcCompressedClusters != nullptr) != 1) {
    throw std::runtime_error("Invalid input for gpu tracking");
  }

  const auto& holdData = o2::tpc::TPCTrackingDigitsPreCheck::runPrecheck(&ptrs, mConfig.get());

  calibObjectStruct oldCalibObjects;
  doCalibUpdates(pc, oldCalibObjects);

  lockDecodeInput.reset();

  if (mConfParam->dump) {
    if (mNTFs == 1) {
      mGPUReco->DumpSettings();
    }
    mGPUReco->DumpEvent(mNTFs - 1, &ptrs);
  }
  std::unique_ptr<GPUTrackingInOutPointers> ptrsDump;
  if (mConfParam->dumpBadTFMode == 2) {
    ptrsDump.reset(new GPUTrackingInOutPointers);
    memcpy((void*)ptrsDump.get(), (const void*)&ptrs, sizeof(ptrs));
  }

  int retVal = 0;
  if (mSpecConfig.enableDoublePipeline) {
    if (!pipelineContext->jobSubmitted) {
      enqueuePipelinedJob(&ptrs, &outputRegions, pipelineContext.get(), true);
    } else {
      finalizeInputPipelinedJob(&ptrs, &outputRegions, pipelineContext.get());
    }
    std::unique_lock lk(pipelineContext->jobFinishedMutex);
    pipelineContext->jobFinishedNotify.wait(lk, [context = pipelineContext.get()]() { return context->jobFinished; });
    retVal = pipelineContext->jobReturnValue;
  } else {
    // unsigned int threadIndex = pc.services().get<ThreadPool>().threadIndex;
    unsigned int threadIndex = mNextThreadIndex;
    if (mConfig->configProcessing.doublePipeline) {
      mNextThreadIndex = (mNextThreadIndex + 1) % 2;
    }

    retVal = runMain(&pc, &ptrs, &outputRegions, threadIndex);
  }
  if (retVal != 0) {
    debugTFDump = true;
  }
  cleanOldCalibsTPCPtrs(oldCalibObjects);

  o2::utils::DebugStreamer::instance()->flush(); // flushing debug output to file

  if (debugTFDump && mNDebugDumps < mConfParam->dumpBadTFs) {
    mNDebugDumps++;
    if (mConfParam->dumpBadTFMode <= 1) {
      std::string filename = std::string("tpc_dump_") + std::to_string(pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId) + "_" + std::to_string(mNDebugDumps) + ".dump";
      FILE* fp = fopen(filename.c_str(), "w+b");
      std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
      for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
        auto data = pc.inputs().get<gsl::span<char>>(ref);
        if (mConfParam->dumpBadTFMode == 1) {
          unsigned long size = data.size();
          fwrite(&size, 1, sizeof(size), fp);
        }
        fwrite(data.data(), 1, data.size(), fp);
      }
      fclose(fp);
    } else if (mConfParam->dumpBadTFMode == 2) {
      mGPUReco->DumpEvent(mNDebugDumps - 1, ptrsDump.get());
    }
  }

  if (mConfParam->dump == 2) {
    return;
  }

  // ------------------------------ Varios postprocessing steps ------------------------------

  bool createEmptyOutput = false;
  if (retVal != 0) {
    if (retVal == 3 && mConfig->configProcessing.ignoreNonFatalGPUErrors) {
      if (mConfig->configProcessing.throttleAlarms) {
        LOG(warning) << "GPU Reconstruction aborted with non fatal error code, ignoring";
      } else {
        LOG(alarm) << "GPU Reconstruction aborted with non fatal error code, ignoring";
      }
      createEmptyOutput = !mConfParam->partialOutputForNonFatalErrors;
    } else {
      throw std::runtime_error("tracker returned error code " + std::to_string(retVal));
    }
  }

  std::unique_ptr<o2::tpc::ClusterNativeAccess> tmpEmptyClNative;
  if (createEmptyOutput) {
    memset(&ptrs, 0, sizeof(ptrs));
    for (unsigned int i = 0; i < outputRegions.count(); i++) {
      if (outputBuffers[i].first) {
        size_t toSize = 0;
        if (i == outputRegions.getIndex(outputRegions.compressedClusters)) {
          toSize = sizeof(*ptrs.tpcCompressedClusters);
        } else if (i == outputRegions.getIndex(outputRegions.clustersNative)) {
          toSize = sizeof(o2::tpc::ClusterCountIndex);
        }
        outputBuffers[i].first->get().resize(toSize);
        outputBuffers[i].second = outputBuffers[i].first->get().data();
        if (toSize) {
          memset(outputBuffers[i].second, 0, toSize);
        }
      }
    }
    tmpEmptyClNative = std::make_unique<o2::tpc::ClusterNativeAccess>();
    memset(tmpEmptyClNative.get(), 0, sizeof(*tmpEmptyClNative));
    ptrs.clustersNative = tmpEmptyClNative.get();
    if (mSpecConfig.processMC) {
      MCLabelContainer cont;
      cont.flatten_to(clustersMCBuffer.first);
      clustersMCBuffer.second = clustersMCBuffer.first;
      tmpEmptyClNative->clustersMCTruth = &clustersMCBuffer.second;
    }
  } else {
    gsl::span<const o2::tpc::TrackTPC> spanOutputTracks = {ptrs.outputTracksTPCO2, ptrs.nOutputTracksTPCO2};
    gsl::span<const uint32_t> spanOutputClusRefs = {ptrs.outputClusRefsTPCO2, ptrs.nOutputClusRefsTPCO2};
    gsl::span<const o2::MCCompLabel> spanOutputTracksMCTruth = {ptrs.outputTracksTPCO2MC, ptrs.outputTracksTPCO2MC ? ptrs.nOutputTracksTPCO2 : 0};
    if (!mConfParam->allocateOutputOnTheFly) {
      for (unsigned int i = 0; i < outputRegions.count(); i++) {
        if (outputRegions.asArray()[i].ptrBase) {
          if (outputRegions.asArray()[i].size == 1) {
            throw std::runtime_error("Preallocated buffer size exceeded");
          }
          outputRegions.asArray()[i].checkCurrent();
          downSizeBuffer(outputBuffers[i], (char*)outputRegions.asArray()[i].ptrCurrent - (char*)outputBuffers[i].second);
        }
      }
    }
    downSizeBufferToSpan(outputRegions.tpcTracksO2, spanOutputTracks);
    downSizeBufferToSpan(outputRegions.tpcTracksO2ClusRefs, spanOutputClusRefs);
    downSizeBufferToSpan(outputRegions.tpcTracksO2Labels, spanOutputTracksMCTruth);

    // if requested, tune TPC tracks
    if (ptrs.nOutputTracksTPCO2) {
      doTrackTuneTPC(ptrs, outputBuffers[outputRegions.getIndex(outputRegions.tpcTracksO2)].first->get().data());
    }

    if (mClusterOutputIds.size() > 0 && (void*)ptrs.clustersNative->clustersLinear != (void*)(outputBuffers[outputRegions.getIndex(outputRegions.clustersNative)].second + sizeof(o2::tpc::ClusterCountIndex))) {
      throw std::runtime_error("cluster native output ptrs out of sync"); // sanity check
    }
  }

  if (mConfig->configWorkflow.outputs.isSet(GPUDataTypes::InOutType::TPCMergedTracks)) {
    LOG(info) << "found " << ptrs.nOutputTracksTPCO2 << " track(s)";
  }

  if (mSpecConfig.outputCompClusters) {
    o2::tpc::CompressedClustersROOT compressedClusters = *ptrs.tpcCompressedClusters;
    pc.outputs().snapshot(Output{gDataOriginTPC, "COMPCLUSTERS", 0}, ROOTSerialized<o2::tpc::CompressedClustersROOT const>(compressedClusters));
  }

  if (mClusterOutputIds.size() > 0) {
    o2::tpc::ClusterNativeAccess const& accessIndex = *ptrs.clustersNative;
    if (mSpecConfig.sendClustersPerSector) {
      // Clusters are shipped by sector, we are copying into per-sector buffers (anyway only for ROOT output)
      for (unsigned int i = 0; i < NSectors; i++) {
        if (mTPCSectorMask & (1ul << i)) {
          DataHeader::SubSpecificationType subspec = i;
          clusterOutputSectorHeader.sectorBits = (1ul << i);
          char* buffer = pc.outputs().make<char>({gDataOriginTPC, "CLUSTERNATIVE", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, accessIndex.nClustersSector[i] * sizeof(*accessIndex.clustersLinear) + sizeof(o2::tpc::ClusterCountIndex)).data();
          o2::tpc::ClusterCountIndex* outIndex = reinterpret_cast<o2::tpc::ClusterCountIndex*>(buffer);
          memset(outIndex, 0, sizeof(*outIndex));
          for (int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
            outIndex->nClusters[i][j] = accessIndex.nClusters[i][j];
          }
          memcpy(buffer + sizeof(*outIndex), accessIndex.clusters[i][0], accessIndex.nClustersSector[i] * sizeof(*accessIndex.clustersLinear));
          if (mSpecConfig.processMC && accessIndex.clustersMCTruth) {
            MCLabelContainer cont;
            for (unsigned int j = 0; j < accessIndex.nClustersSector[i]; j++) {
              const auto& labels = accessIndex.clustersMCTruth->getLabels(accessIndex.clusterOffset[i][0] + j);
              for (const auto& label : labels) {
                cont.addElement(j, label);
              }
            }
            ConstMCLabelContainer contflat;
            cont.flatten_to(contflat);
            pc.outputs().snapshot({gDataOriginTPC, "CLNATIVEMCLBL", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, contflat);
          }
        }
      }
    } else {
      // Clusters are shipped as single message, fill ClusterCountIndex
      DataHeader::SubSpecificationType subspec = NSectors;
      o2::tpc::ClusterCountIndex* outIndex = reinterpret_cast<o2::tpc::ClusterCountIndex*>(outputBuffers[outputRegions.getIndex(outputRegions.clustersNative)].second);
      static_assert(sizeof(o2::tpc::ClusterCountIndex) == sizeof(accessIndex.nClusters));
      memcpy(outIndex, &accessIndex.nClusters[0][0], sizeof(o2::tpc::ClusterCountIndex));
      if (mSpecConfig.processMC && mSpecConfig.caClusterer && accessIndex.clustersMCTruth) {
        pc.outputs().snapshot({gDataOriginTPC, "CLNATIVEMCLBL", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, clustersMCBuffer.first);
      }
    }
  }
  if (mSpecConfig.outputQA) {
    TObjArray out;
    bool sendQAOutput = !createEmptyOutput && outputRegions.qa.newQAHistsCreated;
    auto getoutput = [sendQAOutput](auto ptr) { return sendQAOutput && ptr ? *ptr : std::decay_t<decltype(*ptr)>(); };
    std::vector<TH1F> copy1 = getoutput(outputRegions.qa.hist1); // Internally, this will also be used as output, so we need a non-const copy
    std::vector<TH2F> copy2 = getoutput(outputRegions.qa.hist2);
    std::vector<TH1D> copy3 = getoutput(outputRegions.qa.hist3);
    std::vector<TGraphAsymmErrors> copy4 = getoutput(outputRegions.qa.hist4);
    if (sendQAOutput) {
      mQA->postprocessExternal(copy1, copy2, copy3, copy4, out, mQATaskMask ? mQATaskMask : -1);
    }
    pc.outputs().snapshot({gDataOriginTPC, "TRACKINGQA", 0, Lifetime::Timeframe}, out);
    if (sendQAOutput) {
      mQA->cleanup();
    }
  }
  if (mSpecConfig.outputErrorQA) {
    pc.outputs().snapshot({gDataOriginGPU, "ERRORQA", 0, Lifetime::Timeframe}, mErrorQA);
    mErrorQA.clear(); // FIXME: This is a race condition once we run multi-threaded!
  }
  if (mSpecConfig.tpcTriggerHandling && !(mSpecConfig.zsOnTheFly || mSpecConfig.zsDecoder)) {
    pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(Output{gDataOriginTPC, "TRIGGERWORDS", 0, Lifetime::Timeframe}, 0u);
  }
  mTimer->Stop();
  LOG(info) << "GPU Reoncstruction time for this TF " << mTimer->CpuTime() - cput << " s (cpu), " << mTimer->RealTime() - realt << " s (wall)";
}

void GPURecoWorkflowSpec::doCalibUpdates(o2::framework::ProcessingContext& pc, calibObjectStruct& oldCalibObjects)
{
  GPUCalibObjectsConst newCalibObjects;
  GPUNewCalibValues newCalibValues;
  // check for updates of TPC calibration objects
  bool needCalibUpdate = false;
  if (mGRPGeomUpdated) {
    mGRPGeomUpdated = false;
    needCalibUpdate = true;

    if (!mITSGeometryCreated) {
      o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
      geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
      mITSGeometryCreated = true;
    }

    if (mAutoSolenoidBz) {
      newCalibValues.newSolenoidField = true;
      newCalibValues.solenoidField = mConfig->configGRP.solenoidBz = (5.00668f / 30000.f) * GRPGeomHelper::instance().getGRPMagField()->getL3Current();
    }
    LOG(info) << "Updating solenoid field " << newCalibValues.solenoidField;
    if (mAutoContinuousMaxTimeBin) {
      mConfig->configGRP.continuousMaxTimeBin = (mTFSettings->nHBFPerTF * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
      newCalibValues.newContinuousMaxTimeBin = true;
      newCalibValues.continuousMaxTimeBin = mConfig->configGRP.continuousMaxTimeBin;
      LOG(info) << "Updating max time bin " << newCalibValues.continuousMaxTimeBin;
    }

    if (!mPropagatorInstanceCreated) {
      newCalibObjects.o2Propagator = mConfig->configCalib.o2Propagator = Propagator::Instance();
      mPropagatorInstanceCreated = true;
    }

    if (!mMatLUTCreated) {
      if (mConfParam->matLUTFile.size() == 0) {
        newCalibObjects.matLUT = GRPGeomHelper::instance().getMatLUT();
        LOG(info) << "Loaded material budget lookup table";
      }
      mMatLUTCreated = true;
    }
    if (!mTRDGeometryCreated) {
      if (mSpecConfig.readTRDtracklets) {
        auto gm = o2::trd::Geometry::instance();
        gm->createPadPlaneArray();
        gm->createClusterMatrixArray();
        mTRDGeometry = std::make_unique<o2::trd::GeometryFlat>(*gm);
        newCalibObjects.trdGeometry = mConfig->configCalib.trdGeometry = mTRDGeometry.get();
        LOG(info) << "Loaded TRD geometry";
      }
      mTRDGeometryCreated = true;
    }
  }
  needCalibUpdate = fetchCalibsCCDBTPC(pc, newCalibObjects, oldCalibObjects) || needCalibUpdate;
  if (mSpecConfig.runITSTracking) {
    needCalibUpdate = fetchCalibsCCDBITS(pc) || needCalibUpdate;
  }
  if (needCalibUpdate) {
    LOG(info) << "Updating GPUReconstruction calibration objects";
    mGPUReco->UpdateCalibration(newCalibObjects, newCalibValues);
  }
}

Options GPURecoWorkflowSpec::options()
{
  Options opts;
  if (mSpecConfig.enableDoublePipeline) {
    bool send = mSpecConfig.enableDoublePipeline == 2;
    char* o2jobid = getenv("O2JOBID");
    char* numaid = getenv("NUMAID");
    int chanid = o2jobid ? atoi(o2jobid) : (numaid ? atoi(numaid) : 0);
    std::string chan = std::string("name=gpu-prepare-channel,type=") + (send ? "push" : "pull") + ",method=" + (send ? "connect" : "bind") + ",address=ipc://@gpu-prepare-channel-" + std::to_string(chanid) + "-{timeslice0},transport=shmem,rateLogging=0";
    opts.emplace_back(o2::framework::ConfigParamSpec{"channel-config", o2::framework::VariantType::String, chan, {"Out-of-band channel config"}});
  }
  if (mSpecConfig.enableDoublePipeline == 2) {
    return opts;
  }
  if (mSpecConfig.outputTracks) {
    o2::tpc::CorrectionMapsLoader::addOptions(opts);
  }
  return opts;
}

Inputs GPURecoWorkflowSpec::inputs()
{
  Inputs inputs;
  if (mSpecConfig.zsDecoder) {
    // All ZS raw data is published with subspec 0 by the o2-raw-file-reader-workflow and DataDistribution
    // creates subspec fom CRU and endpoint id, we create one single input route subscribing to all TPC/RAWDATA
    inputs.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Timeframe});
    if (mSpecConfig.askDISTSTF) {
      inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
    }
  }
  if (mSpecConfig.enableDoublePipeline == 2) {
    if (!mSpecConfig.zsDecoder) {
      LOG(fatal) << "Double pipeline mode can only work with zsraw input";
    }
    return inputs;
  } else if (mSpecConfig.enableDoublePipeline == 1) {
    inputs.emplace_back("pipelineprepare", gDataOriginGPU, "PIPELINEPREPARE", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.outputTracks) {
    // loading calibration objects from the CCDB
    inputs.emplace_back("tpcgain", gDataOriginTPC, "PADGAINFULL", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalPadGainFull)));
    inputs.emplace_back("tpcgainresidual", gDataOriginTPC, "PADGAINRESIDUAL", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalPadGainResidual)));
    inputs.emplace_back("tpctimegain", gDataOriginTPC, "TIMEGAIN", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalTimeGain)));
    inputs.emplace_back("tpctopologygain", gDataOriginTPC, "TOPOLOGYGAIN", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalTopologyGain)));
    inputs.emplace_back("tpcthreshold", gDataOriginTPC, "PADTHRESHOLD", 0, Lifetime::Condition, ccdbParamSpec("TPC/Config/FEEPad"));
    o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
    Options optsDummy;
    mCalibObjects.mFastTransformHelper->requestCCDBInputs(inputs, optsDummy, mSpecConfig.requireCTPLumi, mSpecConfig.lumiScaleMode); // option filled here is lost
  }
  if (mSpecConfig.decompressTPC) {
    inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, mSpecConfig.decompressTPCFromROOT ? o2::header::DataDescription("COMPCLUSTERS") : o2::header::DataDescription("COMPCLUSTERSFLAT")}, Lifetime::Timeframe});
  } else if (mSpecConfig.caClusterer) {
    // if the output type are tracks, then the input spec for the gain map is already defined
    if (!mSpecConfig.outputTracks) {
      inputs.emplace_back("tpcgain", gDataOriginTPC, "PADGAINFULL", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalPadGainFull)));
    }

    // We accept digits and MC labels also if we run on ZS Raw data, since they are needed for MC label propagation
    if ((!mSpecConfig.zsOnTheFly || mSpecConfig.processMC) && !mSpecConfig.zsDecoder) {
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe});
      mPolicyData->emplace_back(o2::framework::InputSpec{"digits", o2::framework::ConcreteDataTypeMatcher{"TPC", "DIGITS"}});
    }
  } else if (mSpecConfig.runTPCTracking) {
    inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe});
    mPolicyData->emplace_back(o2::framework::InputSpec{"clusters", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}});
  }
  if (mSpecConfig.processMC) {
    if (mSpecConfig.caClusterer) {
      if (!mSpecConfig.zsDecoder) {
        inputs.emplace_back(InputSpec{"mclblin", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITSMCTR"}, Lifetime::Timeframe});
        mPolicyData->emplace_back(o2::framework::InputSpec{"digitsmc", o2::framework::ConcreteDataTypeMatcher{"TPC", "DIGITSMCTR"}});
      }
    } else {
      inputs.emplace_back(InputSpec{"mclblin", ConcreteDataTypeMatcher{gDataOriginTPC, "CLNATIVEMCLBL"}, Lifetime::Timeframe});
      mPolicyData->emplace_back(o2::framework::InputSpec{"clustersmc", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLNATIVEMCLBL"}});
    }
  }

  if (mSpecConfig.zsOnTheFly) {
    inputs.emplace_back(InputSpec{"zsinput", ConcreteDataTypeMatcher{"TPC", "TPCZS"}, Lifetime::Timeframe});
    inputs.emplace_back(InputSpec{"zsinputsizes", ConcreteDataTypeMatcher{"TPC", "ZSSIZES"}, Lifetime::Timeframe});
  }
  if (mSpecConfig.readTRDtracklets) {
    inputs.emplace_back("trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe);
    inputs.emplace_back("trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe);
    inputs.emplace_back("trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);
    inputs.emplace_back("trdtrigrecmask", o2::header::gDataOriginTRD, "TRIGRECMASK", 0, Lifetime::Timeframe);
  }

  if (mSpecConfig.runITSTracking) {
    inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
    if (mSpecConfig.itsTriggerType == 1) {
      inputs.emplace_back("phystrig", "ITS", "PHYSTRIG", 0, Lifetime::Timeframe);
    } else if (mSpecConfig.itsTriggerType == 2) {
      inputs.emplace_back("phystrig", "TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
    }
    inputs.emplace_back("itscldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
    inputs.emplace_back("itsalppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));

    if (mSpecConfig.itsOverrBeamEst) {
      inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
    }
    if (mSpecConfig.processMC) {
      inputs.emplace_back("itsmclabels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
      inputs.emplace_back("ITSMC2ROframes", "ITS", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    }
  }

  return inputs;
};

Outputs GPURecoWorkflowSpec::outputs()
{
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  std::vector<OutputSpec> outputSpecs;
  if (mSpecConfig.enableDoublePipeline == 2) {
    outputSpecs.emplace_back(gDataOriginGPU, "PIPELINEPREPARE", 0, Lifetime::Timeframe);
    return outputSpecs;
  }
  if (mSpecConfig.outputTracks) {
    outputSpecs.emplace_back(gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.processMC && mSpecConfig.outputTracks) {
    outputSpecs.emplace_back(gDataOriginTPC, "TRACKSMCLBL", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.outputCompClusters) {
    outputSpecs.emplace_back(gDataOriginTPC, "COMPCLUSTERS", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.outputCompClustersFlat) {
    outputSpecs.emplace_back(gDataOriginTPC, "COMPCLUSTERSFLAT", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.outputCAClusters) {
    for (auto const& sector : mTPCSectors) {
      mClusterOutputIds.emplace_back(sector);
    }
    if (mSpecConfig.sendClustersPerSector) {
      outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVETMP", NSectors, Lifetime::Timeframe); // Dummy buffer the TPC tracker writes the inital linear clusters to
      for (const auto sector : mTPCSectors) {
        outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", sector, Lifetime::Timeframe);
      }
    } else {
      outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", NSectors, Lifetime::Timeframe);
    }
    if (mSpecConfig.processMC) {
      if (mSpecConfig.sendClustersPerSector) {
        for (const auto sector : mTPCSectors) {
          outputSpecs.emplace_back(gDataOriginTPC, "CLNATIVEMCLBL", sector, Lifetime::Timeframe);
        }
      } else {
        outputSpecs.emplace_back(gDataOriginTPC, "CLNATIVEMCLBL", NSectors, Lifetime::Timeframe);
      }
    }
  }
  if (mSpecConfig.outputSharedClusterMap) {
    outputSpecs.emplace_back(gDataOriginTPC, "CLSHAREDMAP", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.tpcTriggerHandling) {
    outputSpecs.emplace_back(gDataOriginTPC, "TRIGGERWORDS", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.outputQA) {
    outputSpecs.emplace_back(gDataOriginTPC, "TRACKINGQA", 0, Lifetime::Timeframe);
  }
  if (mSpecConfig.outputErrorQA) {
    outputSpecs.emplace_back(gDataOriginGPU, "ERRORQA", 0, Lifetime::Timeframe);
  }

  if (mSpecConfig.runITSTracking) {
    outputSpecs.emplace_back(gDataOriginITS, "TRACKS", 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginITS, "TRACKCLSID", 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginITS, "ITSTrackROF", 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginITS, "VERTICES", 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginITS, "VERTICESROF", 0, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginITS, "IRFRAMES", 0, Lifetime::Timeframe);

    if (mSpecConfig.processMC) {
      outputSpecs.emplace_back(gDataOriginITS, "VERTICESMCTR", 0, Lifetime::Timeframe);
      outputSpecs.emplace_back(gDataOriginITS, "TRACKSMCTR", 0, Lifetime::Timeframe);
      outputSpecs.emplace_back(gDataOriginITS, "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
    }
  }

  return outputSpecs;
};

void GPURecoWorkflowSpec::deinitialize()
{
  ExitPipeline();
  mQA.reset(nullptr);
  mDisplayFrontend.reset(nullptr);
  mGPUReco.reset(nullptr);
}

} // namespace o2::gpu
