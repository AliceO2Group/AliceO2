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
/// @author Matthias Richter
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
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/WorkflowHelper.h"
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
#include "CommonUtils/VerbosityConfig.h"
#include <filesystem>
#include <memory> // for make_shared
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>
#include "GPUReconstructionConvert.h"
#include "DetectorsRaw/RDHUtils.h"
#include <TStopwatch.h>
#include <TObjArray.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH1D.h>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;
using namespace o2::tpc;

namespace o2::gpu
{

GPURecoWorkflowSpec::GPURecoWorkflowSpec(GPURecoWorkflowSpec::CompletionPolicyData* policyData, Config const& specconfig, std::vector<int> const& tpcsectors, unsigned long tpcSectorMask, std::shared_ptr<o2::base::GRPGeomRequest>& ggr) : o2::framework::Task(), mPolicyData(policyData), mTPCSectorMask(tpcSectorMask), mTPCSectors(tpcsectors), mSpecConfig(specconfig), mGGR(ggr)
{
  if (mSpecConfig.outputCAClusters && !mSpecConfig.caClusterer && !mSpecConfig.decompressTPC) {
    throw std::runtime_error("inconsistent configuration: cluster output is only possible if CA clusterer is activated");
  }

  mConfig.reset(new GPUO2InterfaceConfiguration);
  mConfParam.reset(new GPUSettingsO2);
  mTFSettings.reset(new GPUSettingsTF);
  mTimer.reset(new TStopwatch);
}

GPURecoWorkflowSpec::~GPURecoWorkflowSpec() = default;

void GPURecoWorkflowSpec::init(InitContext& ic)
{
  GRPGeomHelper::instance().setRequest(mGGR);
  GPUO2InterfaceConfiguration& config = *mConfig.get();
  {
    mParser = std::make_unique<o2::algorithm::ForwardParser<ClusterGroupHeader>>();
    mTracker = std::make_unique<GPUO2Interface>();

    // Create configuration object and fill settings
    mConfig->configGRP.solenoidBz = 0;
    mTFSettings->hasSimStartOrbit = 1;
    auto& hbfu = o2::raw::HBFUtils::Instance();
    mTFSettings->simStartOrbit = hbfu.getFirstIRofTF(o2::InteractionRecord(0, hbfu.orbitFirstSampled)).orbit;

    *mConfParam = mConfig->ReadConfigurableParam();
    mConfig->configInterface.dumpEvents = mConfParam->dump;
    if (mConfParam->display) {
      mDisplayFrontend.reset(GPUDisplayFrontendInterface::getFrontend(mConfig->configDisplay.displayFrontend.c_str()));
      mConfig->configProcessing.eventDisplay = mDisplayFrontend.get();
      if (mConfig->configProcessing.eventDisplay != nullptr) {
        LOG(info) << "Event display enabled";
      } else {
        throw std::runtime_error("GPU Event Display frontend could not be created!");
      }
    }

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
    if (mSpecConfig.outputSharedClusterMap) {
      mConfig->configProcessing.outputSharedClusterMap = true;
    }
    mConfig->configProcessing.createO2Output = mSpecConfig.outputTracks ? 2 : 0; // Skip GPU-formatted output if QA is not requested

    if (mConfParam->transformationFile.size() || mConfParam->transformationSCFile.size()) {
      LOG(fatal) << "Deprecated configurable param options GPU_global.transformationFile or transformationSCFile used\n"
                 << "Instead, link the corresponding file as <somedir>/TPC/Calib/CorrectionMap/snapshot.root and use it via\n"
                 << "--condition-remap file://<somdir>=TPC/Calib/CorrectionMap option";
    }

    // initialize TPC calib objects
    initFunctionTPC();
    mConfig->configCalib.fastTransform = mFastTransformHelper->getCorrMap();
    mConfig->configCalib.fastTransformRef = mFastTransformHelper->getCorrMapRef();
    mConfig->configCalib.fastTransformHelper = mFastTransformHelper.get();
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
    if (mTracker->Initialize(config) != 0) {
      throw std::invalid_argument("GPU Reconstruction initialization failed");
    }
    if (mSpecConfig.outputQA) {
      mQA = std::make_unique<GPUO2InterfaceQA>(mConfig.get());
    }
    mTimer->Stop();
    mTimer->Reset();
  }

  auto& callbacks = ic.services().get<CallbackService>();
  callbacks.set(CallbackService::Id::RegionInfoCallback, [this](fair::mq::RegionInfo const& info) {
    if (info.size == 0) {
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
    if (mTracker->registerMemoryForGPU(info.ptr, info.size)) {
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
}

void GPURecoWorkflowSpec::stop()
{
  LOGF(info, "TPC CATracker total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer->CpuTime(), mTimer->RealTime(), mTimer->Counter() - 1);
}

void GPURecoWorkflowSpec::endOfStream(EndOfStreamContext& ec)
{
}

void GPURecoWorkflowSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  finaliseCCDBTPC(matcher, obj);
  if (GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    mGRPGeomUpdated = true;
    return;
  }
}

void GPURecoWorkflowSpec::run(ProcessingContext& pc)
{
  constexpr static size_t NSectors = Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;

  auto cput = mTimer->CpuTime();
  auto realt = mTimer->RealTime();
  mTimer->Start(false);

  GRPGeomHelper::instance().checkUpdates(pc);
  if (mConfParam->tpcTriggeredMode ^ !GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::TPC)) {
    LOG(fatal) << "configKeyValue tpcTriggeredMode does not match GRP isDetContinuousReadOut(TPC) setting";
  }

  std::vector<gsl::span<const char>> inputs;

  const CompressedClustersFlat* pCompClustersFlat = nullptr;
  size_t compClustersFlatDummyMemory[(sizeof(CompressedClustersFlat) + sizeof(size_t) - 1) / sizeof(size_t)];
  CompressedClustersFlat& compClustersFlatDummy = reinterpret_cast<CompressedClustersFlat&>(compClustersFlatDummyMemory);
  CompressedClusters compClustersDummy;
  o2::gpu::GPUTrackingInOutZS tpcZS;
  std::vector<const void*> tpcZSmetaPointers[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  std::vector<unsigned int> tpcZSmetaSizes[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  const void** tpcZSmetaPointers2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  const unsigned int* tpcZSmetaSizes2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  std::array<unsigned int, NEndpoints * NSectors> tpcZSonTheFlySizes;
  gsl::span<const ZeroSuppressedContainer8kb> inputZS;

  bool getWorkflowTPCInput_clusters = false, getWorkflowTPCInput_mc = false, getWorkflowTPCInput_digits = false;

  // unsigned int totalZSPages = 0;
  if (mSpecConfig.processMC) {
    getWorkflowTPCInput_mc = true;
  }
  if (!mSpecConfig.decompressTPC && !mSpecConfig.caClusterer) {
    getWorkflowTPCInput_clusters = true;
  }
  if (!mSpecConfig.decompressTPC && mSpecConfig.caClusterer && ((!mSpecConfig.zsOnTheFly || mSpecConfig.processMC) && !mSpecConfig.zsDecoder)) {
    getWorkflowTPCInput_digits = true;
  }

  if (mSpecConfig.zsOnTheFly || mSpecConfig.zsDecoder) {
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        tpcZSmetaPointers[i][j].clear();
        tpcZSmetaSizes[i][j].clear();
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
      inputZS = pc.inputs().get<gsl::span<ZeroSuppressedContainer8kb>>(ref);
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
    auto insertPages = [&tpcZSmetaPointers, &tpcZSmetaSizes](const char* ptr, size_t count, uint32_t subSpec) -> void {
      if (subSpec == 0xdeadbeef) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        static int contDeadBeef = 0;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [TPC/RAWDATA/0xdeadbeef] assuming no payload for all links in this TF{}", contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        return;
      }
      int rawcru = rdh_utils::getCRU(ptr);
      int rawendpoint = rdh_utils::getEndPoint(ptr);
      if ((rdh_utils::getLink(ptr) == rdh_utils::UserLogicLinkID || rdh_utils::getLink(ptr) == rdh_utils::ILBZSLinkID || rdh_utils::getLink(ptr) == rdh_utils::DLBZSLinkID) && o2::raw::RDHUtils::getDetectorField(ptr) == 2) {
        tpcZSmetaPointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(ptr);
        tpcZSmetaSizes[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(count);
      }
    };
    // the sequencer processes all inputs matching the filter and finds sequences of consecutive
    // raw pages based on the matcher predicate, and calls the inserter for each sequence
    DPLRawPageSequencer(pc.inputs(), filter)(isSameRdh, insertPages);

    int totalCount = 0;
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        tpcZSmetaPointers2[i][j] = tpcZSmetaPointers[i][j].data();
        tpcZSmetaSizes2[i][j] = tpcZSmetaSizes[i][j].data();
        tpcZS.slice[i].zsPtr[j] = tpcZSmetaPointers2[i][j];
        tpcZS.slice[i].nZSPtr[j] = tpcZSmetaSizes2[i][j];
        tpcZS.slice[i].count[j] = tpcZSmetaPointers[i][j].size();
        totalCount += tpcZSmetaPointers[i][j].size();
      }
    }
  } else if (mSpecConfig.decompressTPC) {
    if (mSpecConfig.decompressTPCFromROOT) {
      compClustersDummy = *pc.inputs().get<CompressedClustersROOT*>("input");
      compClustersFlatDummy.setForward(&compClustersDummy);
      pCompClustersFlat = &compClustersFlatDummy;
    } else {
      pCompClustersFlat = pc.inputs().get<CompressedClustersFlat*>("input").get();
    }
  } else if (!mSpecConfig.zsOnTheFly) {
    if (mVerbosity) {
      LOGF(info, "running tracking for sector(s) 0x%09x", mTPCSectorMask);
    }
  }

  const auto& inputsClustersDigits = getWorkflowTPCInput(pc, mVerbosity, getWorkflowTPCInput_mc, getWorkflowTPCInput_clusters, mTPCSectorMask, getWorkflowTPCInput_digits);
  GPUTrackingInOutPointers ptrs;

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

  // a byte size resizable vector object, the DataAllocator returns reference to internal object
  // initialize optional pointer to the vector object
  TPCSectorHeader clusterOutputSectorHeader{0};
  if (mClusterOutputIds.size() > 0) {
    clusterOutputSectorHeader.sectorBits = mTPCSectorMask;
    // subspecs [0, NSectors - 1] are used to identify sector data, we use NSectors to indicate the full TPC
    clusterOutputSectorHeader.activeSectors = mTPCSectorMask;
  }

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
  setOutputAllocator("CLUSTERNATIVE", mClusterOutputIds.size() > 0, outputRegions.clustersNative, std::make_tuple(gDataOriginTPC, mSpecConfig.sendClustersPerSector ? (DataDescription) "CLUSTERNATIVETMP" : (DataDescription) "CLUSTERNATIVE", NSectors, Lifetime::Timeframe, clusterOutputSectorHeader), sizeof(ClusterCountIndex));
  setOutputAllocator("CLSHAREDMAP", mSpecConfig.outputSharedClusterMap, outputRegions.sharedClusterMap, std::make_tuple(gDataOriginTPC, (DataDescription) "CLSHAREDMAP", 0));
  setOutputAllocator("TRACKS", mSpecConfig.outputTracks, outputRegions.tpcTracksO2, std::make_tuple(gDataOriginTPC, (DataDescription) "TRACKS", 0));
  setOutputAllocator("CLUSREFS", mSpecConfig.outputTracks, outputRegions.tpcTracksO2ClusRefs, std::make_tuple(gDataOriginTPC, (DataDescription) "CLUSREFS", 0));
  setOutputAllocator("TRACKSMCLBL", mSpecConfig.outputTracks && mSpecConfig.processMC, outputRegions.tpcTracksO2Labels, std::make_tuple(gDataOriginTPC, (DataDescription) "TRACKSMCLBL", 0));
  ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clustersMCBuffer;
  if (mSpecConfig.processMC && mSpecConfig.caClusterer) {
    outputRegions.clusterLabels.allocator = [&clustersMCBuffer](size_t size) -> void* { return &clustersMCBuffer; };
  }

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

  if ((int)(ptrs.tpcZS != nullptr) + (int)(ptrs.tpcPackedDigits != nullptr && (ptrs.tpcZS == nullptr || ptrs.tpcPackedDigits->tpcDigitsMC == nullptr)) + (int)(ptrs.clustersNative != nullptr) + (int)(ptrs.tpcCompressedClusters != nullptr) != 1) {
    throw std::runtime_error("Invalid input for gpu tracking");
  }

  const auto& holdData = TPCTrackingDigitsPreCheck::runPrecheck(&ptrs, mConfig.get());

  doCalibUpdates(pc);

  int retVal = mTracker->RunTracking(&ptrs, &outputRegions);

  // setting TPC calibration objects
  storeUpdatedCalibsTPCPtrs();

  mTracker->Clear(false);

  if (mConfParam->dump == 2) {
    return;
  }
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

  std::unique_ptr<ClusterNativeAccess> tmpEmptyClNative;
  if (createEmptyOutput) {
    memset(&ptrs, 0, sizeof(ptrs));
    for (unsigned int i = 0; i < outputRegions.count(); i++) {
      if (outputBuffers[i].first) {
        size_t toSize = 0;
        if (i == outputRegions.getIndex(outputRegions.compressedClusters)) {
          toSize = sizeof(*ptrs.tpcCompressedClusters);
        } else if (i == outputRegions.getIndex(outputRegions.clustersNative)) {
          toSize = sizeof(ClusterCountIndex);
        }
        outputBuffers[i].first->get().resize(toSize);
        outputBuffers[i].second = outputBuffers[i].first->get().data();
        if (toSize) {
          memset(outputBuffers[i].second, 0, toSize);
        }
      }
    }
    tmpEmptyClNative = std::make_unique<ClusterNativeAccess>();
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

    if (mClusterOutputIds.size() > 0 && (void*)ptrs.clustersNative->clustersLinear != (void*)(outputBuffers[outputRegions.getIndex(outputRegions.clustersNative)].second + sizeof(ClusterCountIndex))) {
      throw std::runtime_error("cluster native output ptrs out of sync"); // sanity check
    }
  }

  if (mConfig->configWorkflow.outputs.isSet(GPUDataTypes::InOutType::TPCMergedTracks)) {
    LOG(info) << "found " << ptrs.nOutputTracksTPCO2 << " track(s)";
  }

  if (mSpecConfig.outputCompClusters) {
    CompressedClustersROOT compressedClusters = *ptrs.tpcCompressedClusters;
    pc.outputs().snapshot(Output{gDataOriginTPC, "COMPCLUSTERS", 0}, ROOTSerialized<CompressedClustersROOT const>(compressedClusters));
  }

  if (mClusterOutputIds.size() > 0) {
    ClusterNativeAccess const& accessIndex = *ptrs.clustersNative;
    if (mSpecConfig.sendClustersPerSector) {
      // Clusters are shipped by sector, we are copying into per-sector buffers (anyway only for ROOT output)
      for (unsigned int i = 0; i < NSectors; i++) {
        if (mTPCSectorMask & (1ul << i)) {
          DataHeader::SubSpecificationType subspec = i;
          clusterOutputSectorHeader.sectorBits = (1ul << i);
          char* buffer = pc.outputs().make<char>({gDataOriginTPC, "CLUSTERNATIVE", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, accessIndex.nClustersSector[i] * sizeof(*accessIndex.clustersLinear) + sizeof(ClusterCountIndex)).data();
          ClusterCountIndex* outIndex = reinterpret_cast<ClusterCountIndex*>(buffer);
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
      ClusterCountIndex* outIndex = reinterpret_cast<ClusterCountIndex*>(outputBuffers[outputRegions.getIndex(outputRegions.clustersNative)].second);
      static_assert(sizeof(ClusterCountIndex) == sizeof(accessIndex.nClusters));
      memcpy(outIndex, &accessIndex.nClusters[0][0], sizeof(ClusterCountIndex));
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
    if (sendQAOutput) {
      mQA->postprocessExternal(copy1, copy2, copy3, out, mQATaskMask ? mQATaskMask : -1);
    }
    pc.outputs().snapshot({gDataOriginTPC, "TRACKINGQA", 0, Lifetime::Timeframe}, out);
    if (sendQAOutput) {
      mQA->cleanup();
    }
  }
  mTimer->Stop();
  LOG(info) << "GPU Reoncstruction time for this TF " << mTimer->CpuTime() - cput << " s (cpu), " << mTimer->RealTime() - realt << " s (wall)";
}

void GPURecoWorkflowSpec::doCalibUpdates(o2::framework::ProcessingContext& pc)
{
  GPUCalibObjectsConst newCalibObjects;
  GPUNewCalibValues newCalibValues;
  // check for updates of TPC calibration objects
  bool needCalibUpdate = fetchCalibsCCDBTPC(pc, newCalibObjects);
  if (mGRPGeomUpdated) {
    mGRPGeomUpdated = false;
    needCalibUpdate = true;

    newCalibValues.newSolenoidField = true;
    newCalibValues.solenoidField = mConfig->configGRP.solenoidBz = (5.00668f / 30000.f) * GRPGeomHelper::instance().getGRPMagField()->getL3Current();
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

    if (!mGeometryCreated) {
      if (mConfParam->matLUTFile.size() == 0) {
        newCalibObjects.matLUT = GRPGeomHelper::instance().getMatLUT();
        LOG(info) << "Loaded material budget lookup table";
      }
      if (mSpecConfig.readTRDtracklets) {
        auto gm = o2::trd::Geometry::instance();
        gm->createPadPlaneArray();
        gm->createClusterMatrixArray();
        mTRDGeometry = std::make_unique<o2::trd::GeometryFlat>(*gm);
        newCalibObjects.trdGeometry = mConfig->configCalib.trdGeometry = mTRDGeometry.get();
        LOG(info) << "Loaded TRD geometry";
      }
      mGeometryCreated = true;
    }
  }
  if (needCalibUpdate) {
    LOG(info) << "Updating GPUReconstruction calibration objects";
    mTracker->UpdateCalibration(newCalibObjects, newCalibValues);
  }
}

Inputs GPURecoWorkflowSpec::inputs()
{
  Inputs inputs;
  if (mSpecConfig.outputTracks) {
    // loading calibration objects from the CCDB
    inputs.emplace_back("tpcgain", gDataOriginTPC, "PADGAINFULL", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalPadGainFull)));
    inputs.emplace_back("tpcgainresidual", gDataOriginTPC, "PADGAINRESIDUAL", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalPadGainResidual)));
    inputs.emplace_back("tpctimegain", gDataOriginTPC, "TIMEGAIN", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalTimeGain)));
    inputs.emplace_back("tpctopologygain", gDataOriginTPC, "TOPOLOGYGAIN", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalTopologyGain)));
    inputs.emplace_back("tpcthreshold", gDataOriginTPC, "PADTHRESHOLD", 0, Lifetime::Condition, ccdbParamSpec("TPC/Config/FEEPad"));
    o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
    o2::tpc::CorrectionMapsLoader::requestCCDBInputs(inputs);
  }
  if (mSpecConfig.decompressTPC) {
    inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, mSpecConfig.decompressTPCFromROOT ? o2::header::DataDescription("COMPCLUSTERS") : o2::header::DataDescription("COMPCLUSTERSFLAT")}, Lifetime::Timeframe});
  } else if (mSpecConfig.caClusterer) {
    // if the output type are tracks, then the input spec for the gain map is already defined
    if (!mSpecConfig.outputTracks) {
      inputs.emplace_back("tpcgain", gDataOriginTPC, "PADGAINFULL", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalPadGainFull)));
    }

    // We accept digits and MC labels also if we run on ZS Raw data, since they are needed for MC label propagation
    if ((!mSpecConfig.zsOnTheFly || mSpecConfig.processMC) && !mSpecConfig.zsDecoder) {
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe});
      mPolicyData->emplace_back(o2::framework::InputSpec{"digits", o2::framework::ConcreteDataTypeMatcher{"TPC", "DIGITS"}});
    }
  } else {
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

  if (mSpecConfig.zsDecoder) {
    // All ZS raw data is published with subspec 0 by the o2-raw-file-reader-workflow and DataDistribution
    // creates subspec fom CRU and endpoint id, we create one single input route subscribing to all TPC/RAWDATA
    inputs.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Optional});
    if (mSpecConfig.askDISTSTF) {
      inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
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
  return inputs;
};

Outputs GPURecoWorkflowSpec::outputs()
{
  constexpr static size_t NSectors = Sector::MAXSECTOR;
  std::vector<OutputSpec> outputSpecs;
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
    outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", mSpecConfig.sendClustersPerSector ? 0 : NSectors, Lifetime::Timeframe);
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
  if (mSpecConfig.outputQA) {
    outputSpecs.emplace_back(gDataOriginTPC, "TRACKINGQA", 0, Lifetime::Timeframe);
  }
  return outputSpecs;
};

void GPURecoWorkflowSpec::initFunctionTPC()
{
  mdEdxCalibContainer.reset(new o2::tpc::CalibdEdxContainer());
  mTPCVDriftHelper.reset(new o2::tpc::VDriftHelper());
  mFastTransformHelper.reset(new o2::tpc::CorrectionMapsLoader());
  mFastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));
  mFastTransformRef = std::move(TPCFastTransformHelperO2::instance()->create(0));
  mFastTransformHelper->setCorrMap(mFastTransform.get()); // just to reserve the space
  mFastTransformHelper->setCorrMapRef(mFastTransformRef.get());

  if (mConfParam->dEdxDisableTopologyPol) {
    LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);
  }

  if (mConfParam->dEdxDisableThresholdMap) {
    LOGP(info, "Disabling loading of threshold map from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap);
  }

  if (mConfParam->dEdxDisableGainMap) {
    LOGP(info, "Disabling loading of gain map from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalGainMap);
  }

  if (mConfParam->dEdxDisableResidualGainMap) {
    LOGP(info, "Disabling loading of residual gain map from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalResidualGainMap);
  }

  if (mConfParam->dEdxDisableResidualGain) {
    LOGP(info, "Disabling loading of residual gain calibration from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain);
  }

  if (mConfParam->dEdxUseFullGainMap) {
    LOGP(info, "Using the full gain map for correcting the cluster charge during calculation of the dE/dx");
    mdEdxCalibContainer->setUsageOfFullGainMap(true);
  }

  if (mConfParam->gainCalibDisableCCDB) {
    LOGP(info, "Disabling loading the TPC pad gain calibration from the CCDB");
    mUpdateGainMapCCDB = false;
  }

  // load from file
  if (!mConfParam->dEdxPolTopologyCorrFile.empty() || !mConfParam->dEdxCorrFile.empty() || !mConfParam->dEdxSplineTopologyCorrFile.empty()) {
    if (!mConfParam->dEdxPolTopologyCorrFile.empty()) {
      LOGP(info, "Loading dE/dx polynomial track topology correction from file: {}", mConfParam->dEdxPolTopologyCorrFile);
      mdEdxCalibContainer->loadPolTopologyCorrectionFromFile(mConfParam->dEdxPolTopologyCorrFile);

      LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB as it was already loaded from input file");
      mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);

      if (std::filesystem::exists(mConfParam->thresholdCalibFile)) {
        LOG(info) << "Loading tpc zero supression map from file " << mConfParam->thresholdCalibFile;
        const auto* thresholdMap = o2::tpc::utils::readCalPads(mConfParam->thresholdCalibFile, "ThresholdMap")[0];
        mdEdxCalibContainer->setZeroSupresssionThreshold(*thresholdMap);

        LOGP(info, "Disabling loading of threshold map from CCDB as it was already loaded from input file");
        mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap);
      } else {
        if (not mConfParam->thresholdCalibFile.empty()) {
          LOG(warn) << "Couldn't find tpc zero supression file " << mConfParam->thresholdCalibFile << ". Not setting any zero supression.";
        }
        LOG(info) << "Setting default zero supression map";
        mdEdxCalibContainer->setDefaultZeroSupresssionThreshold();
      }
    } else if (!mConfParam->dEdxSplineTopologyCorrFile.empty()) {
      LOGP(info, "Loading dE/dx spline track topology correction from file: {}", mConfParam->dEdxSplineTopologyCorrFile);
      mdEdxCalibContainer->loadSplineTopologyCorrectionFromFile(mConfParam->dEdxSplineTopologyCorrFile);

      LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB as splines were loaded from input file");
      mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);
    }
    if (!mConfParam->dEdxCorrFile.empty()) {
      LOGP(info, "Loading dEdx correction from file: {}", mConfParam->dEdxCorrFile);
      mdEdxCalibContainer->loadResidualCorrectionFromFile(mConfParam->dEdxCorrFile);

      LOGP(info, "Disabling loading of residual gain calibration from CCDB as it was already loaded from input file");
      mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain);
    }
  }

  if (mConfParam->dEdxPolTopologyCorrFile.empty() && mConfParam->dEdxSplineTopologyCorrFile.empty()) {
    // setting default topology correction to allocate enough memory
    LOG(info) << "Setting default dE/dx polynomial track topology correction to allocate enough memory";
    mdEdxCalibContainer->setDefaultPolTopologyCorrection();
  }

  GPUO2InterfaceConfiguration& config = *mConfig.get();
  mConfig->configCalib.dEdxCalibContainer = mdEdxCalibContainer.get();

  if (std::filesystem::exists(mConfParam->gainCalibFile)) {
    LOG(info) << "Loading tpc gain correction from file " << mConfParam->gainCalibFile;
    const auto* gainMap = o2::tpc::utils::readCalPads(mConfParam->gainCalibFile, "GainMap")[0];
    mTPCPadGainCalib = GPUO2Interface::getPadGainCalib(*gainMap);

    LOGP(info, "Disabling loading the TPC gain correction map from the CCDB as it was already loaded from input file");
    mUpdateGainMapCCDB = false;
  } else {
    if (not mConfParam->gainCalibFile.empty()) {
      LOG(warn) << "Couldn't find tpc gain correction file " << mConfParam->gainCalibFile << ". Not applying any gain correction.";
    }
    mTPCPadGainCalib = GPUO2Interface::getPadGainCalibDefault();
    mTPCPadGainCalib->getGainCorrection(30, 5, 5);
  }
  mConfig->configCalib.tpcPadGain = mTPCPadGainCalib.get();

  mTPCZSLinkMapping.reset(new TPCZSLinkMapping{tpc::Mapper::instance()});
  mConfig->configCalib.tpcZSLinkMapping = mTPCZSLinkMapping.get();
}

void GPURecoWorkflowSpec::finaliseCCDBTPC(ConcreteDataMatcher& matcher, void* obj)
{
  const CalibdEdxContainer* dEdxCalibContainer = mdEdxCalibContainer.get();

  auto copyCalibsToBuffer = [this, dEdxCalibContainer]() {
    if (!(mdEdxCalibContainerBufferNew)) {
      mdEdxCalibContainerBufferNew = std::make_unique<o2::tpc::CalibdEdxContainer>();
      mdEdxCalibContainerBufferNew->cloneFromObject(*dEdxCalibContainer, nullptr);
    }
  };

  if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADGAINFULL", 0)) {
    LOGP(info, "Updating gain map from CCDB");
    const auto* gainMap = static_cast<o2::tpc::CalDet<float>*>(obj);

    if (dEdxCalibContainer->isCorrectionCCDB(CalibsdEdx::CalGainMap) && mSpecConfig.outputTracks) {
      copyCalibsToBuffer();
      const float minGain = 0;
      const float maxGain = 2;
      mdEdxCalibContainerBufferNew.get()->setGainMap(*gainMap, minGain, maxGain);
    }

    if (mUpdateGainMapCCDB && mSpecConfig.caClusterer) {
      mTPCPadGainCalibBufferNew = GPUO2Interface::getPadGainCalib(*gainMap);
    }

  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADGAINRESIDUAL", 0)) {
    LOGP(info, "Updating residual gain map from CCDB");
    copyCalibsToBuffer();
    const auto* gainMapResidual = static_cast<std::unordered_map<string, o2::tpc::CalDet<float>>*>(obj);
    const float minResidualGain = 0.7f;
    const float maxResidualGain = 1.3f;
    mdEdxCalibContainerBufferNew.get()->setGainMapResidual(gainMapResidual->at("GainMap"), minResidualGain, maxResidualGain);
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADTHRESHOLD", 0)) {
    LOGP(info, "Updating threshold map from CCDB");
    copyCalibsToBuffer();
    const auto* thresholdMap = static_cast<std::unordered_map<string, o2::tpc::CalDet<float>>*>(obj);
    mdEdxCalibContainerBufferNew.get()->setZeroSupresssionThreshold(thresholdMap->at("ThresholdMap"));
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "TOPOLOGYGAIN", 0) && !(dEdxCalibContainer->isTopologyCorrectionSplinesSet())) {
    LOGP(info, "Updating Q topology correction from CCDB");
    copyCalibsToBuffer();
    const auto* topologyCorr = static_cast<o2::tpc::CalibdEdxTrackTopologyPolContainer*>(obj);
    CalibdEdxTrackTopologyPol calibTrackTopology;
    calibTrackTopology.setFromContainer(*topologyCorr);
    mdEdxCalibContainerBufferNew->setPolTopologyCorrection(calibTrackTopology);
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "TIMEGAIN", 0)) {
    LOGP(info, "Updating residual gain correction from CCDB");
    copyCalibsToBuffer();
    const auto* residualCorr = static_cast<o2::tpc::CalibdEdxCorrection*>(obj);
    mdEdxCalibContainerBufferNew->setResidualCorrection(*residualCorr);
  } else if (mTPCVDriftHelper->accountCCDBInputs(matcher, obj)) {
  } else if (mFastTransformHelper->accountCCDBInputs(matcher, obj)) {
  }
}

template <class T>
bool GPURecoWorkflowSpec::fetchCalibsCCDBTPC(ProcessingContext& pc, T& newCalibObjects)
{
  // update calibrations for clustering and tracking
  mMustUpdateFastTransform = false;
  if ((mSpecConfig.outputTracks || mSpecConfig.caClusterer) && !mConfParam->disableCalibUpdates) {
    const CalibdEdxContainer* dEdxCalibContainer = mdEdxCalibContainer.get();

    // this calibration is defined for clustering and tracking
    if (dEdxCalibContainer->isCorrectionCCDB(CalibsdEdx::CalGainMap) || mUpdateGainMapCCDB) {
      pc.inputs().get<o2::tpc::CalDet<float>*>("tpcgain");
    }

    // these calibrations are only defined for the tracking
    if (mSpecConfig.outputTracks) {
      // update the calibration objects in case they changed in the CCDB
      if (dEdxCalibContainer->isCorrectionCCDB(CalibsdEdx::CalThresholdMap)) {
        pc.inputs().get<std::unordered_map<std::string, o2::tpc::CalDet<float>>*>("tpcthreshold");
      }

      if (dEdxCalibContainer->isCorrectionCCDB(CalibsdEdx::CalResidualGainMap)) {
        pc.inputs().get<std::unordered_map<std::string, o2::tpc::CalDet<float>>*>("tpcgainresidual");
      }

      if (dEdxCalibContainer->isCorrectionCCDB(CalibsdEdx::CalTopologyPol)) {
        pc.inputs().get<o2::tpc::CalibdEdxTrackTopologyPolContainer*>("tpctopologygain");
      }

      if (dEdxCalibContainer->isCorrectionCCDB(CalibsdEdx::CalTimeGain)) {
        pc.inputs().get<o2::tpc::CalibdEdxCorrection*>("tpctimegain");
      }

      if (mSpecConfig.outputTracks) {
        o2::tpc::VDriftHelper::extractCCDBInputs(pc);
        o2::tpc::CorrectionMapsLoader::extractCCDBInputs(pc);
      }
      if (mTPCVDriftHelper->isUpdated() || mFastTransformHelper->isUpdated()) {
        LOGP(info, "Updating{}TPC fast transform map and/or VDrift factor of {} wrt reference {} from source {}",
             mFastTransformHelper->isUpdated() ? " new " : " old ",
             mTPCVDriftHelper->getVDriftObject().corrFact, mTPCVDriftHelper->getVDriftObject().refVDrift, mTPCVDriftHelper->getSourceName());

        if (mTPCVDriftHelper->isUpdated() || mFastTransformHelper->isUpdatedMap()) {
          mFastTransformNew.reset(new TPCFastTransform);
          mFastTransformNew->cloneFromObject(*mFastTransformHelper->getCorrMap(), nullptr);
          TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransformNew, 0, mTPCVDriftHelper->getVDriftObject().corrFact, mTPCVDriftHelper->getVDriftObject().refVDrift);
          newCalibObjects.fastTransform = mFastTransformNew.get();
        }
        if (mTPCVDriftHelper->isUpdated() || mFastTransformHelper->isUpdatedMapRef()) {
          mFastTransformRefNew.reset(new TPCFastTransform);
          mFastTransformRefNew->cloneFromObject(*mFastTransformHelper->getCorrMapRef(), nullptr);
          TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransformRefNew, 0, mTPCVDriftHelper->getVDriftObject().corrFact, mTPCVDriftHelper->getVDriftObject().refVDrift);
          newCalibObjects.fastTransformRef = mFastTransformRefNew.get();
        }
        if (mFastTransformNew || mFastTransformRefNew || mFastTransformHelper->isUpdatedLumi()) {
          mFastTransformHelperNew.reset(new o2::tpc::CorrectionMapsLoader);
          mFastTransformHelperNew->setInstLumi(mFastTransformHelper->getInstLumi());
          mFastTransformHelperNew->setMeanLumi(mFastTransformHelper->getMeanLumi());
          mFastTransformHelperNew->setCorrMap(mFastTransformNew ? mFastTransformNew.get() : mFastTransform.get());
          mFastTransformHelperNew->setCorrMapRef(mFastTransformRefNew ? mFastTransformRefNew.get() : mFastTransformRef.get());
          newCalibObjects.fastTransformHelper = mFastTransformHelperNew.get();
        }
        mMustUpdateFastTransform = true;
        mTPCVDriftHelper->acknowledgeUpdate();
        mFastTransformHelper->acknowledgeUpdate();
      }
    }

    if (mdEdxCalibContainerBufferNew) {
      newCalibObjects.dEdxCalibContainer = mdEdxCalibContainerBufferNew.get();
    }

    if (mTPCPadGainCalibBufferNew) {
      newCalibObjects.tpcPadGain = mTPCPadGainCalibBufferNew.get();
    }

    return mdEdxCalibContainerBufferNew || mTPCPadGainCalibBufferNew || mMustUpdateFastTransform;
  }
  return false;
}

void GPURecoWorkflowSpec::storeUpdatedCalibsTPCPtrs()
{
  if (mdEdxCalibContainerBufferNew) {
    mdEdxCalibContainer = std::move(mdEdxCalibContainerBufferNew);
  }

  if (mTPCPadGainCalibBufferNew) {
    mTPCPadGainCalib = std::move(mTPCPadGainCalibBufferNew);
  }

  if (mFastTransformNew) {
    mFastTransform = std::move(mFastTransformNew);
  }
  if (mFastTransformRefNew) {
    mFastTransformRef = std::move(mFastTransformRefNew);
  }
  if (mFastTransformHelperNew) {
    mFastTransformHelper = std::move(mFastTransformHelperNew);
  }
}

void GPURecoWorkflowSpec::deinitialize()
{
  mQA.reset(nullptr);
  mDisplayFrontend.reset(nullptr);
  mTracker.reset(nullptr);
}

} // namespace o2::gpu
