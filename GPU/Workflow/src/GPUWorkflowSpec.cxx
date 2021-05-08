// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCdEdxCalibrationSplines.h"
#include "DPLUtils/DPLRawParser.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "TPCBase/RDHUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceQA.h"
#include "GPUO2Interface.h"
#include "TPCPadGainCalib.h"
#include "GPUDisplayBackend.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY
#include "GPUDisplayBackendGlfw.h"
#endif
#include "DataFormatsParameters/GRPObject.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Algorithm/Parser.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/GeometryFlat.h"
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
DataProcessorSpec getGPURecoWorkflowSpec(gpuworkflow::CompletionPolicyData* policyData, gpuworkflow::Config const& specconfig, std::vector<int> const& tpcsectors, unsigned long tpcSectorMask, std::string processorName)
{
  if (specconfig.outputCAClusters && !specconfig.caClusterer && !specconfig.decompressTPC) {
    throw std::runtime_error("inconsistent configuration: cluster output is only possible if CA clusterer is activated");
  }

  static TStopwatch timer;

  constexpr static size_t NSectors = Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = 20; //TODO: get from mapper?
  using ClusterGroupParser = o2::algorithm::ForwardParser<ClusterGroupHeader>;
  struct ProcessAttributes {
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<GPUO2Interface> tracker;
    std::unique_ptr<GPUDisplayBackend> displayBackend;
    std::unique_ptr<TPCFastTransform> fastTransform;
    std::unique_ptr<TPCdEdxCalibrationSplines> dEdxSplines;
    std::unique_ptr<TPCPadGainCalib> tpcPadGainCalib;
    std::unique_ptr<o2::trd::GeometryFlat> trdGeometry;
    std::unique_ptr<GPUO2InterfaceConfiguration> config;
    int qaTaskMask = 0;
    std::unique_ptr<GPUO2InterfaceQA> qa;
    std::vector<int> clusterOutputIds;
    unsigned long outputBufferSize = 0;
    unsigned long tpcSectorMask = 0;
    unsigned int nHBFPerTF;
    int verbosity = 0;
    bool readyToQuit = false;
    bool allocateOutputOnTheFly = false;
    bool suppressOutput = false;
  };

  auto processAttributes = std::make_shared<ProcessAttributes>();
  processAttributes->tpcSectorMask = tpcSectorMask;

  auto initFunction = [processAttributes, specconfig](InitContext& ic) {
    processAttributes->config.reset(new GPUO2InterfaceConfiguration);
    GPUO2InterfaceConfiguration& config = *processAttributes->config.get();
    GPUSettingsO2 confParam;
    {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      parser = std::make_unique<ClusterGroupParser>();
      tracker = std::make_unique<GPUO2Interface>();

      // Create configuration object and fill settings
      const auto grp = o2::parameters::GRPObject::loadFrom();
      o2::base::GeometryManager::loadGeometry();
      o2::base::Propagator::initFieldFromGRP();
      if (!grp) {
        throw std::runtime_error("Failed to initialize run parameters from GRP");
      }
      config.configGRP.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
      config.configGRP.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0; // Number of timebins in timeframe if continuous, 0 otherwise
      processAttributes->nHBFPerTF = grp->getNHBFPerTF();
      LOG(INFO) << "Initializing run paramerers from GRP bz=" << config.configGRP.solenoidBz << " cont=" << grp->isDetContinuousReadOut(o2::detectors::DetID::TPC);

      confParam = config.ReadConfigurableParam();
      processAttributes->allocateOutputOnTheFly = confParam.allocateOutputOnTheFly;
      processAttributes->outputBufferSize = confParam.outputBufferSize;
      processAttributes->suppressOutput = (confParam.dump == 2);
      config.configInterface.dumpEvents = confParam.dump;
      config.configInterface.memoryBufferScaleFactor = confParam.memoryBufferScaleFactor;
      if (confParam.display) {
#ifdef GPUCA_BUILD_EVENT_DISPLAY
        processAttributes->displayBackend.reset(new GPUDisplayBackendGlfw);
        config.configProcessing.eventDisplay = processAttributes->displayBackend.get();
        LOG(INFO) << "Event display enabled";
#else
        throw std::runtime_error("Standalone Event Display not enabled at build time!");
#endif
      }

      if (config.configGRP.continuousMaxTimeBin == -1) {
        config.configGRP.continuousMaxTimeBin = (processAttributes->nHBFPerTF * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
      }
      if (config.configProcessing.deviceNum == -2) {
        int myId = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
        int idMax = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
        config.configProcessing.deviceNum = myId;
        LOG(INFO) << "GPU device number selected from pipeline id: " << myId << " / " << idMax;
      }
      if (config.configProcessing.debugLevel >= 3 && processAttributes->verbosity == 0) {
        processAttributes->verbosity = 1;
      }
      config.configProcessing.runMC = specconfig.processMC;
      if (specconfig.outputQA) {
        if (!specconfig.processMC && !config.configQA.clusterRejectionHistograms) {
          throw std::runtime_error("Need MC information to create QA plots");
        }
        if (!specconfig.processMC) {
          config.configQA.noMC = true;
        }
        config.configQA.shipToQC = true;
        if (!config.configProcessing.runQA) {
          config.configQA.enableLocalOutput = false;
          processAttributes->qaTaskMask = (specconfig.processMC ? 15 : 0) | (config.configQA.clusterRejectionHistograms ? 32 : 0);
          config.configProcessing.runQA = -processAttributes->qaTaskMask;
        }
      }
      config.configReconstruction.NWaysOuter = true;
      config.configInterface.outputToExternalBuffers = true;

      // Configure the "GPU workflow" i.e. which steps we run on the GPU (or CPU)
      config.configWorkflow.steps.set(GPUDataTypes::RecoStep::TPCConversion,
                                      GPUDataTypes::RecoStep::TPCSliceTracking,
                                      GPUDataTypes::RecoStep::TPCMerging,
                                      GPUDataTypes::RecoStep::TPCCompression);

      config.configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCdEdx, !confParam.synchronousProcessing);
      if (confParam.synchronousProcessing) {
        config.configReconstruction.useMatLUT = false;
      }

      // Alternative steps: TRDTracking | ITSTracking
      config.configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCClusters);
      // Alternative inputs: GPUDataTypes::InOutType::TRDTracklets
      config.configWorkflow.outputs.set(GPUDataTypes::InOutType::TPCMergedTracks, GPUDataTypes::InOutType::TPCCompressedClusters);
      // Alternative outputs: GPUDataTypes::InOutType::TPCSectorTracks, GPUDataTypes::InOutType::TRDTracks
      if (specconfig.caClusterer) { // Override some settings if we have raw data as input
        config.configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCRaw);
        config.configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCClusterFinding, true);
        config.configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCClusters, true);
      }
      if (specconfig.decompressTPC) {
        config.configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCCompression, false);
        config.configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCDecompression, true);
        config.configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCCompressedClusters);
        config.configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCClusters, true);
        config.configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCCompressedClusters, false);
        if (processAttributes->tpcSectorMask != 0xFFFFFFFFF) {
          throw std::invalid_argument("Cannot run TPC decompression with a sector mask");
        }
      }
      if (specconfig.outputSharedClusterMap) {
        config.configProcessing.outputSharedClusterMap = true;
      }
      config.configProcessing.createO2Output = 2; // Skip GPU-formatted output if QA is not requested

      // Create and forward data objects for TPC transformation, material LUT, ...
      if (confParam.transformationFile.size()) {
        processAttributes->fastTransform = nullptr;
        config.configCalib.fastTransform = TPCFastTransform::loadFromFile(confParam.transformationFile.c_str());
      } else {
        processAttributes->fastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));
        config.configCalib.fastTransform = processAttributes->fastTransform.get();
      }
      if (config.configCalib.fastTransform == nullptr) {
        throw std::invalid_argument("GPU workflow: initialization of the TPC transformation failed");
      }

      if (confParam.matLUTFile.size()) {
        config.configCalib.matLUT = o2::base::MatLayerCylSet::loadFromFile(confParam.matLUTFile.c_str(), "MatBud");
      }

      if (confParam.dEdxFile.size()) {
        processAttributes->dEdxSplines.reset(new TPCdEdxCalibrationSplines(confParam.dEdxFile.c_str()));
      } else {
        processAttributes->dEdxSplines.reset(new TPCdEdxCalibrationSplines);
      }
      config.configCalib.dEdxSplines = processAttributes->dEdxSplines.get();

      if (std::filesystem::exists(confParam.gainCalibFile)) {
        LOG(INFO) << "Loading tpc gain correction from file " << confParam.gainCalibFile;
        const auto* gainMap = o2::tpc::utils::readCalPads(confParam.gainCalibFile, "GainMap")[0];
        processAttributes->tpcPadGainCalib = GPUO2Interface::getPadGainCalib(*gainMap);
      } else {
        if (not confParam.gainCalibFile.empty()) {
          LOG(WARN) << "Couldn't find tpc gain correction file " << confParam.gainCalibFile << ". Not applying any gain correction.";
        }
        processAttributes->tpcPadGainCalib = GPUO2Interface::getPadGainCalibDefault();
      }
      config.configCalib.tpcPadGain = processAttributes->tpcPadGainCalib.get();

      config.configCalib.o2Propagator = Propagator::Instance();

      if (specconfig.readTRDtracklets) {
        auto gm = o2::trd::Geometry::instance();
        gm->createPadPlaneArray();
        gm->createClusterMatrixArray();
        processAttributes->trdGeometry = std::make_unique<o2::trd::GeometryFlat>(*gm);
        config.configCalib.trdGeometry = processAttributes->trdGeometry.get();
      }

      // Configuration is prepared, initialize the tracker.
      if (tracker->Initialize(config) != 0) {
        throw std::invalid_argument("GPU Reconstruction initialization failed");
      }
      if (specconfig.outputQA) {
        processAttributes->qa = std::make_unique<GPUO2InterfaceQA>(processAttributes->config.get());
      }
      timer.Stop();
      timer.Reset();
    }

    auto& callbacks = ic.services().get<CallbackService>();
    callbacks.set(CallbackService::Id::RegionInfoCallback, [&processAttributes, confParam](FairMQRegionInfo const& info) {
      if (info.size) {
        int fd = 0;
        if (confParam.mutexMemReg) {
          fd = open("/tmp/o2_gpu_memlock_mutex.lock", O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR);
          if (fd == -1) {
            throw std::runtime_error("Error opening lock file");
          }
          if (lockf(fd, F_LOCK, 0)) {
            throw std::runtime_error("Error locking file");
          }
        }
        auto& tracker = processAttributes->tracker;
        if (tracker->registerMemoryForGPU(info.ptr, info.size)) {
          throw std::runtime_error("Error registering memory for GPU");
        }
        if (confParam.mutexMemReg) {
          if (lockf(fd, F_ULOCK, 0)) {
            throw std::runtime_error("Error unlocking file");
          }
          close(fd);
        }
      }
    });

    // the callback to be set as hook at stop of processing for the framework
    auto printTiming = []() {
      LOGF(INFO, "TPC CATracker total timing: Cpu: %.3e Real: %.3e s in %d slots", timer.CpuTime(), timer.RealTime(), timer.Counter() - 1);
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, printTiming);

    auto processingFct = [processAttributes, specconfig](ProcessingContext& pc) {
      if (processAttributes->readyToQuit) {
        return;
      }
      auto cput = timer.CpuTime();
      auto realt = timer.RealTime();
      timer.Start(false);
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      auto& verbosity = processAttributes->verbosity;
      std::vector<gsl::span<const char>> inputs;

      const CompressedClustersFlat* pCompClustersFlat;
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
      o2::gpu::GPUSettingsTF tfSettings;

      bool getWorkflowTPCInput_clusters = false, getWorkflowTPCInput_mc = false, getWorkflowTPCInput_digits = false;

      // unsigned int totalZSPages = 0;
      if (specconfig.processMC) {
        getWorkflowTPCInput_mc = true;
      }
      if (!specconfig.decompressTPC && !specconfig.caClusterer) {
        getWorkflowTPCInput_clusters = true;
      }
      if (!specconfig.decompressTPC && specconfig.caClusterer && ((!specconfig.zsOnTheFly || specconfig.processMC) && !specconfig.zsDecoder)) {
        getWorkflowTPCInput_digits = true;
      }

      if (specconfig.zsOnTheFly || specconfig.zsDecoder) {
        for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
          for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
            tpcZSmetaPointers[i][j].clear();
            tpcZSmetaSizes[i][j].clear();
          }
        }
      }
      if (specconfig.zsOnTheFly) {
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
          throw std::runtime_error("TPC ZS data not received");
        }

        unsigned int offset = 0;
        for (unsigned int i = 0; i < NSectors; i++) {
          unsigned int pageSector = 0;
          for (unsigned int j = 0; j < NEndpoints; j++) {
            pageSector += tpcZSonTheFlySizes[i * NEndpoints + j];
            offset += tpcZSonTheFlySizes[i * NEndpoints + j];
          }
          if (verbosity >= 1) {
            LOG(INFO) << "GOT ZS pages FOR SECTOR " << i << " ->  pages: " << pageSector;
          }
        }
      }
      if (specconfig.zsDecoder) {
        std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          const DataHeader* dh = DataRefUtils::getHeader<DataHeader*>(ref);
          if (dh->payloadSize == 0 && dh->subSpecification == 0xDEADBEEF) {
            LOG(INFO) << "Received 0xDEADBEEF message with no RAW input, performing dummy processing on emtpry input data";
            continue; // Dummy message inserted if there is no input, just ignore
          }
          const gsl::span<const char> raw = pc.inputs().get<gsl::span<char>>(ref);
          o2::framework::RawParser parser(raw.data(), raw.size());

          const unsigned char* ptr = nullptr;
          int count = 0;
          rdh_utils::FEEIDType lastFEE = -1;
          int rawcru = 0;
          int rawendpoint = 0;
          size_t totalSize = 0;
          for (auto it = parser.begin(); it != parser.end(); it++) {
            const unsigned char* current = it.raw();
            const RAWDataHeader* rdh = (const RAWDataHeader*)current;
            if (current == nullptr || it.size() == 0 || (current - ptr) % TPCZSHDR::TPC_ZS_PAGE_SIZE || o2::raw::RDHUtils::getFEEID(*rdh) != lastFEE) {
              if (count) {
                tpcZSmetaPointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(ptr);
                tpcZSmetaSizes[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(count);
              }
              count = 0;
              lastFEE = o2::raw::RDHUtils::getFEEID(*rdh);
              rawcru = o2::raw::RDHUtils::getCRUID(*rdh);
              rawendpoint = o2::raw::RDHUtils::getEndPointID(*rdh);
              //lastFEE = int(rdh->feeId);
              //rawcru = int(rdh->cruID);
              //rawendpoint = int(rdh->endPointID);
              if (it.size() == 0 && tpcZSmetaPointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].size()) {
                ptr = nullptr;
                continue;
              }
              ptr = current;
            } else if (ptr == nullptr) {
              ptr = current;
            }
            count++;
          }
          if (count) {
            tpcZSmetaPointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(ptr);
            tpcZSmetaSizes[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(count);
          }
        }
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
        /*DPLRawParser parser(pc.inputs(), filter);
        for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
          // retrieving RDH v4
          auto const* rdh = it.get_if<RAWDataHeaderV4>();
          // retrieving the raw pointer of the page
          auto const* raw = it.raw();
          // retrieving payload pointer of the page
          auto const* payload = it.data();
          // size of payload
          size_t payloadSize = it.size();
          // offset of payload in the raw page
          size_t offset = it.offset();
          const auto* dh = it.o2DataHeader();
          unsigned long subspec = dh->subSpecification;
          printf("Test: rdh %p, raw %p, payload %p, payloadSize %lld, offset %lld, %s %s %lld\n", rdh, raw, payload, (long long int)payloadSize, (long long int)offset, dh->dataOrigin.as<std::string>().c_str(), dh->dataDescription.as<std::string>().c_str(), (long long int)dh->subSpecification);
        }*/
      } else if (specconfig.decompressTPC) {
        if (specconfig.decompressTPCFromROOT) {
          compClustersDummy = *pc.inputs().get<CompressedClustersROOT*>("input");
          compClustersFlatDummy.setForward(&compClustersDummy);
          pCompClustersFlat = &compClustersFlatDummy;
        } else {
          pCompClustersFlat = pc.inputs().get<CompressedClustersFlat*>("input").get();
        }
      } else if (!specconfig.zsOnTheFly) {
        if (verbosity) {
          LOGF(INFO, "running tracking for sector(s) 0x%09x", processAttributes->tpcSectorMask);
        }
      }

      const auto& inputsClustersDigits = getWorkflowTPCInput(pc, verbosity, getWorkflowTPCInput_mc, getWorkflowTPCInput_clusters, processAttributes->tpcSectorMask, getWorkflowTPCInput_digits);
      GPUTrackingInOutPointers ptrs;

      o2::globaltracking::RecoContainer inputTracksTRD;
      decltype(o2::trd::getRecoInputContainer(pc, &ptrs, &inputTracksTRD)) trdInputContainer;
      if (specconfig.readTRDtracklets) {
        o2::globaltracking::DataRequest dataRequestTRD;
        dataRequestTRD.requestTracks(o2::dataformats::GlobalTrackID::getSourcesMask(o2::dataformats::GlobalTrackID::NONE), false);
        inputTracksTRD.collectData(pc, dataRequestTRD);
        trdInputContainer = std::move(o2::trd::getRecoInputContainer(pc, &ptrs, &inputTracksTRD));
      }

      void* ptrEp[NSectors * NEndpoints] = {};
      bool doInputDigits = false, doInputDigitsMC = false;
      if (specconfig.decompressTPC) {
        ptrs.tpcCompressedClusters = pCompClustersFlat;
      } else if (specconfig.zsOnTheFly) {
        const unsigned long long int* buffer = reinterpret_cast<const unsigned long long int*>(&inputZS[0]);
        o2::gpu::GPUReconstructionConvert::RunZSEncoderCreateMeta(buffer, tpcZSonTheFlySizes.data(), *&ptrEp, &tpcZS);
        ptrs.tpcZS = &tpcZS;
        doInputDigits = doInputDigitsMC = specconfig.processMC;
      } else if (specconfig.zsDecoder) {
        ptrs.tpcZS = &tpcZS;
        if (specconfig.processMC) {
          throw std::runtime_error("Cannot process MC information, none available");
        }
      } else if (specconfig.caClusterer) {
        doInputDigits = true;
        doInputDigitsMC = specconfig.processMC;
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
      if (processAttributes->clusterOutputIds.size() > 0) {
        clusterOutputSectorHeader.sectorBits = processAttributes->tpcSectorMask;
        // subspecs [0, NSectors - 1] are used to identify sector data, we use NSectors to indicate the full TPC
        clusterOutputSectorHeader.activeSectors = processAttributes->tpcSectorMask;
      }

      GPUInterfaceOutputs outputRegions;
      using outputDataType = char;
      using outputBufferUninitializedVector = std::decay_t<decltype(pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(Output{"", "", 0}))>;
      using outputBufferType = std::pair<std::optional<std::reference_wrapper<outputBufferUninitializedVector>>, outputDataType*>;
      std::vector<outputBufferType> outputBuffers(GPUInterfaceOutputs::count(), {std::nullopt, nullptr});

      auto setOutputAllocator = [&specconfig, &outputBuffers, &outputRegions, &processAttributes, &pc, verbosity](const char* name, bool condition, GPUOutputControl& region, auto&& outputSpec, size_t offset = 0) {
        if (condition) {
          auto& buffer = outputBuffers[outputRegions.getIndex(region)];
          if (processAttributes->allocateOutputOnTheFly) {
            region.allocator = [name, &buffer, &pc, outputSpec = std::move(outputSpec), verbosity, offset](size_t size) -> void* {
              size += offset;
              if (verbosity) {
                LOG(INFO) << "ALLOCATING " << size << " bytes for " << std::get<DataOrigin>(outputSpec).template as<std::string>() << "/" << std::get<DataDescription>(outputSpec).template as<std::string>() << "/" << std::get<2>(outputSpec);
              }
              std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
              if (verbosity) {
                start = std::chrono::high_resolution_clock::now();
              }
              buffer.first.emplace(pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(std::make_from_tuple<Output>(outputSpec), size));
              if (verbosity) {
                end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                LOG(INFO) << "Allocation time for " << name << " (" << size << " bytes)" << ": " << elapsed_seconds.count() << "s";
              }
              return (buffer.second = buffer.first->get().data()) + offset;
            };
          } else {
            buffer.first.emplace(pc.outputs().make<DataAllocator::UninitializedVector<outputDataType>>(std::make_from_tuple<Output>(outputSpec), processAttributes->outputBufferSize));
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

      auto downSizeBufferByName = [&outputBuffers, &outputRegions, &downSizeBuffer](GPUOutputControl& region, size_t size) {
        auto& buffer = outputBuffers[outputRegions.getIndex(region)];
        downSizeBuffer(buffer, size);
      };

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

      setOutputAllocator("COMPCLUSTERSFLAT", specconfig.outputCompClustersFlat, outputRegions.compressedClusters, std::make_tuple(gDataOriginTPC, (DataDescription) "COMPCLUSTERSFLAT", 0));
      setOutputAllocator("CLUSTERNATIVE", processAttributes->clusterOutputIds.size() > 0, outputRegions.clustersNative, std::make_tuple(gDataOriginTPC, specconfig.sendClustersPerSector ? (DataDescription) "CLUSTERNATIVETMP" : (DataDescription) "CLUSTERNATIVE", NSectors, Lifetime::Timeframe, clusterOutputSectorHeader), sizeof(ClusterCountIndex));
      setOutputAllocator("CLSHAREDMAP", specconfig.outputSharedClusterMap, outputRegions.sharedClusterMap, std::make_tuple(gDataOriginTPC, (DataDescription) "CLSHAREDMAP", 0));
      setOutputAllocator("TRACKS", specconfig.outputTracks, outputRegions.tpcTracksO2, std::make_tuple(gDataOriginTPC, (DataDescription) "TRACKS", 0));
      setOutputAllocator("CLUSREFS", specconfig.outputTracks, outputRegions.tpcTracksO2ClusRefs, std::make_tuple(gDataOriginTPC, (DataDescription) "CLUSREFS", 0));
      setOutputAllocator("TRACKSMCLBL", specconfig.outputTracks && specconfig.processMC, outputRegions.tpcTracksO2Labels, std::make_tuple(gDataOriginTPC, (DataDescription) "TRACKSMCLBL", 0));
      ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clustersMCBuffer;
      if (specconfig.processMC && specconfig.caClusterer) {
        outputRegions.clusterLabels.allocator = [&clustersMCBuffer](size_t size) -> void* { return &clustersMCBuffer; };
      }

      const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
      tfSettings.tfStartOrbit = dh->firstTForbit;
      tfSettings.hasTfStartOrbit = 1;
      tfSettings.nHBFPerTF = processAttributes->nHBFPerTF;
      tfSettings.hasNHBFPerTF = 1;
      ptrs.settingsTF = &tfSettings;

      if (processAttributes->tpcSectorMask != 0xFFFFFFFFF) {
        // Clean out the unused sectors, such that if they were present by chance, they are not processed, and if the values are uninitialized, we should not crash
        for (int i = 0; i < NSectors; i++) {
          if (!(processAttributes->tpcSectorMask & (1ul << i))) {
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

      const auto& holdData = TPCTrackingDigitsPreCheck::runPrecheck(&ptrs, processAttributes->config.get());
      int retVal = tracker->RunTracking(&ptrs, &outputRegions);
      gsl::span<const o2::tpc::TrackTPC> spanOutputTracks = {ptrs.outputTracksTPCO2, ptrs.nOutputTracksTPCO2};
      gsl::span<const uint32_t> spanOutputClusRefs = {ptrs.outputClusRefsTPCO2, ptrs.nOutputClusRefsTPCO2};
      gsl::span<const o2::MCCompLabel> spanOutputTracksMCTruth = {ptrs.outputTracksTPCO2MC, ptrs.outputTracksTPCO2MC ? ptrs.nOutputTracksTPCO2 : 0};

      tracker->Clear(false);

      if (processAttributes->suppressOutput) {
        return;
      }
      if (retVal != 0) {
        throw std::runtime_error("tracker returned error code " + std::to_string(retVal));
      }

      if (!processAttributes->allocateOutputOnTheFly) {
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

      LOG(INFO) << "found " << spanOutputTracks.size() << " track(s)";

      if (specconfig.outputCompClusters) {
        CompressedClustersROOT compressedClusters = *ptrs.tpcCompressedClusters;
        pc.outputs().snapshot(Output{gDataOriginTPC, "COMPCLUSTERS", 0}, ROOTSerialized<CompressedClustersROOT const>(compressedClusters));
      }

      if (processAttributes->clusterOutputIds.size() > 0) {
        if ((void*)ptrs.clustersNative->clustersLinear != (void*)(outputBuffers[outputRegions.getIndex(outputRegions.clustersNative)].second + sizeof(ClusterCountIndex))) {
          throw std::runtime_error("cluster native output ptrs out of sync"); // sanity check
        }

        ClusterNativeAccess const& accessIndex = *ptrs.clustersNative;
        if (specconfig.sendClustersPerSector) {
          // Clusters are shipped by sector, we are copying into per-sector buffers (anyway only for ROOT output)
          for (int i = 0; i < NSectors; i++) {
            if (processAttributes->tpcSectorMask & (1ul << i)) {
              DataHeader::SubSpecificationType subspec = i;
              clusterOutputSectorHeader.sectorBits = (1ul << i);
              char* buffer = pc.outputs().make<char>({gDataOriginTPC, "CLUSTERNATIVE", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, accessIndex.nClustersSector[i] * sizeof(*accessIndex.clustersLinear) + sizeof(ClusterCountIndex)).data();
              ClusterCountIndex* outIndex = reinterpret_cast<ClusterCountIndex*>(buffer);
              memset(outIndex, 0, sizeof(*outIndex));
              for (int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
                outIndex->nClusters[i][j] = accessIndex.nClusters[i][j];
              }
              memcpy(buffer + sizeof(*outIndex), accessIndex.clusters[i][0], accessIndex.nClustersSector[i] * sizeof(*accessIndex.clustersLinear));
              if (specconfig.processMC && accessIndex.clustersMCTruth) {
                MCLabelContainer cont;
                for (int j = 0; j < accessIndex.nClustersSector[i]; j++) {
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
          if (specconfig.processMC && specconfig.caClusterer && accessIndex.clustersMCTruth) {
            pc.outputs().snapshot({gDataOriginTPC, "CLNATIVEMCLBL", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, clustersMCBuffer.first);
          }
        }
      }
      if (specconfig.outputQA) {
        TObjArray out;
        std::vector<TH1F> copy1 = *outputRegions.qa.hist1; // Internally, this will also be used as output, so we need a non-const copy
        std::vector<TH2F> copy2 = *outputRegions.qa.hist2;
        std::vector<TH1D> copy3 = *outputRegions.qa.hist3;
        processAttributes->qa->postprocessExternal(copy1, copy2, copy3, out, processAttributes->qaTaskMask ? processAttributes->qaTaskMask : -1);
        pc.outputs().snapshot({gDataOriginTPC, "TRACKINGQA", 0, Lifetime::Timeframe}, out);
        processAttributes->qa->cleanup();
      }
      timer.Stop();
      LOG(INFO) << "GPU Reoncstruction time for this TF " << timer.CpuTime() - cput << " s (cpu), " << timer.RealTime() - realt << " s (wall)";
    };

    return processingFct;
  };

  // FIXME: find out how to handle merge inputs in a simple and intuitive way
  // changing the binding name of the input in order to identify inputs by unique labels
  // in the processing. Think about how the processing can be made agnostic of input size,
  // e.g. by providing a span of inputs under a certain label
  auto createInputSpecs = [&tpcsectors, &specconfig, policyData]() {
    Inputs inputs;
    if (specconfig.decompressTPC) {
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, specconfig.decompressTPCFromROOT ? o2::header::DataDescription("COMPCLUSTERS") : o2::header::DataDescription("COMPCLUSTERSFLAT")}, Lifetime::Timeframe});
    } else if (specconfig.caClusterer) {
      // We accept digits and MC labels also if we run on ZS Raw data, since they are needed for MC label propagation
      if ((!specconfig.zsOnTheFly || specconfig.processMC) && !specconfig.zsDecoder) {
        inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe});
        policyData->emplace_back(o2::framework::InputSpec{"digits", o2::framework::ConcreteDataTypeMatcher{"TPC", "DIGITS"}});
      }
    } else {
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe});
      policyData->emplace_back(o2::framework::InputSpec{"clusters", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}});
    }
    if (specconfig.processMC) {
      if (specconfig.caClusterer) {
        if (!specconfig.zsDecoder) {
          inputs.emplace_back(InputSpec{"mclblin", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITSMCTR"}, Lifetime::Timeframe});
          policyData->emplace_back(o2::framework::InputSpec{"digitsmc", o2::framework::ConcreteDataTypeMatcher{"TPC", "DIGITSMCTR"}});
        }
      } else {
        inputs.emplace_back(InputSpec{"mclblin", ConcreteDataTypeMatcher{gDataOriginTPC, "CLNATIVEMCLBL"}, Lifetime::Timeframe});
        policyData->emplace_back(o2::framework::InputSpec{"clustersmc", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLNATIVEMCLBL"}});
      }
    }

    if (specconfig.zsDecoder) {
      // All ZS raw data is published with subspec 0 by the o2-raw-file-reader-workflow and DataDistribution
      // creates subspec fom CRU and endpoint id, we create one single input route subscribing to all TPC/RAWDATA
      inputs.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Optional});
      if (specconfig.askDISTSTF) {
        inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
      }
    }
    if (specconfig.zsOnTheFly) {
      inputs.emplace_back(InputSpec{"zsinput", ConcreteDataTypeMatcher{"TPC", "TPCZS"}, Lifetime::Timeframe});
      inputs.emplace_back(InputSpec{"zsinputsizes", ConcreteDataTypeMatcher{"TPC", "ZSSIZES"}, Lifetime::Timeframe});
    }
    if (specconfig.readTRDtracklets) {
      inputs.emplace_back("trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe);
      inputs.emplace_back("trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe);
      inputs.emplace_back("trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);
    }
    return inputs;
  };

  auto createOutputSpecs = [&specconfig, &tpcsectors, &processAttributes]() {
    std::vector<OutputSpec> outputSpecs;
    if (specconfig.outputTracks) {
      outputSpecs.emplace_back(gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe);
      outputSpecs.emplace_back(gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe);
    }
    if (specconfig.processMC && specconfig.outputTracks) {
      outputSpecs.emplace_back(gDataOriginTPC, "TRACKSMCLBL", 0, Lifetime::Timeframe);
    }
    if (specconfig.outputCompClusters) {
      outputSpecs.emplace_back(gDataOriginTPC, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    }
    if (specconfig.outputCompClustersFlat) {
      outputSpecs.emplace_back(gDataOriginTPC, "COMPCLUSTERSFLAT", 0, Lifetime::Timeframe);
    }
    if (specconfig.outputCAClusters) {
      for (auto const& sector : tpcsectors) {
        processAttributes->clusterOutputIds.emplace_back(sector);
      }
      outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", specconfig.sendClustersPerSector ? 0 : NSectors, Lifetime::Timeframe);
      if (specconfig.sendClustersPerSector) {
        outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVETMP", NSectors, Lifetime::Timeframe); // Dummy buffer the TPC tracker writes the inital linear clusters to
        for (const auto sector : tpcsectors) {
          outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", sector, Lifetime::Timeframe);
        }
      } else {
        outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", NSectors, Lifetime::Timeframe);
      }
      if (specconfig.processMC) {
        if (specconfig.sendClustersPerSector) {
          for (const auto sector : tpcsectors) {
            outputSpecs.emplace_back(gDataOriginTPC, "CLNATIVEMCLBL", sector, Lifetime::Timeframe);
          }
        } else {
          outputSpecs.emplace_back(gDataOriginTPC, "CLNATIVEMCLBL", NSectors, Lifetime::Timeframe);
        }
      }
    }
    if (specconfig.outputSharedClusterMap) {
      outputSpecs.emplace_back(gDataOriginTPC, "CLSHAREDMAP", 0, Lifetime::Timeframe);
    }
    if (specconfig.outputQA) {
      outputSpecs.emplace_back(gDataOriginTPC, "TRACKINGQA", 0, Lifetime::Timeframe);
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{processorName, // process id
                           {createInputSpecs()},
                           {createOutputSpecs()},
                           AlgorithmSpec(initFunction)};
}
} // namespace o2::gpu
