// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CATrackerSpec.cxx
/// @author Matthias Richter
/// @since  2018-04-18
/// @brief  Processor spec for running TPC CA tracking

#include "TPCWorkflow/CATrackerSpec.h"
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
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "TPCReconstruction/GPUCATracking.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCFastTransform.h"
#include "TPCdEdxCalibrationSplines.h"
#include "DPLUtils/DPLRawParser.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsRaw/HBFUtils.h"
#include "TPCBase/RDHUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCCFCalibration.h"
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
#include <boost/filesystem.hpp>
#include <memory> // for make_shared
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "GPUReconstructionConvert.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;
using namespace o2::tpc::reco_workflow;

namespace o2
{
namespace tpc
{

DataProcessorSpec getCATrackerSpec(CompletionPolicyData* policyData, ca::Config const& specconfig, std::vector<int> const& tpcsectors)
{
  if (specconfig.outputCAClusters && !specconfig.caClusterer && !specconfig.decompressTPC) {
    throw std::runtime_error("inconsistent configuration: cluster output is only possible if CA clusterer is activated");
  }

  constexpr static size_t NSectors = Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = 20; //TODO: get from mapper?
  using ClusterGroupParser = o2::algorithm::ForwardParser<ClusterGroupHeader>;
  struct ProcessAttributes {
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<GPUCATracking> tracker;
    std::unique_ptr<GPUDisplayBackend> displayBackend;
    std::unique_ptr<TPCFastTransform> fastTransform;
    std::unique_ptr<TPCdEdxCalibrationSplines> dEdxSplines;
    std::unique_ptr<TPCCFCalibration> tpcCalibration;
    std::vector<int> clusterOutputIds;
    unsigned long outputBufferSize = 0;
    unsigned long tpcSectorMask = 0;
    int verbosity = 1;
    bool readyToQuit = false;
    bool allocateOutputOnTheFly = false;
    bool suppressOutput = false;
  };

  auto processAttributes = std::make_shared<ProcessAttributes>();
  for (auto s : tpcsectors) {
    processAttributes->tpcSectorMask |= (1ul << s);
  }
  auto initFunction = [processAttributes, specconfig](InitContext& ic) {
    GPUO2InterfaceConfiguration config;
    GPUSettingsO2 confParam;
    {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      parser = std::make_unique<ClusterGroupParser>();
      tracker = std::make_unique<GPUCATracking>();

      // Create configuration object and fill settings
      const auto grp = o2::parameters::GRPObject::loadFrom("o2sim_grp.root");
      if (grp) {
        config.configEvent.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
        config.configEvent.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0; // Number of timebins in timeframe if continuous, 0 otherwise
        LOG(INFO) << "Initializing run paramerers from GRP bz=" << config.configEvent.solenoidBz << " cont=" << grp->isDetContinuousReadOut(o2::detectors::DetID::TPC);
      } else {
        throw std::runtime_error("Failed to initialize run parameters from GRP");
      }
      confParam = config.ReadConfigurableParam();
      processAttributes->allocateOutputOnTheFly = confParam.allocateOutputOnTheFly;
      processAttributes->outputBufferSize = confParam.outputBufferSize;
      processAttributes->suppressOutput = (confParam.dump == 2);
      config.configInterface.dumpEvents = confParam.dump;
      config.configInterface.dropSecondaryLegs = confParam.dropSecondaryLegs;
      if (confParam.display) {
#ifdef GPUCA_BUILD_EVENT_DISPLAY
        processAttributes->displayBackend.reset(new GPUDisplayBackendGlfw);
        config.configProcessing.eventDisplay = processAttributes->displayBackend.get();
        LOG(INFO) << "Event display enabled";
#else
        throw std::runtime_error("Standalone Event Display not enabled at build time!");
#endif
      }

      if (config.configEvent.continuousMaxTimeBin == -1) {
        config.configEvent.continuousMaxTimeBin = (o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * o2::constants::lhc::LHCMaxBunches + 2 * constants::LHCBCPERTIMEBIN - 2) / constants::LHCBCPERTIMEBIN;
      }
      if (config.configProcessing.deviceNum == -2) {
        int myId = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
        int idMax = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
        config.configProcessing.deviceNum = myId;
        LOG(INFO) << "GPU device number selected from pipeline id: " << myId << " / " << idMax;
      }
      config.configProcessing.runMC = specconfig.processMC;
      config.configReconstruction.NWaysOuter = true;
      config.configInterface.outputToExternalBuffers = true;

      // Configure the "GPU workflow" i.e. which steps we run on the GPU (or CPU) with this instance of GPUCATracking
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
      }

      // Create and forward data objects for TPC transformation, material LUT, ...
      if (confParam.transformationFile.size()) {
        processAttributes->fastTransform = nullptr;
        config.configCalib.fastTransform = TPCFastTransform::loadFromFile(confParam.transformationFile.c_str());
      } else {
        processAttributes->fastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));
        config.configCalib.fastTransform = processAttributes->fastTransform.get();
      }
      if (config.configCalib.fastTransform == nullptr) {
        throw std::invalid_argument("GPUCATracking: initialization of the TPC transformation failed");
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

      if (boost::filesystem::exists(confParam.gainCalibFile)) {
        LOG(INFO) << "Loading tpc gain correction from file " << confParam.gainCalibFile;
        const auto* gainMap = o2::tpc::utils::readCalPads(confParam.gainCalibFile, "GainMap")[0];
        processAttributes->tpcCalibration.reset(new TPCCFCalibration{*gainMap});
      } else {
        if (not confParam.gainCalibFile.empty()) {
          LOG(WARN) << "Couldn't find tpc gain correction file " << confParam.gainCalibFile << ". Not applying any gain correction.";
        }
        processAttributes->tpcCalibration.reset(new TPCCFCalibration{});
      }
      config.configCalib.tpcCalibration = processAttributes->tpcCalibration.get();

      // Sample code what needs to be done for the TRD Geometry, when we extend this to TRD tracking.
      /*o2::base::GeometryManager::loadGeometry();
      o2::trd::Geometry gm;
      gm.createPadPlaneArray();
      gm.createClusterMatrixArray();
      std::unique_ptr<o2::trd::GeometryFlat> gf(gm);
      config.trdGeometry = gf.get();*/

      // Configuration is prepared, initialize the tracker.
      if (tracker->initialize(config) != 0) {
        throw std::invalid_argument("GPUCATracking initialization failed");
      }
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

    auto processingFct = [processAttributes, specconfig](ProcessingContext& pc) {
      if (processAttributes->readyToQuit) {
        return;
      }
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      auto& verbosity = processAttributes->verbosity;
      // FIXME cleanup almost duplicated code
      std::vector<ConstMCLabelContainerView> mcInputs;
      std::vector<gsl::span<const char>> inputs;
      struct InputRef {
        DataRef data;
        DataRef labels;
      };
      std::map<int, InputRef> inputrefs;
      const CompressedClustersFlat* pCompClustersFlat;
      size_t compClustersFlatDummyMemory[(sizeof(CompressedClustersFlat) + sizeof(size_t) - 1) / sizeof(size_t)];
      CompressedClustersFlat& compClustersFlatDummy = reinterpret_cast<CompressedClustersFlat&>(compClustersFlatDummyMemory);
      CompressedClusters compClustersDummy;
      o2::gpu::GPUTrackingInOutZS tpcZS;
      std::vector<const void*> tpcZSmetaPointers[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      std::vector<unsigned int> tpcZSmetaSizes[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      const void** tpcZSmetaPointers2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      const unsigned int* tpcZSmetaSizes2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      std::array<gsl::span<const o2::tpc::Digit>, NSectors> inputDigits;
      std::vector<ConstMCLabelContainerView> inputDigitsMC;
      std::array<int, constants::MAXSECTOR> inputDigitsMCIndex;
      std::array<const ConstMCLabelContainerView*, constants::MAXSECTOR> inputDigitsMCPtrs;
      std::array<unsigned int, NEndpoints * NSectors> tpcZSonTheFlySizes;
      gsl::span<const ZeroSuppressedContainer8kb> inputZS;

      // unsigned int totalZSPages = 0;
      if (specconfig.processMC) {
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITSMCTR"}, Lifetime::Timeframe},
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLNATIVEMCLBL"}, Lifetime::Timeframe},
        };
        unsigned long recvMask = 0;
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(ref);
          if (sectorHeader == nullptr) {
            // FIXME: think about error policy
            LOG(ERROR) << "sector header missing on header stack";
            return;
          }
          const int sector = sectorHeader->sector();
          if (sector < 0) {
            continue;
          }
          if (recvMask & sectorHeader->sectorBits) {
            throw std::runtime_error("can only have one MC data set per sector");
          }
          recvMask |= sectorHeader->sectorBits;
          inputrefs[sector].labels = ref;
          if (specconfig.caClusterer) {
            inputDigitsMCIndex[sector] = inputDigitsMC.size();
            inputDigitsMC.emplace_back(ConstMCLabelContainerView(pc.inputs().get<gsl::span<char>>(ref)));
          }
        }
        if (recvMask != processAttributes->tpcSectorMask) {
          throw std::runtime_error("Incomplete set of MC labels received");
        }
        if (specconfig.caClusterer) {
          for (unsigned int i = 0; i < NSectors; i++) {
            LOG(INFO) << "GOT MC LABELS FOR SECTOR " << i << " -> " << inputDigitsMC[inputDigitsMCIndex[i]].getNElements();
            inputDigitsMCPtrs[i] = &inputDigitsMC[inputDigitsMCIndex[i]];
          }
        }
      }

      if (!specconfig.decompressTPC && (!specconfig.caClusterer || ((!specconfig.zsOnTheFly || specconfig.processMC) && !specconfig.zsDecoder))) {
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe},
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe},
        };
        unsigned long recvMask = 0;
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(ref);
          if (sectorHeader == nullptr) {
            throw std::runtime_error("sector header missing on header stack");
          }
          const int sector = sectorHeader->sector();
          if (sector < 0) {
            continue;
          }
          if (recvMask & sectorHeader->sectorBits) {
            throw std::runtime_error("can only have one cluster data set per sector");
          }
          recvMask |= sectorHeader->sectorBits;
          inputrefs[sector].data = ref;
          if (specconfig.caClusterer && (!specconfig.zsOnTheFly || specconfig.processMC)) {
            inputDigits[sector] = pc.inputs().get<gsl::span<o2::tpc::Digit>>(ref);
            LOG(INFO) << "GOT DIGITS SPAN FOR SECTOR " << sector << " -> " << inputDigits[sector].size();
          }
        }
        if (recvMask != processAttributes->tpcSectorMask) {
          throw std::runtime_error("Incomplete set of clusters/digits received");
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
        for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
          for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
            tpcZSmetaPointers[i][j].clear();
            tpcZSmetaSizes[i][j].clear();
          }
        }

        unsigned int offset = 0;
        for (unsigned int i = 0; i < NSectors; i++) {
          unsigned int pageSector = 0;
          for (unsigned int j = 0; j < NEndpoints; j++) {
            pageSector += tpcZSonTheFlySizes[i * NEndpoints + j];
            offset += tpcZSonTheFlySizes[i * NEndpoints + j];
          }
          LOG(INFO) << "GOT ZS pages FOR SECTOR " << i << " ->  pages: " << pageSector;
        }
      }
      if (specconfig.zsDecoder) {
        for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
          for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
            tpcZSmetaPointers[i][j].clear();
            tpcZSmetaSizes[i][j].clear();
          }
        }
        std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          const o2::header::DataHeader* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
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
            const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)current;
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
          auto const* rdh = it.get_if<o2::header::RAWDataHeaderV4>();
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
        for (auto const& refentry : inputrefs) {
          auto& sector = refentry.first;
          auto& ref = refentry.second.data;
          if (!specconfig.caClusterer) {
            if (ref.payload == nullptr) {
              // skip zero-length message
              continue;
            }
            if (refentry.second.labels.header != nullptr && refentry.second.labels.payload != nullptr) {
              mcInputs.emplace_back(ConstMCLabelContainerView(pc.inputs().get<gsl::span<char>>(refentry.second.labels)));
            }
            inputs.emplace_back(gsl::span(ref.payload, DataRefUtils::getPayloadSize(ref)));
          }
          if (verbosity > 1) {
            LOG(INFO) << "received " << *(ref.spec) << ", size " << DataRefUtils::getPayloadSize(ref) << " for sector " << sector;
          }
        }
        if (verbosity) {
          LOGF(INFO, "running tracking for sector(s) 0x%09x", processAttributes->tpcSectorMask);
        }
      }

      std::vector<TrackTPC> tracks;
      std::vector<uint32_t> clusRefs;
      std::vector<o2::MCCompLabel> tracksMCTruth;
      GPUO2InterfaceIOPtrs ptrs;
      ClusterNativeAccess clusterIndex;
      std::unique_ptr<ClusterNative[]> clusterBuffer;
      ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clustersMCBuffer;
      void* ptrEp[NSectors * NEndpoints] = {};
      ptrs.outputTracks = &tracks;
      ptrs.outputClusRefs = &clusRefs;
      ptrs.outputTracksMCTruth = (specconfig.processMC ? &tracksMCTruth : nullptr);
      if (specconfig.caClusterer) {
        if (specconfig.zsOnTheFly) {
          const unsigned long long int* buffer = reinterpret_cast<const unsigned long long int*>(&inputZS[0]);
          o2::gpu::GPUReconstructionConvert::RunZSEncoderCreateMeta(buffer, tpcZSonTheFlySizes.data(), *&ptrEp, &tpcZS);
          ptrs.tpcZS = &tpcZS;
          if (specconfig.processMC) {
            ptrs.o2Digits = &inputDigits;
            ptrs.o2DigitsMC = &inputDigitsMCPtrs;
          }
        } else if (specconfig.zsDecoder) {
          ptrs.tpcZS = &tpcZS;
          if (specconfig.processMC) {
            throw std::runtime_error("Cannot process MC information, none available");
          }
        } else {
          ptrs.o2Digits = &inputDigits;
          if (specconfig.processMC) {
            ptrs.o2DigitsMC = &inputDigitsMCPtrs;
          }
        }
      } else if (specconfig.decompressTPC) {
        ptrs.compressedClusters = pCompClustersFlat;
      } else {
        memset(&clusterIndex, 0, sizeof(clusterIndex));
        ClusterNativeHelper::Reader::fillIndex(clusterIndex, clusterBuffer, clustersMCBuffer, inputs, mcInputs, [&processAttributes](auto& index) { return processAttributes->tpcSectorMask & (1ul << index); });
        ptrs.clusters = &clusterIndex;
      }
      // a byte size resizable vector object, the DataAllocator returns reference to internal object
      // initialize optional pointer to the vector object
      using O2CharVectorOutputType = std::decay_t<decltype(pc.outputs().make<std::vector<char>>(Output{"", "", 0}))>;
      TPCSectorHeader clusterOutputSectorHeader{0};
      if (processAttributes->clusterOutputIds.size() > 0) {
        clusterOutputSectorHeader.sectorBits = processAttributes->tpcSectorMask;
        // subspecs [0, NSectors - 1] are used to identify sector data, we use NSectors
        // to indicate the full TPC
        clusterOutputSectorHeader.activeSectors = processAttributes->tpcSectorMask;
      }

      GPUInterfaceOutputs outputRegions;
      std::optional<std::reference_wrapper<O2CharVectorOutputType>> clusterOutput = std::nullopt, bufferCompressedClusters = std::nullopt, bufferTPCTracks = std::nullopt;
      char *clusterOutputChar = nullptr, *bufferCompressedClustersChar = nullptr, *bufferTPCTracksChar = nullptr;
      if (specconfig.outputCompClustersFlat) {
        if (processAttributes->allocateOutputOnTheFly) {
          outputRegions.compressedClusters.allocator = [&bufferCompressedClustersChar, &pc](size_t size) -> void* {bufferCompressedClustersChar = pc.outputs().make<char>(Output{gDataOriginTPC, "COMPCLUSTERSFLAT", 0}, size).data(); return bufferCompressedClustersChar; };
        } else {
          bufferCompressedClusters.emplace(pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "COMPCLUSTERSFLAT", 0}, processAttributes->outputBufferSize));
          outputRegions.compressedClusters.ptr = bufferCompressedClustersChar = bufferCompressedClusters->get().data();
          outputRegions.compressedClusters.size = bufferCompressedClusters->get().size();
        }
      }
      if (processAttributes->clusterOutputIds.size() > 0) {
        const o2::header::DataDescription outputLabel = specconfig.sendClustersPerSector ? (o2::header::DataDescription) "CLUSTERNATIVETMP" : (o2::header::DataDescription) "CLUSTERNATIVE";
        if (processAttributes->allocateOutputOnTheFly) {
          outputRegions.clustersNative.allocator = [&clusterOutputChar, &pc, clusterOutputSectorHeader, outputLabel](size_t size) -> void* {clusterOutputChar = pc.outputs().make<char>({gDataOriginTPC, outputLabel, NSectors, Lifetime::Timeframe, {clusterOutputSectorHeader}}, size + sizeof(ClusterCountIndex)).data(); return clusterOutputChar + sizeof(ClusterCountIndex); };
        } else {
          clusterOutput.emplace(pc.outputs().make<std::vector<char>>({gDataOriginTPC, outputLabel, NSectors, Lifetime::Timeframe, {clusterOutputSectorHeader}}, processAttributes->outputBufferSize));
          clusterOutputChar = clusterOutput->get().data();
          outputRegions.clustersNative.ptr = clusterOutputChar + sizeof(ClusterCountIndex);
          outputRegions.clustersNative.size = clusterOutput->get().size() - sizeof(ClusterCountIndex);
        }
      }
      if (processAttributes->allocateOutputOnTheFly) {
        outputRegions.tpcTracks.allocator = [&bufferTPCTracksChar, &pc](size_t size) -> void* {bufferTPCTracksChar = pc.outputs().make<char>(Output{gDataOriginTPC, "TRACKSGPU", 0}, size).data(); return bufferTPCTracksChar; };
      } else {
        bufferTPCTracks.emplace(pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "TRACKSGPU", 0}, processAttributes->outputBufferSize));
        outputRegions.tpcTracks.ptr = bufferTPCTracksChar = bufferTPCTracks->get().data();
        outputRegions.tpcTracks.size = bufferTPCTracks->get().size();
      }
      if (specconfig.processMC) {
        outputRegions.clusterLabels.allocator = [&clustersMCBuffer](size_t size) -> void* { return &clustersMCBuffer; };
      }

      int retVal = tracker->runTracking(&ptrs, &outputRegions);
      if (processAttributes->suppressOutput) {
        return;
      }
      if (retVal != 0) {
        throw std::runtime_error("tracker returned error code " + std::to_string(retVal));
      }
      LOG(INFO) << "found " << tracks.size() << " track(s)";
      // tracks are published if the output channel is configured
      if (specconfig.outputTracks) {
        pc.outputs().snapshot(OutputRef{"outTracks"}, tracks);
        pc.outputs().snapshot(OutputRef{"outClusRefs"}, clusRefs);
        if (specconfig.processMC) {
          LOG(INFO) << "sending " << tracksMCTruth.size() << " track label(s)";
          pc.outputs().snapshot(OutputRef{"mclblout"}, tracksMCTruth);
        }
      }

      if (ptrs.compressedClusters != nullptr) {
        if (specconfig.outputCompClustersFlat) {
          if (!processAttributes->allocateOutputOnTheFly) {
            bufferCompressedClusters->get().resize(outputRegions.compressedClusters.size);
          }
          if ((void*)ptrs.compressedClusters != (void*)bufferCompressedClustersChar) {
            throw std::runtime_error("compressed cluster output ptrs out of sync"); // sanity check
          }
        }
        if (specconfig.outputCompClusters) {
          CompressedClustersROOT compressedClusters = *ptrs.compressedClusters;
          pc.outputs().snapshot(Output{gDataOriginTPC, "COMPCLUSTERS", 0}, ROOTSerialized<CompressedClustersROOT const>(compressedClusters));
        }
      } else {
        LOG(ERROR) << "unable to get compressed cluster info from track";
      }

      // publish clusters produced by CA clusterer sector-wise if the outputs are configured
      if (processAttributes->clusterOutputIds.size() > 0 && ptrs.clusters == nullptr) {
        throw std::logic_error("No cluster index object provided by GPU processor");
      }
      // previously, clusters have been published individually for the enabled sectors
      // clusters are now published as one block, subspec is NSectors
      if (processAttributes->clusterOutputIds.size() > 0) {
        if (!processAttributes->allocateOutputOnTheFly) {
          clusterOutput->get().resize(sizeof(ClusterCountIndex) + outputRegions.clustersNative.size);
        }
        if ((void*)ptrs.clusters->clustersLinear != (void*)(clusterOutputChar + sizeof(ClusterCountIndex))) {
          throw std::runtime_error("cluster native output ptrs out of sync"); // sanity check
        }

        ClusterNativeAccess const& accessIndex = *ptrs.clusters;
        if (specconfig.sendClustersPerSector) {
          for (int i = 0; i < NSectors; i++) {
            if (processAttributes->tpcSectorMask & (1ul << i)) {
              o2::header::DataHeader::SubSpecificationType subspec = i;
              clusterOutputSectorHeader.sectorBits = (1ul << i);
              char* buffer = pc.outputs().make<char>({gDataOriginTPC, "CLUSTERNATIVE", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, accessIndex.nClustersSector[i] * sizeof(*accessIndex.clustersLinear) + sizeof(ClusterCountIndex)).data();
              ClusterCountIndex* outIndex = reinterpret_cast<ClusterCountIndex*>(buffer);
              memset(outIndex, 0, sizeof(*outIndex));
              for (int j = 0; j < constants::MAXGLOBALPADROW; j++) {
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
          o2::header::DataHeader::SubSpecificationType subspec = NSectors;
          ClusterCountIndex* outIndex = reinterpret_cast<ClusterCountIndex*>(clusterOutputChar);
          static_assert(sizeof(ClusterCountIndex) == sizeof(accessIndex.nClusters));
          memcpy(outIndex, &accessIndex.nClusters[0][0], sizeof(ClusterCountIndex));
          if (specconfig.processMC && accessIndex.clustersMCTruth) {
            pc.outputs().snapshot({gDataOriginTPC, "CLNATIVEMCLBL", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, clustersMCBuffer.first);
          }
        }
      }
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
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, specconfig.decompressTPCFromROOT ? header::DataDescription("COMPCLUSTERS") : header::DataDescription("COMPCLUSTERSFLAT")}, Lifetime::Timeframe});
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
      inputs.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Timeframe});
    }
    if (specconfig.zsOnTheFly) {
      inputs.emplace_back(InputSpec{"zsinput", ConcreteDataTypeMatcher{"TPC", "TPCZS"}, Lifetime::Timeframe});
      inputs.emplace_back(InputSpec{"zsinputsizes", ConcreteDataTypeMatcher{"TPC", "ZSSIZES"}, Lifetime::Timeframe});
    }
    return inputs;
  };

  //o2::framework::InputSpec{"cluster", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}},
  //  o2::framework::InputSpec{"digits", o2::framework::ConcreteDataTypeMatcher{"TPC", "DIGITS"}})());

  auto createOutputSpecs = [&specconfig, &tpcsectors, &processAttributes]() {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{{"outTracks"}, gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe},
      OutputSpec{{"outClusRefs"}, gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe},
      // This is not really used as an output, but merely to allocate a GPU-registered memory where the GPU can write the track output.
      // Right now, the tracks are still reformatted, and copied in the above buffers.
      // This one will not have any consumer and just be dropped.
      // But we need something to provide to the GPU as external buffer to test direct writing of tracks in the shared memory.
      OutputSpec{{"outTracksGPUBuffer"}, gDataOriginTPC, "TRACKSGPU", 0, Lifetime::Timeframe},
    };
    if (!specconfig.outputTracks) {
      // this case is the less unlikely one, that's why the logic this way
      outputSpecs.clear();
    }
    if (specconfig.processMC && specconfig.outputTracks) {
      OutputLabel label{"mclblout"};
      constexpr o2::header::DataDescription datadesc("TRACKSMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
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
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{"tpc-tracker", // process id
                           {createInputSpecs()},
                           {createOutputSpecs()},
                           AlgorithmSpec(initFunction)};
}

} // namespace tpc
} // namespace o2
