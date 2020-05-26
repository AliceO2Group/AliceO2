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
#include "GPUDisplayBackend.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY
#include "GPUDisplayBackendGlfw.h"
#endif
#include "DataFormatsParameters/GRPObject.h"
#include "TPCBase/Sector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Algorithm/Parser.h"
#include <memory> // for make_shared
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include "GPUReconstructionConvert.h"

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;

namespace o2
{
namespace tpc
{

DataProcessorSpec getCATrackerSpec(ca::Config const& specconfig, std::vector<int> const& tpcsectors)
{
  auto& processMC = specconfig.processMC;
  auto& caClusterer = specconfig.caClusterer;
  auto& zsDecoder = specconfig.zsDecoder;
  auto& zsOnTheFly = specconfig.zsOnTheFly;
  if (specconfig.outputCAClusters && !specconfig.caClusterer) {
    throw std::runtime_error("inconsistent configuration: cluster output is only possible if CA clusterer is activated");
  }

  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = 20; //TODO: get from mapper?
  using ClusterGroupParser = o2::algorithm::ForwardParser<o2::tpc::ClusterGroupHeader>;
  struct ProcessAttributes {
    std::bitset<NSectors> validInputs = 0;
    std::bitset<NSectors> validMcInputs = 0;
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<o2::tpc::GPUCATracking> tracker;
    std::unique_ptr<o2::gpu::GPUDisplayBackend> displayBackend;
    std::unique_ptr<TPCFastTransform> fastTransform;
    std::unique_ptr<o2::gpu::TPCdEdxCalibrationSplines> dEdxSplines;
    int verbosity = 1;
    std::vector<int> clusterOutputIds;
    bool readyToQuit = false;
  };

  auto processAttributes = std::make_shared<ProcessAttributes>();
  auto initFunction = [processAttributes, processMC, caClusterer, zsDecoder, zsOnTheFly](InitContext& ic) {
    auto options = ic.options().get<std::string>("tracker-options");
    {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      parser = std::make_unique<ClusterGroupParser>();
      tracker = std::make_unique<o2::tpc::GPUCATracking>();

      // Prepare initialization of CATracker - we parse the deprecated option string here,
      // and create the proper configuration objects for compatibility.
      // This should go away eventually.

      // Default Settings
      float solenoidBz = 5.00668;                // B-field
      float refX = 83.;                          // transport tracks to this x after tracking, >500 for disabling
      bool continuous = false;                   // time frame data v.s. triggered events
      int nThreads = 1;                          // number of threads if we run on the CPU, 1 = default, 0 = auto-detect
      bool useGPU = false;                       // use a GPU for processing, if false uses GPU
      int debugLevel = 0;                        // Enable additional debug output
      bool dump = false;                         // create memory dump of processed events for standalone runs
      char gpuType[1024] = "CUDA";               // Type of GPU device, if useGPU is set to true
      GPUDisplayBackend* display = nullptr;      // Ptr to display backend (enables event display)
      bool qa = false;                           // Run the QA after tracking
      bool readTransformationFromFile = false;   // Read the TPC transformation from the file
      char tpcTransformationFileName[1024] = ""; // A file with the TPC transformation
      char matBudFileName[1024] = "";            // Material budget file name
      char dEdxSplinesFile[1024] = "";           // File containing dEdx splines

      const auto grp = o2::parameters::GRPObject::loadFrom("o2sim_grp.root");
      if (grp) {
        solenoidBz *= grp->getL3Current() / 30000.;
        continuous = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC);
        LOG(INFO) << "Initializing run paramerers from GRP bz=" << solenoidBz << " cont=" << continuous;
      } else {
        throw std::runtime_error("Failed to initialize run parameters from GRP");
      }

      // Parse the config string
      const char* opt = options.c_str();
      if (opt && *opt) {
        printf("Received options %s\n", opt);
        const char* optPtr = opt;
        while (optPtr && *optPtr) {
          while (*optPtr == ' ') {
            optPtr++;
          }
          const char* nextPtr = strstr(optPtr, " ");
          const int optLen = nextPtr ? nextPtr - optPtr : strlen(optPtr);
          if (strncmp(optPtr, "cont", optLen) == 0) {
            continuous = true;
            printf("Continuous tracking mode enabled\n");
          } else if (strncmp(optPtr, "dump", optLen) == 0) {
            dump = true;
            printf("Dumping of input events enabled\n");
          } else if (strncmp(optPtr, "display", optLen) == 0) {
#ifdef GPUCA_BUILD_EVENT_DISPLAY
            processAttributes->displayBackend.reset(new GPUDisplayBackendGlfw);
            display = processAttributes->displayBackend.get();
            printf("Event display enabled\n");
#else
            printf("Standalone Event Display not enabled at build time!\n");
#endif
          } else if (strncmp(optPtr, "qa", optLen) == 0) {
            qa = true;
            printf("Enabling TPC Standalone QA\n");
          } else if (optLen > 3 && strncmp(optPtr, "bz=", 3) == 0) {
            sscanf(optPtr + 3, "%f", &solenoidBz);
            printf("Using solenoid field %f\n", solenoidBz);
          } else if (optLen > 5 && strncmp(optPtr, "refX=", 5) == 0) {
            sscanf(optPtr + 5, "%f", &refX);
            printf("Propagating to reference X %f\n", refX);
          } else if (optLen > 5 && strncmp(optPtr, "debug=", 6) == 0) {
            sscanf(optPtr + 6, "%d", &debugLevel);
            printf("Debug level set to %d\n", debugLevel);
          } else if (optLen > 8 && strncmp(optPtr, "threads=", 8) == 0) {
            sscanf(optPtr + 8, "%d", &nThreads);
            printf("Using %d threads\n", nThreads);
          } else if (optLen > 8 && strncmp(optPtr, "gpuType=", 8) == 0) {
            int len = std::min(optLen - 8, 1023);
            memcpy(gpuType, optPtr + 8, len);
            gpuType[len] = 0;
            useGPU = true;
            printf("Using GPU Type %s\n", gpuType);
          } else if (optLen > 8 && strncmp(optPtr, "matBudFile=", 8) == 0) {
            int len = std::min(optLen - 11, 1023);
            memcpy(matBudFileName, optPtr + 11, len);
            matBudFileName[len] = 0;
          } else if (optLen > 8 && strncmp(optPtr, "dEdxFile=", 9) == 0) {
            int len = std::min(optLen - 9, 1023);
            memcpy(dEdxSplinesFile, optPtr + 9, len);
            dEdxSplinesFile[len] = 0;
          } else if (optLen > 15 && strncmp(optPtr, "transformation=", 15) == 0) {
            int len = std::min(optLen - 15, 1023);
            memcpy(tpcTransformationFileName, optPtr + 15, len);
            tpcTransformationFileName[len] = 0;
            readTransformationFromFile = true;
            printf("Read TPC transformation from the file \"%s\"\n", tpcTransformationFileName);
          } else {
            printf("Unknown option: %s\n", optPtr);
            throw std::invalid_argument("Unknown config string option");
          }
          optPtr = nextPtr;
        }
      }

      // Create configuration object and fill settings
      GPUO2InterfaceConfiguration config;
      if (useGPU) {
        config.configProcessing.deviceType = GPUDataTypes::GetDeviceType(gpuType);
      } else {
        config.configProcessing.deviceType = GPUDataTypes::DeviceType::CPU;
      }
      config.configProcessing.forceDeviceType = true; // If we request a GPU, we force that it is available - no CPU fallback

      config.configDeviceProcessing.nThreads = nThreads;
      config.configDeviceProcessing.runQA = qa;              // Run QA after tracking
      config.configDeviceProcessing.runMC = processMC;       // Propagate MC labels
      config.configDeviceProcessing.eventDisplay = display;  // Ptr to event display backend, for running standalone OpenGL event display
      config.configDeviceProcessing.debugLevel = debugLevel; // Debug verbosity
      config.configDeviceProcessing.forceMemoryPoolSize = 1; // Some memory auto-detection

      config.configEvent.solenoidBz = solenoidBz;
      int maxContTimeBin = (o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * o2::constants::lhc::LHCMaxBunches + 2 * Constants::LHCBCPERTIMEBIN - 2) / Constants::LHCBCPERTIMEBIN;
      config.configEvent.continuousMaxTimeBin = continuous ? maxContTimeBin : 0; // Number of timebins in timeframe if continuous, 0 otherwise

      config.configReconstruction.NWays = 3;               // Should always be 3!
      config.configReconstruction.NWaysOuter = true;       // Will create outer param for TRD
      config.configReconstruction.SearchWindowDZDR = 2.5f; // Should always be 2.5 for looper-finding and/or continuous tracking
      config.configReconstruction.TrackReferenceX = refX;

      // Settings for TPC Compression:
      config.configReconstruction.tpcRejectionMode = GPUSettings::RejectionStrategyA; // Implement TPC Strategy A
      config.configReconstruction.tpcRejectQPt = 1.f / 0.05f;                         // Reject clusters of tracks < 50 MeV
      config.configReconstruction.tpcCompressionModes = GPUSettings::CompressionFull; // Activate all compression steps
      config.configReconstruction.tpcCompressionSortOrder = GPUSettings::SortPad;     // Sort order for differences compression
      config.configReconstruction.tpcSigBitsCharge = 4;                               // Number of significant bits in TPC cluster chargs
      config.configReconstruction.tpcSigBitsWidth = 3;                                // Number of significant bits in TPC cluster width

      config.configInterface.dumpEvents = dump;
      config.configInterface.outputToPreallocatedBuffers = true;

      // Configure the "GPU workflow" i.e. which steps we run on the GPU (or CPU) with this instance of GPUCATracking
      config.configWorkflow.steps.set(GPUDataTypes::RecoStep::TPCConversion,
                                      GPUDataTypes::RecoStep::TPCSliceTracking,
                                      GPUDataTypes::RecoStep::TPCMerging,
                                      GPUDataTypes::RecoStep::TPCCompression,
                                      GPUDataTypes::RecoStep::TPCdEdx);
      // Alternative steps: TRDTracking | ITSTracking
      config.configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCClusters);
      // Alternative inputs: GPUDataTypes::InOutType::TRDTracklets
      config.configWorkflow.outputs.set(GPUDataTypes::InOutType::TPCMergedTracks, GPUDataTypes::InOutType::TPCCompressedClusters);
      // Alternative outputs: GPUDataTypes::InOutType::TPCSectorTracks, GPUDataTypes::InOutType::TRDTracks
      if (caClusterer) { // Override some settings if we have raw data as input
        config.configWorkflow.inputs.set(GPUDataTypes::InOutType::TPCRaw);
        config.configWorkflow.steps.setBits(GPUDataTypes::RecoStep::TPCClusterFinding, true);
        config.configWorkflow.outputs.setBits(GPUDataTypes::InOutType::TPCClusters, true);
      }

      // Create and forward data objects for TPC transformation, material LUT, ...
      if (readTransformationFromFile) {
        processAttributes->fastTransform = nullptr;
        config.configCalib.fastTransform = TPCFastTransform::loadFromFile(tpcTransformationFileName);
      } else {
        processAttributes->fastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));
        config.configCalib.fastTransform = processAttributes->fastTransform.get();
      }
      if (config.configCalib.fastTransform == nullptr) {
        throw std::invalid_argument("GPUCATracking: initialization of the TPC transformation failed");
      }
      if (strlen(matBudFileName)) {
        config.configCalib.matLUT = o2::base::MatLayerCylSet::loadFromFile(matBudFileName, "MatBud");
      }
      processAttributes->dEdxSplines.reset(new TPCdEdxCalibrationSplines);
      if (strlen(dEdxSplinesFile)) {
        TFile dEdxFile(dEdxSplinesFile);
        processAttributes->dEdxSplines->setSplinesFromFile(dEdxFile);
      }
      config.configCalib.dEdxSplines = processAttributes->dEdxSplines.get();

      // Sample code what needs to be done for the TRD Geometry, when we extend this to TRD tracking.
      /*o2::base::GeometryManager::loadGeometry();
      o2::trd::TRDGeometry gm;
      gm.createPadPlaneArray();
      gm.createClusterMatrixArray();
      std::unique_ptr<o2::trd::TRDGeometryFlat> gf(gm);
      config.trdGeometry = gf.get();*/

      // Configuration is prepared, initialize the tracker.
      if (tracker->initialize(config) != 0) {
        throw std::invalid_argument("GPUCATracking initialization failed");
      }
      processAttributes->validInputs.reset();
      processAttributes->validMcInputs.reset();
    }

    auto& callbacks = ic.services().get<CallbackService>();
    callbacks.set(CallbackService::Id::RegionInfoCallback, [processAttributes](FairMQRegionInfo const& info) {
      if (info.size) {
        auto& tracker = processAttributes->tracker;
        if (tracker->registerMemoryForGPU(info.ptr, info.size)) {
          throw std::runtime_error("Error registering memory for GPU");
        }
      }
    });

    auto processingFct = [processAttributes, processMC, caClusterer, zsDecoder, zsOnTheFly](ProcessingContext& pc) {
      if (processAttributes->readyToQuit) {
        return;
      }
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      uint64_t activeSectors = 0;
      auto& verbosity = processAttributes->verbosity;
      // FIXME cleanup almost duplicated code
      auto& validMcInputs = processAttributes->validMcInputs;
      using CachedMCLabelContainer = decltype(std::declval<InputRecord>().get<std::vector<MCLabelContainer>>(DataRef{nullptr, nullptr, nullptr}));
      std::array<CachedMCLabelContainer, NSectors> mcInputs;
      std::array<gsl::span<const char>, NSectors> inputs;
      o2::gpu::GPUTrackingInOutZS tpcZS;
      std::vector<const void*> tpcZSmetaPointers[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      std::vector<unsigned int> tpcZSmetaSizes[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      const void** tpcZSmetaPointers2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      const unsigned int* tpcZSmetaSizes2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      std::array<gsl::span<const o2::tpc::Digit>, NSectors> inputDigits;
      std::array<std::unique_ptr<const MCLabelContainer>, NSectors> inputDigitsMC;
      std::array<unsigned int, NEndpoints * NSectors> sizes;
      gsl::span<const ZeroSuppressedContainer8kb> inputZS;

      // unsigned int totalZSPages = 0;
      if (processMC) {
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITSMCTR"}, Lifetime::Timeframe},
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLNATIVEMCLBL"}, Lifetime::Timeframe},
        };
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
          if (sectorHeader == nullptr) {
            // FIXME: think about error policy
            LOG(ERROR) << "sector header missing on header stack";
            return;
          }
          const int sector = sectorHeader->sector();
          if (sector < 0) {
            continue;
          }
          // the TPCSectorHeader now allows to transport information for more than one sector,
          // e.g. for transporting clusters in one single data block. For the moment, the
          // implemenation here requires single sectors
          if (sector >= TPCSectorHeader::NSectors) {
            throw std::runtime_error("Expecting data for single sectors");
          }
          if (validMcInputs.test(sector)) {
            // have already data for this sector, this should not happen in the current
            // sequential implementation, for parallel path merged at the tracker stage
            // multiple buffers need to be handled
            throw std::runtime_error("can only have one MC data set per sector");
          }
          if (caClusterer) {
            inputDigitsMC[sector] = std::move(pc.inputs().get<const MCLabelContainer*>(ref));
          } else {
            mcInputs[sector] = std::move(pc.inputs().get<std::vector<MCLabelContainer>>(ref));
          }
          validMcInputs.set(sector);
          activeSectors |= sectorHeader->activeSectors;
          if (verbosity > 1) {
            LOG(INFO) << "received " << *(ref.spec) << " MC label containers"
                      << " for sector " << sector                                      //
                      << std::endl                                                     //
                      << "  mc input status:   " << validMcInputs                      //
                      << std::endl                                                     //
                      << "  active sectors: " << std::bitset<NSectors>(activeSectors); //
          }
        }
      }

      auto& validInputs = processAttributes->validInputs;
      int operation = 0;
      std::map<int, DataRef> datarefs;
      std::vector<InputSpec> filter = {
        {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe},
        {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe},
      };

      for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
        auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
        if (sectorHeader == nullptr) {
          throw std::runtime_error("sector header missing on header stack");
        }
        const int sector = sectorHeader->sector();
        if (sector < 0) {
          //throw std::runtime_error("lagacy input, custom eos is not expected anymore")
          continue;
        }
        // the TPCSectorHeader now allows to transport information for more than one sector,
        // e.g. for transporting clusters in one single data block. For the moment, the
        // implemenation here requires single sectors
        if (sector >= TPCSectorHeader::NSectors) {
          throw std::runtime_error("Expecting data for single sectors");
        }
        if (validInputs.test(sector)) {
          // have already data for this sector, this should not happen in the current
          // sequential implementation, for parallel path merged at the tracker stage
          // multiple buffers need to be handled
          throw std::runtime_error("can only have one cluster data set per sector");
        }
        activeSectors |= sectorHeader->activeSectors;
        validInputs.set(sector);
        datarefs[sector] = ref;
        if (caClusterer && !zsOnTheFly) {
          inputDigits[sector] = pc.inputs().get<gsl::span<o2::tpc::Digit>>(ref);
          LOG(INFO) << "GOT SPAN FOR SECTOR " << sector << " -> " << inputDigits[sector].size();
        }
      }
      if (zsOnTheFly) {
        sizes = {0};
        // sizes: #zs pages per endpoint:
        std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "ZSSIZES"}, Lifetime::Timeframe}};
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          sizes = pc.inputs().get<std::array<unsigned int, NEndpoints * NSectors>>(ref);
        }
        // zs pages
        std::vector<InputSpec> filter2 = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "TPCZS"}, Lifetime::Timeframe}};
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter2)) {
          inputZS = pc.inputs().get<gsl::span<ZeroSuppressedContainer8kb>>(ref);
        }
        //set all sectors as active and as valid inputs
        for (int s = 0; s < NSectors; s++) {
          activeSectors |= 1 << s;
          validInputs.set(s);
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
            pageSector += sizes[i * NEndpoints + j];
            offset += sizes[i * NEndpoints + j];
          }
          LOG(INFO) << "GOT ZS pages FOR SECTOR " << i << " ->  pages: " << pageSector;
        }
      }
      if (zsDecoder) {
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
              //lastFEE = o2::raw::RDHUtils::getFEEID(*rdh);
              lastFEE = int(rdh->feeId);
              rawcru = int(rdh->cruID);
              rawendpoint = int(rdh->endPointID);
              if (it.size() == 0 && tpcZSmetaPointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].size()) {
                ptr = nullptr;
                continue;
              }
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
      } else if (!zsOnTheFly) {
        // FIXME: We can have digits input in zs decoder mode for MC labels
        // This code path should run optionally also for the zs decoder version
        auto printInputLog = [&verbosity, &validInputs, &activeSectors](auto& r, const char* comment, auto& s) {
          if (verbosity > 1) {
            LOG(INFO) << comment << " " << *(r.spec) << ", size " << DataRefUtils::getPayloadSize(r) //
                      << " for sector " << s                                                         //
                      << std::endl                                                                   //
                      << "  input status:   " << validInputs                                         //
                      << std::endl                                                                   //
                      << "  active sectors: " << std::bitset<NSectors>(activeSectors);               //
          }
        };
        // for digits and clusters we always have the sector information, activeSectors being zero
        // is thus an error condition. The completion policy makes sure that the data set is complete
        if (activeSectors == 0 || (activeSectors & validInputs.to_ulong()) != activeSectors) {
          throw std::runtime_error("Incomplete input data, expecting complete data set, buffering has been removed ");
        }
        // MC label blocks must be in the same multimessage with the corresponding data, the completion
        // policy does not check for the MC labels and expects them to be present and thus complete if
        // the data is complete
        if (processMC && (activeSectors & validMcInputs.to_ulong()) != activeSectors) {
          throw std::runtime_error("Incomplete mc label input, expecting complete data set, buffering has been removed");
        }
        assert(processMC == false || validMcInputs == validInputs);
        for (auto const& refentry : datarefs) {
          auto& sector = refentry.first;
          auto& ref = refentry.second;
          inputs[sector] = gsl::span(ref.payload, DataRefUtils::getPayloadSize(ref));
          printInputLog(ref, "received", sector);
        }
        if (verbosity > 0) {
          // make human readable information from the bitfield
          std::string bitInfo;
          auto nActiveBits = validInputs.count();
          if (((uint64_t)0x1 << nActiveBits) == validInputs.to_ulong() + 1) {
            // sectors 0 to some upper bound are active
            bitInfo = "0-" + std::to_string(nActiveBits - 1);
          } else {
            int rangeStart = -1;
            int rangeEnd = -1;
            for (size_t sector = 0; sector < validInputs.size(); sector++) {
              if (validInputs.test(sector)) {
                if (rangeStart < 0) {
                  if (rangeEnd >= 0) {
                    bitInfo += ",";
                  }
                  bitInfo += std::to_string(sector);
                  if (nActiveBits == 1) {
                    break;
                  }
                  rangeStart = sector;
                }
                rangeEnd = sector;
              } else {
                if (rangeStart >= 0 && rangeEnd > rangeStart) {
                  bitInfo += "-" + std::to_string(rangeEnd);
                }
                rangeStart = -1;
              }
            }
            if (rangeStart >= 0 && rangeEnd > rangeStart) {
              bitInfo += "-" + std::to_string(rangeEnd);
            }
          }
          LOG(INFO) << "running tracking for sector(s) " << bitInfo;
        }
      }

      bool doOutputCompressedClustersFlat = pc.outputs().isAllowed({gDataOriginTPC, "COMPCLUSTERSFLAT", 0});
      bool doOutputCompressedClustersROOT = pc.outputs().isAllowed({gDataOriginTPC, "COMPCLUSTERS", 0});

      std::vector<TrackTPC> tracks;
      std::vector<uint32_t> clusRefs;
      MCLabelContainer tracksMCTruth;
      GPUO2InterfaceIOPtrs ptrs;
      ClusterNativeAccess clusterIndex;
      std::unique_ptr<ClusterNative[]> clusterBuffer;
      MCLabelContainer clustersMCBuffer;
      void* ptrEp[NSectors * NEndpoints] = {};
      ptrs.outputTracks = &tracks;
      ptrs.outputClusRefs = &clusRefs;
      ptrs.outputTracksMCTruth = (processMC ? &tracksMCTruth : nullptr);
      if (caClusterer) {
        if (zsOnTheFly) {
          std::cout << "inputZS size: " << inputZS.size() << std::endl;
          const unsigned long long int* buffer = reinterpret_cast<const unsigned long long int*>(&inputZS[0]);
          o2::gpu::GPUReconstructionConvert::RunZSEncoderCreateMeta(buffer, sizes.data(), *&ptrEp, &tpcZS);
          ptrs.tpcZS = &tpcZS;
          if (processMC) {
            throw std::runtime_error("Currently unable to process MC information, tpc-tracker crashing when MC propagated");
            ptrs.o2DigitsMC = &inputDigitsMC;
          }
        } else if (zsDecoder) {
          ptrs.tpcZS = &tpcZS;
          if (processMC) {
            throw std::runtime_error("Cannot process MC information, none available"); // In fact, passing in MC data with ZS TPC Raw is not yet available
          }
        } else {
          ptrs.o2Digits = &inputDigits; // TODO: We will also create ClusterNative as output stored in ptrs. Should be added to the output
          if (processMC) {
            ptrs.o2DigitsMC = &inputDigitsMC;
          }
        }

      } else {
        memset(&clusterIndex, 0, sizeof(clusterIndex));
        ClusterNativeHelper::Reader::fillIndex(clusterIndex, clusterBuffer, clustersMCBuffer, inputs, mcInputs, [&validInputs](auto& index) { return validInputs.test(index); });
        ptrs.clusters = &clusterIndex;
      }
      GPUInterfaceOutputs outputRegions;
      size_t bufferSize = 2048ul * 1024 * 1024; // TODO: Just allocated some large buffer for now, should estimate this correctly;
      auto* bufferCompressedClusters = doOutputCompressedClustersFlat ? &pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "COMPCLUSTERSFLAT", 0}, bufferSize) : nullptr;
      if (doOutputCompressedClustersFlat) {
        outputRegions.compressedClusters.ptr = bufferCompressedClusters->data();
        outputRegions.compressedClusters.size = bufferCompressedClusters->size();
      }
      int retVal = tracker->runTracking(&ptrs, &outputRegions);
      if (retVal != 0) {
        throw std::runtime_error("tracker returned error code " + std::to_string(retVal));
      }
      LOG(INFO) << "found " << tracks.size() << " track(s)";
      // tracks are published if the output channel is configured
      if (pc.outputs().isAllowed({gDataOriginTPC, "TRACKS", 0})) {
        pc.outputs().snapshot(OutputRef{"outTracks"}, tracks);
        pc.outputs().snapshot(OutputRef{"outClusRefs"}, clusRefs);
        if (pc.outputs().isAllowed({gDataOriginTPC, "TRACKSMCLBL", 0})) {
          LOG(INFO) << "sending " << tracksMCTruth.getIndexedSize() << " track label(s)";
          pc.outputs().snapshot(OutputRef{"mclblout"}, tracksMCTruth);
        }
      }

      // The tracker produces a ROOT-serializable container with compressed TPC clusters
      // It is published if the output channel for the CompClusters container is configured
      // Example to decompress clusters
      //#include "TPCClusterDecompressor.cxx"
      //o2::tpc::ClusterNativeAccess clustersNativeDecoded; // Cluster native access structure as used by the tracker
      //std::vector<o2::tpc::ClusterNative> clusterBuffer; // std::vector that will hold the actual clusters, clustersNativeDecoded will point inside here
      //mDecoder.decompress(clustersCompressed, clustersNativeDecoded, clusterBuffer, param); // Run decompressor
      if (ptrs.compressedClusters != nullptr) {
        if (doOutputCompressedClustersFlat) {
          if ((void*)ptrs.compressedClusters != (void*)bufferCompressedClusters->data()) {
            throw std::runtime_error("output ptrs out of sync"); // sanity check
          }
          bufferCompressedClusters->resize(outputRegions.compressedClusters.size);
        }
        if (doOutputCompressedClustersROOT) {
          o2::tpc::CompressedClustersROOT compressedClusters = *ptrs.compressedClusters;
          printf("FOOOO %d\n", compressedClusters.nTracks);
          pc.outputs().snapshot(Output{gDataOriginTPC, "COMPCLUSTERS", 0}, ROOTSerialized<o2::tpc::CompressedClustersROOT const>(compressedClusters));
        }
      } else {
        LOG(ERROR) << "unable to get compressed cluster info from track";
      }

      // publish clusters produced by CA clusterer sector-wise if the outputs are configured
      if (processAttributes->clusterOutputIds.size() > 0 && ptrs.clusters == nullptr) {
        throw std::logic_error("No cluster index object provided by GPU processor");
      }
      if (processAttributes->clusterOutputIds.size() > 0 && activeSectors == 0) {
        // there is no sector header shipped with the ZS raw data and thus we do not have
        // a valid activeSector variable, though it will be needed downstream
        // FIXME: check if this can be provided upstream
        for (auto const& sector : processAttributes->clusterOutputIds) {
          activeSectors |= 0x1 << sector;
        }
      }
      for (auto const& sector : processAttributes->clusterOutputIds) {
        o2::tpc::TPCSectorHeader header{sector};
        o2::header::DataHeader::SubSpecificationType subspec = sector;
        header.activeSectors = activeSectors;
        auto& target = pc.outputs().make<std::vector<char>>({gDataOriginTPC, "CLUSTERNATIVE", subspec, Lifetime::Timeframe, {header}});
        std::vector<MCLabelContainer> labels;
        ClusterNativeHelper::copySectorData(*ptrs.clusters, sector, target, labels);
        if (pc.outputs().isAllowed({gDataOriginTPC, "CLNATIVEMCLBL", subspec})) {
          pc.outputs().snapshot({gDataOriginTPC, "CLNATIVEMCLBL", subspec, Lifetime::Timeframe, {header}}, labels);
        }
      }

      validInputs.reset();
      if (processMC) {
        validMcInputs.reset();
        for (auto& mcInput : mcInputs) {
          mcInput.clear();
        }
      }
    };

    return processingFct;
  };

  // FIXME: find out how to handle merge inputs in a simple and intuitive way
  // changing the binding name of the input in order to identify inputs by unique labels
  // in the processing. Think about how the processing can be made agnostic of input size,
  // e.g. by providing a span of inputs under a certain label
  auto createInputSpecs = [&tpcsectors, &specconfig]() {
    Inputs inputs;
    if (specconfig.caClusterer) {
      // We accept digits and MC labels also if we run on ZS Raw data, since they are needed for MC label propagation
      if (!specconfig.zsOnTheFly && !specconfig.zsDecoder) { // FIXME: We can have digits input in zs decoder mode for MC labels, to be made optional
        inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe});
      }
    } else {
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe});
    }
    if (specconfig.processMC) {
      if (!specconfig.zsOnTheFly && specconfig.caClusterer) {
        constexpr o2::header::DataDescription datadesc("DIGITSMCTR");
        if (!specconfig.zsDecoder) { // FIXME: We can have digits input in zs decoder mode for MC labels, to be made optional
          inputs.emplace_back(InputSpec{"mclblin", ConcreteDataTypeMatcher{gDataOriginTPC, datadesc}, Lifetime::Timeframe});
        }
      } else {
        inputs.emplace_back(InputSpec{"mclblin", ConcreteDataTypeMatcher{gDataOriginTPC, "CLNATIVEMCLBL"}, Lifetime::Timeframe});
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

  auto createOutputSpecs = [&specconfig, &tpcsectors, &processAttributes]() {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{{"outTracks"}, gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe},
      OutputSpec{{"outClusRefs"}, gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe},
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
        o2::header::DataHeader::SubSpecificationType id = sector;
        outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", id, Lifetime::Timeframe);
        processAttributes->clusterOutputIds.emplace_back(sector);
        if (specconfig.processMC) {
          outputSpecs.emplace_back(OutputSpec{gDataOriginTPC, "CLNATIVEMCLBL", id, Lifetime::Timeframe});
        }
      }
    }
    return std::move(outputSpecs);
  };

  return DataProcessorSpec{"tpc-tracker", // process id
                           {createInputSpecs()},
                           {createOutputSpecs()},
                           AlgorithmSpec(initFunction),
                           Options{
                             {"tracker-options", VariantType::String, "", {"Option string passed to tracker"}},
                           }};
}

} // namespace tpc
} // namespace o2
