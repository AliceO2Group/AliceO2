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
#include "DetectorsRaw/RDHUtils.h"

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
  if (specconfig.outputCAClusters && !specconfig.caClusterer && !specconfig.decompressTPC) {
    throw std::runtime_error("inconsistent configuration: cluster output is only possible if CA clusterer is activated");
  }

  constexpr static size_t NSectors = Sector::MAXSECTOR;
  constexpr static size_t NEndpoints = 20; //TODO: get from mapper?
  using ClusterGroupParser = o2::algorithm::ForwardParser<ClusterGroupHeader>;
  struct ProcessAttributes {
    std::bitset<NSectors> validInputs = 0;
    std::bitset<NSectors> validMcInputs = 0;
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<GPUCATracking> tracker;
    std::unique_ptr<GPUDisplayBackend> displayBackend;
    std::unique_ptr<TPCFastTransform> fastTransform;
    std::unique_ptr<TPCdEdxCalibrationSplines> dEdxSplines;
    int verbosity = 1;
    std::vector<int> clusterOutputIds;
    bool readyToQuit = false;
    bool allocateOutputOnTheFly = false;
    bool suppressOutput = false;
  };

  auto processAttributes = std::make_shared<ProcessAttributes>();
  auto initFunction = [processAttributes, specconfig](InitContext& ic) {
    auto options = ic.options().get<std::string>("tracker-options");
    {
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      parser = std::make_unique<ClusterGroupParser>();
      tracker = std::make_unique<GPUCATracking>();

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
      int dump = 0;                              // create memory dump of processed events for standalone runs, 2 to dump only and skip processing
      char gpuType[1024] = "CUDA";               // Type of GPU device, if useGPU is set to true
      int gpuDevice = -1;                        // Select GPU device id (-1 = auto-detect fastest, -2 = use pipeline-slice)
      GPUDisplayBackend* display = nullptr;      // Ptr to display backend (enables event display)
      bool qa = false;                           // Run the QA after tracking
      bool readTransformationFromFile = false;   // Read the TPC transformation from the file
      bool allocateOutputOnTheFly = true;        // Provide a callback to allocate output buffer on the fly instead of preallocating
      char tpcTransformationFileName[1024] = ""; // A file with the TPC transformation
      char matBudFileName[1024] = "";            // Material budget file name
      char dEdxSplinesFile[1024] = "";           // File containing dEdx splines
      int tpcRejectionMode = GPUSettings::RejectionStrategyA;
      size_t memoryPoolSize = 1;
      size_t hostMemoryPoolSize = 0;

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
            dump = 1;
            printf("Dumping of input events enabled\n");
          } else if (strncmp(optPtr, "dumponly", optLen) == 0) {
            dump = 2;
            printf("Dumping of input events enabled, processing disabled\n");
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
          } else if (optLen > 21 && strncmp(optPtr, "tpcRejectionStrategy=", 21) == 0) {
            sscanf(optPtr + 21, "%d", &tpcRejectionMode);
            tpcRejectionMode = tpcRejectionMode == 0 ? GPUSettings::RejectionNone : tpcRejectionMode == 1 ? GPUSettings::RejectionStrategyA : GPUSettings::RejectionStrategyB;
            printf("TPC Rejection Mode: %d\n", tpcRejectionMode);
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
          } else if (optLen > 8 && strncmp(optPtr, "gpuNum=", 7) == 0) {
            sscanf(optPtr + 7, "%d", &gpuDevice);
            printf("Using GPU device %d\n", gpuDevice);
          } else if (optLen > 8 && strncmp(optPtr, "gpuMemorySize=", 14) == 0) {
            sscanf(optPtr + 14, "%llu", (unsigned long long int*)&memoryPoolSize);
            printf("GPU memory pool size set to %llu\n", (unsigned long long int)memoryPoolSize);
          } else if (optLen > 8 && strncmp(optPtr, "hostMemorySize=", 15) == 0) {
            sscanf(optPtr + 15, "%llu", (unsigned long long int*)&hostMemoryPoolSize);
            printf("Host memory pool size set to %llu\n", (unsigned long long int)hostMemoryPoolSize);
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
      processAttributes->allocateOutputOnTheFly = allocateOutputOnTheFly;
      processAttributes->suppressOutput = (dump == 2);
      GPUO2InterfaceConfiguration config;
      if (useGPU) {
        config.configProcessing.deviceType = GPUDataTypes::GetDeviceType(gpuType);
      } else {
        config.configProcessing.deviceType = GPUDataTypes::DeviceType::CPU;
      }
      config.configProcessing.forceDeviceType = true; // If we request a GPU, we force that it is available - no CPU fallback

      if (gpuDevice == -2) {
        int myId = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
        int idMax = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
        gpuDevice = myId;
        LOG(INFO) << "GPU device number selected from pipeline id: " << myId << " / " << idMax;
      }
      config.configDeviceProcessing.deviceNum = gpuDevice;
      config.configDeviceProcessing.nThreads = nThreads;
      config.configDeviceProcessing.runQA = qa;                                   // Run QA after tracking
      config.configDeviceProcessing.runMC = specconfig.processMC;                 // Propagate MC labels
      config.configDeviceProcessing.eventDisplay = display;                       // Ptr to event display backend, for running standalone OpenGL event display
      config.configDeviceProcessing.debugLevel = debugLevel;                      // Debug verbosity
      config.configDeviceProcessing.forceMemoryPoolSize = memoryPoolSize;         // GPU / Host Memory pool size, default = 1 = auto-detect
      config.configDeviceProcessing.forceHostMemoryPoolSize = hostMemoryPoolSize; // Same for host, overrides the avove value for the host if set
      if (memoryPoolSize || hostMemoryPoolSize) {
        config.configDeviceProcessing.memoryAllocationStrategy = 2;
      }

      config.configEvent.solenoidBz = solenoidBz;
      int maxContTimeBin = (o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * o2::constants::lhc::LHCMaxBunches + 2 * Constants::LHCBCPERTIMEBIN - 2) / Constants::LHCBCPERTIMEBIN;
      config.configEvent.continuousMaxTimeBin = continuous ? maxContTimeBin : 0; // Number of timebins in timeframe if continuous, 0 otherwise

      config.configReconstruction.NWays = 3;               // Should always be 3!
      config.configReconstruction.NWaysOuter = true;       // Will create outer param for TRD
      config.configReconstruction.SearchWindowDZDR = 2.5f; // Should always be 2.5 for looper-finding and/or continuous tracking
      config.configReconstruction.TrackReferenceX = refX;

      // Settings for TPC Compression:
      config.configReconstruction.tpcRejectionMode = tpcRejectionMode;                // Implement TPC Strategy A
      config.configReconstruction.tpcRejectQPt = 1.f / 0.05f;                         // Reject clusters of tracks < 50 MeV
      config.configReconstruction.tpcCompressionModes = GPUSettings::CompressionFull; // Activate all compression steps
      config.configReconstruction.tpcCompressionSortOrder = GPUSettings::SortPad;     // Sort order for differences compression
      config.configReconstruction.tpcSigBitsCharge = 4;                               // Number of significant bits in TPC cluster chargs
      config.configReconstruction.tpcSigBitsWidth = 3;                                // Number of significant bits in TPC cluster width

      config.configInterface.dumpEvents = dump;
      config.configInterface.outputToExternalBuffers = true;

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

    auto processingFct = [processAttributes, specconfig](ProcessingContext& pc) {
      if (processAttributes->readyToQuit) {
        return;
      }
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      uint64_t activeSectors = 0;
      auto& verbosity = processAttributes->verbosity;
      // FIXME cleanup almost duplicated code
      auto& validMcInputs = processAttributes->validMcInputs;
      using CachedMCLabelContainer = decltype(std::declval<InputRecord>().get<MCLabelContainer*>(DataRef{nullptr, nullptr, nullptr}));
      std::vector<CachedMCLabelContainer> mcInputs;
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
      std::array<std::unique_ptr<const MCLabelContainer>, NSectors> inputDigitsMC;
      std::array<unsigned int, NEndpoints * NSectors> tpcZSonTheFlySizes;
      gsl::span<const ZeroSuppressedContainer8kb> inputZS;

      // unsigned int totalZSPages = 0;
      if (specconfig.processMC) {
        std::vector<InputSpec> filter = {
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITSMCTR"}, Lifetime::Timeframe},
          {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLNATIVEMCLBL"}, Lifetime::Timeframe},
        };
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
          std::bitset<NSectors> sectorMask(sectorHeader->sectorBits);
          if ((validMcInputs & sectorMask).any()) {
            // have already data for this sector, this should not happen in the current
            // sequential implementation, for parallel path merged at the tracker stage
            // multiple buffers need to be handled
            throw std::runtime_error("can only have one MC data set per sector");
          }
          inputrefs[sector].labels = ref;
          if (specconfig.caClusterer) {
            inputDigitsMC[sector] = std::move(pc.inputs().get<const MCLabelContainer*>(ref));
          } else {
          }
          validMcInputs |= sectorMask;
          activeSectors |= sectorHeader->activeSectors;
          if (verbosity > 1) {
            LOG(INFO) << "received " << *(ref.spec) << " MC label containers"
                      << " for sectors " << sectorMask                                 //
                      << std::endl                                                     //
                      << "  mc input status:   " << validMcInputs                      //
                      << std::endl                                                     //
                      << "  active sectors: " << std::bitset<NSectors>(activeSectors); //
          }
        }
      }

      auto& validInputs = processAttributes->validInputs;
      int operation = 0;
      std::vector<InputSpec> filter = {
        {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "DIGITS"}, Lifetime::Timeframe},
        {"check", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe},
      };

      for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
        auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(ref);
        if (sectorHeader == nullptr) {
          throw std::runtime_error("sector header missing on header stack");
        }
        const int sector = sectorHeader->sector();
        if (sector < 0) {
          continue;
        }
        std::bitset<NSectors> sectorMask(sectorHeader->sectorBits);
        if ((validInputs & sectorMask).any()) {
          // have already data for this sector, this should not happen in the current
          // sequential implementation, for parallel path merged at the tracker stage
          // multiple buffers need to be handled
          throw std::runtime_error("can only have one cluster data set per sector");
        }
        activeSectors |= sectorHeader->activeSectors;
        validInputs |= sectorMask;
        inputrefs[sector].data = ref;
        if (specconfig.caClusterer && !specconfig.zsOnTheFly) {
          inputDigits[sector] = pc.inputs().get<gsl::span<o2::tpc::Digit>>(ref);
          LOG(INFO) << "GOT SPAN FOR SECTOR " << sector << " -> " << inputDigits[sector].size();
        }
      }
      if (specconfig.zsOnTheFly) {
        tpcZSonTheFlySizes = {0};
        // tpcZSonTheFlySizes: #zs pages per endpoint:
        std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{gDataOriginTPC, "ZSSIZES"}, Lifetime::Timeframe}};
        for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
          tpcZSonTheFlySizes = pc.inputs().get<std::array<unsigned int, NEndpoints * NSectors>>(ref);
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
        if (specconfig.processMC && (activeSectors & validMcInputs.to_ulong()) != activeSectors) {
          throw std::runtime_error("Incomplete mc label input, expecting complete data set, buffering has been removed");
        }
        assert(specconfig.processMC == false || validMcInputs == validInputs);
        for (auto const& refentry : inputrefs) {
          auto& sector = refentry.first;
          auto& ref = refentry.second.data;
          if (ref.payload == nullptr) {
            // skip zero-length message
            continue;
          }
          if (refentry.second.labels.header != nullptr && refentry.second.labels.payload != nullptr) {
            mcInputs.emplace_back(std::move(pc.inputs().get<const MCLabelContainer*>(refentry.second.labels)));
          }
          inputs.emplace_back(gsl::span(ref.payload, DataRefUtils::getPayloadSize(ref)));
          printInputLog(ref, "received", sector);
        }
        assert(mcInputs.size() == 0 || mcInputs.size() == inputs.size());
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
      ptrs.outputTracksMCTruth = (specconfig.processMC ? &tracksMCTruth : nullptr);
      if (specconfig.caClusterer) {
        if (specconfig.zsOnTheFly) {
          std::cout << "inputZS size: " << inputZS.size() << std::endl;
          const unsigned long long int* buffer = reinterpret_cast<const unsigned long long int*>(&inputZS[0]);
          o2::gpu::GPUReconstructionConvert::RunZSEncoderCreateMeta(buffer, tpcZSonTheFlySizes.data(), *&ptrEp, &tpcZS);
          ptrs.tpcZS = &tpcZS;
          if (specconfig.processMC) {
            throw std::runtime_error("Currently unable to process MC information, tpc-tracker crashing when MC propagated");
            ptrs.o2DigitsMC = &inputDigitsMC;
          }
        } else if (specconfig.zsDecoder) {
          ptrs.tpcZS = &tpcZS;
          if (specconfig.processMC) {
            throw std::runtime_error("Cannot process MC information, none available"); // In fact, passing in MC data with ZS TPC Raw is not yet available
          }
        } else {
          ptrs.o2Digits = &inputDigits; // TODO: We will also create ClusterNative as output stored in ptrs. Should be added to the output
          if (specconfig.processMC) {
            ptrs.o2DigitsMC = &inputDigitsMC;
          }
        }
      } else if (specconfig.decompressTPC) {
        ptrs.compressedClusters = pCompClustersFlat;
      } else {
        memset(&clusterIndex, 0, sizeof(clusterIndex));
        ClusterNativeHelper::Reader::fillIndex(clusterIndex, clusterBuffer, clustersMCBuffer, inputs, mcInputs, [&validInputs](auto& index) { return validInputs.test(index); });
        ptrs.clusters = &clusterIndex;
      }
      // a byte size resizable vector object, the DataAllocator returns reference to internal object
      // initialize optional pointer to the vector object
      using ClusterOutputChunkType = std::decay_t<decltype(pc.outputs().make<std::vector<char>>(Output{"", "", 0}))>;
      ClusterOutputChunkType* clusterOutput = nullptr;
      TPCSectorHeader clusterOutputSectorHeader{0};
      if (processAttributes->clusterOutputIds.size() > 0) {
        if (activeSectors == 0) {
          // there is no sector header shipped with the ZS raw data and thus we do not have
          // a valid activeSector variable, though it will be needed downstream
          // FIXME: check if this can be provided upstream
          for (auto const& sector : processAttributes->clusterOutputIds) {
            activeSectors |= 0x1 << sector;
          }
        }
        clusterOutputSectorHeader.sectorBits = activeSectors;
        // subspecs [0, NSectors - 1] are used to identify sector data, we use NSectors
        // to indicate the full TPC
        o2::header::DataHeader::SubSpecificationType subspec = NSectors;
        clusterOutputSectorHeader.activeSectors = activeSectors;
        clusterOutput = &pc.outputs().make<std::vector<char>>({gDataOriginTPC, "CLUSTERNATIVE", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}});
      }

      GPUInterfaceOutputs outputRegions;
      // TODO: For output to preallocated buffer, just allocated some large buffer for now.
      // This should be estimated correctly, but it is not the default for now, so it doesn't matter much.
      size_t bufferSize = 256ul * 1024 * 1024;
      auto* bufferCompressedClusters = !processAttributes->allocateOutputOnTheFly && specconfig.outputCompClustersFlat ? &pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "COMPCLUSTERSFLAT", 0}, bufferSize) : nullptr;
      if (processAttributes->allocateOutputOnTheFly && specconfig.outputCompClustersFlat) {
        outputRegions.compressedClusters.allocator = [&bufferCompressedClusters, &pc](size_t size) -> void* {bufferCompressedClusters = &pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "COMPCLUSTERSFLAT", 0}, size); return bufferCompressedClusters->data(); };
      } else if (specconfig.outputCompClustersFlat) {
        outputRegions.compressedClusters.ptr = bufferCompressedClusters->data();
        outputRegions.compressedClusters.size = bufferCompressedClusters->size();
      }
      if (clusterOutput != nullptr) {
        if (processAttributes->allocateOutputOnTheFly) {
          outputRegions.clustersNative.allocator = [&clusterOutput, &pc](size_t size) -> void* {clusterOutput->resize(size + sizeof(ClusterCountIndex)); return (char*)clusterOutput->data() + sizeof(ClusterCountIndex); };
        } else {
          clusterOutput->resize(bufferSize);
          outputRegions.clustersNative.ptr = (char*)clusterOutput->data() + sizeof(ClusterCountIndex);
          outputRegions.clustersNative.size = clusterOutput->size() * sizeof(*clusterOutput->data()) - sizeof(ClusterCountIndex);
        }
      }
      auto* bufferTPCTracks = !processAttributes->allocateOutputOnTheFly ? &pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "TRACKSGPU", 0}, bufferSize) : nullptr;
      if (processAttributes->allocateOutputOnTheFly) {
        outputRegions.tpcTracks.allocator = [&bufferTPCTracks, &pc](size_t size) -> void* {bufferTPCTracks = &pc.outputs().make<std::vector<char>>(Output{gDataOriginTPC, "TRACKSGPU", 0}, size); return bufferTPCTracks->data(); };
      } else {
        outputRegions.tpcTracks.ptr = bufferTPCTracks->data();
        outputRegions.tpcTracks.size = bufferTPCTracks->size();
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
          LOG(INFO) << "sending " << tracksMCTruth.getIndexedSize() << " track label(s)";
          pc.outputs().snapshot(OutputRef{"mclblout"}, tracksMCTruth);
        }
      }

      if (ptrs.compressedClusters != nullptr) {
        if (specconfig.outputCompClustersFlat) {
          bufferCompressedClusters->resize(outputRegions.compressedClusters.size);
          if ((void*)ptrs.compressedClusters != (void*)bufferCompressedClusters->data()) {
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
      if (clusterOutput != nullptr) {
        clusterOutput->resize(sizeof(ClusterCountIndex) + outputRegions.clustersNative.size);
        if ((void*)ptrs.clusters->clustersLinear != (void*)((char*)clusterOutput->data() + sizeof(ClusterCountIndex))) {
          throw std::runtime_error("cluster native output ptrs out of sync"); // sanity check
        }

        o2::header::DataHeader::SubSpecificationType subspec = NSectors;
        // doing a copy for now, in the future the tracker uses the output buffer directly
        auto& target = *clusterOutput;
        ClusterNativeAccess const& accessIndex = *ptrs.clusters;
        ClusterCountIndex* outIndex = reinterpret_cast<ClusterCountIndex*>(target.data());
        static_assert(sizeof(ClusterCountIndex) == sizeof(accessIndex.nClusters));
        memcpy(outIndex, &accessIndex.nClusters[0][0], sizeof(ClusterCountIndex));
        if (specconfig.processMC && accessIndex.clustersMCTruth) {
          pc.outputs().snapshot({gDataOriginTPC, "CLNATIVEMCLBL", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, *accessIndex.clustersMCTruth);
        }
      }

      validInputs.reset();
      if (specconfig.processMC) {
        validMcInputs.reset();
        for (auto& mcInput : mcInputs) {
          mcInput.reset();
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
    if (specconfig.decompressTPC) {
      inputs.emplace_back(InputSpec{"input", ConcreteDataTypeMatcher{gDataOriginTPC, specconfig.decompressTPCFromROOT ? header::DataDescription("COMPCLUSTERS") : header::DataDescription("COMPCLUSTERSFLAT")}, Lifetime::Timeframe});
    } else if (specconfig.caClusterer) {
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
      outputSpecs.emplace_back(gDataOriginTPC, "CLUSTERNATIVE", NSectors, Lifetime::Timeframe);
      if (specconfig.processMC) {
        outputSpecs.emplace_back(OutputSpec{gDataOriginTPC, "CLNATIVEMCLBL", NSectors, Lifetime::Timeframe});
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
