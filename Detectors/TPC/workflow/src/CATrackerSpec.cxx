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
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "TPCReconstruction/GPUCATracking.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCBase/Digit.h"
#include "TPCFastTransform.h"
#include "DPLUtils/DPLRawParser.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsRaw/HBFUtils.h"
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

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;

namespace o2
{
namespace tpc
{

DataProcessorSpec getCATrackerSpec(ca::Config const& config, std::vector<int> const& inputIds)
{
  auto& processMC = config.processMC;
  auto& caClusterer = config.caClusterer;
  auto& zsDecoder = config.zsDecoder;
  constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
  using ClusterGroupParser = o2::algorithm::ForwardParser<o2::tpc::ClusterGroupHeader>;
  struct ProcessAttributes {
    // the input comes in individual calls and we need to buffer until
    // data set is complete, have to think about a DPL feature to take
    // ownership of an input
    std::array<std::vector<char>, NSectors> bufferedInputs;
    std::array<std::vector<MCLabelContainer>, NSectors> mcInputs;
    std::bitset<NSectors> validInputs = 0;
    std::bitset<NSectors> validMcInputs = 0;
    std::unique_ptr<ClusterGroupParser> parser;
    std::unique_ptr<o2::tpc::GPUCATracking> tracker;
    std::unique_ptr<o2::gpu::GPUDisplayBackend> displayBackend;
    std::unique_ptr<TPCFastTransform> fastTransform;
    int verbosity = 1;
    std::vector<int> inputIds;
    bool readyToQuit = false;
  };

  auto initFunction = [processMC, caClusterer, zsDecoder, inputIds](InitContext& ic) {
    auto options = ic.options().get<std::string>("tracker-options");

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->inputIds = inputIds;
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

      const auto grp = o2::parameters::GRPObject::loadFrom("o2sim_grp.root");
      if (grp) {
        solenoidBz *= grp->getL3Current() / 30000.;
        continuous = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC);
        LOG(INFO) << "Initializing run paramerers from GRP bz=" << solenoidBz << " cont=" << continuous;
      } else {
        LOG(ERROR) << "Failed to initialize run parameters from GRP";
        // should we call fatal here?
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

      config.configEvent.solenoidBz = solenoidBz;
      int maxContTimeBin = (o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * o2::constants::lhc::LHCMaxBunches + Constants::LHCBCPERTIMEBIN - 1) / Constants::LHCBCPERTIMEBIN;
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

    auto processingFct = [processAttributes, processMC, caClusterer, zsDecoder](ProcessingContext& pc) {
      if (processAttributes->readyToQuit) {
        return;
      }
      auto& parser = processAttributes->parser;
      auto& tracker = processAttributes->tracker;
      uint64_t activeSectors = 0;
      auto& verbosity = processAttributes->verbosity;

      // FIXME cleanup almost duplicated code
      auto& validMcInputs = processAttributes->validMcInputs;
      auto& mcInputs = processAttributes->mcInputs;
      std::array<gsl::span<const char>, NSectors> inputs;
      o2::gpu::GPUTrackingInOutZS tpcZS;
      std::vector<const void*> tpcZSmetaPointers[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      std::vector<unsigned int> tpcZSmetaSizes[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      const void** tpcZSmetaPointers2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      const unsigned int* tpcZSmetaSizes2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
      std::array<gsl::span<const o2::tpc::Digit>, NSectors> inputDigits;
      std::array<std::unique_ptr<const MCLabelContainer>, NSectors> inputDigitsMC;
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
          const int sector = sectorHeader->sector;
          if (sector < 0) {
            continue;
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
          // FIXME: think about error policy
          LOG(ERROR) << "sector header missing on header stack";
          return;
        }
        const int sector = sectorHeader->sector;
        // check the current operation, this is used to either signal eod or noop
        // FIXME: the noop is not needed any more once the lane configuration with one
        // channel per sector is used
        if (sector < 0) {
          if (operation < 0 && operation != sector) {
            // we expect the same operation on all inputs
            LOG(ERROR) << "inconsistent lane operation, got " << sector << ", expecting " << operation;
          } else if (operation == 0) {
            // store the operation
            operation = sector;
          }
          continue;
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
        if (caClusterer) {
          inputDigits[sector] = pc.inputs().get<gsl::span<o2::tpc::Digit>>(ref);
          LOG(INFO) << "GOT SPAN FOR SECTOR " << sector << " -> " << inputDigits[sector].size();
        }
      }

      if (operation == -1) {
        // EOD is transmitted in the sectorHeader with sector number equal to -1
        o2::tpc::TPCSectorHeader sh{-1};
        sh.activeSectors = activeSectors;
        pc.outputs().snapshot(OutputRef{"outTracks", 0, {sh}}, -1);
        pc.outputs().snapshot(OutputRef{"outClusRefs", 0, {sh}}, -1);
        if (processMC) {
          pc.outputs().snapshot(OutputRef{"mclblout", 0, {sh}}, -1);
        }
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        processAttributes->readyToQuit = true;
        return;
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
          int lastFEE = -1;
          int rawcru = 0;
          int rawendpoint = 0;
          size_t totalSize = 0;
          for (auto it = parser.begin(); it != parser.end(); it++) {
            const unsigned char* current = it.raw();
            const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)current;
            if (current == nullptr || it.size() == 0 || (current - ptr) % TPCZSHDR::TPC_ZS_PAGE_SIZE || rdh->feeId != lastFEE) {
              if (count) {
                tpcZSmetaPointers[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(ptr);
                tpcZSmetaSizes[rawcru / 10][(rawcru % 10) * 2 + rawendpoint].emplace_back(count);
              }
              count = 0;
              if (it.size() == 0) {
                ptr = nullptr;
                continue;
              }
              lastFEE = rdh->feeId;
              rawcru = lastFEE >> 7;
              rawendpoint = (lastFEE & 64) >> 6;
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

      } else {
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
        auto& bufferedInputs = processAttributes->bufferedInputs;
        if (activeSectors == 0 || (activeSectors & validInputs.to_ulong()) != activeSectors ||
            (processMC && (activeSectors & validMcInputs.to_ulong()) != activeSectors)) {
          // not all sectors available, we have to buffer the inputs
          if (caClusterer) {
            throw std::runtime_error("Buffering not possible with digits");
          }
          for (auto const& refentry : datarefs) {
            auto& sector = refentry.first;
            auto& ref = refentry.second;
            auto payloadSize = DataRefUtils::getPayloadSize(ref);
            bufferedInputs[sector].resize(payloadSize);
            std::copy(ref.payload, ref.payload + payloadSize, bufferedInputs[sector].begin());
            printInputLog(ref, "buffering", sector);
          }

          // not needed to send something, DPL will simply drop this timeslice, whenever the
          // data for all sectors is available, the output is sent in that time slice
          return;
        }
        assert(processMC == false || validMcInputs == validInputs);
        auto inputStatus = validInputs;
        for (auto const& refentry : datarefs) {
          auto& sector = refentry.first;
          auto& ref = refentry.second;
          inputs[sector] = gsl::span(ref.payload, DataRefUtils::getPayloadSize(ref));
          inputStatus.reset(sector);
          printInputLog(ref, "received", sector);
        }
        if (inputStatus.any()) {
          // some of the inputs have been buffered
          for (size_t sector = 0; sector < inputStatus.size(); ++sector) {
            if (inputStatus.test(sector)) {
              inputs[sector] = gsl::span(&bufferedInputs[sector].front(), bufferedInputs[sector].size());
            }
          }
        }
        if (verbosity > 0) {
          if (inputStatus.any()) {
            LOG(INFO) << "using buffered data for " << inputStatus.count() << " sector(s)";
          }
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
      ptrs.outputTracks = &tracks;
      ptrs.outputClusRefs = &clusRefs;
      ptrs.outputTracksMCTruth = (processMC ? &tracksMCTruth : nullptr);
      if (caClusterer) {
        if (zsDecoder) {
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
      int retVal = tracker->runTracking(&ptrs);
      if (retVal != 0) {
        // FIXME: error policy
        LOG(FATAL) << "tracker returned error code " << retVal;
      }
      LOG(INFO) << "found " << tracks.size() << " track(s)";
      // tracks are published if the output channel is configured
      if (pc.outputs().isAllowed({gDataOriginTPC, "TRACKS", 0})) {
        pc.outputs().snapshot(OutputRef{"outTracks"}, tracks);
        pc.outputs().snapshot(OutputRef{"outClusRefs"}, clusRefs);
        if (pc.outputs().isAllowed({gDataOriginTPC, "TRACKMCLBL", 0})) {
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
      if (pc.outputs().isAllowed({gDataOriginTPC, "COMPCLUSTERS", 0})) {
        const o2::tpc::CompressedClusters* compressedClusters = ptrs.compressedClusters;
        if (compressedClusters != nullptr) {
          pc.outputs().snapshot(Output{gDataOriginTPC, "COMPCLUSTERS", 0}, ROOTSerialized<o2::tpc::CompressedClusters const>(*compressedClusters));
        } else {
          LOG(ERROR) << "unable to get compressed cluster info from track";
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
  auto createInputSpecs = [&inputIds, &config]() {
    Inputs inputs;
    if (config.caClusterer) {
      // We accept digits and MC labels also if we run on ZS Raw data, since they are needed for MC label propagation
      if (!config.zsDecoder) { // FIXME: We can have digits input in zs decoder mode for MC labels, to be made optional
        inputs.emplace_back(InputSpec{"input", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe});
      }
    } else {
      inputs.emplace_back(InputSpec{"input", gDataOriginTPC, "CLUSTERNATIVE", 0, Lifetime::Timeframe});
    }
    if (config.processMC) {
      if (config.caClusterer) {
        constexpr o2::header::DataDescription datadesc("DIGITSMCTR");
        if (!config.zsDecoder) { // FIXME: We can have digits input in zs decoder mode for MC labels, to be made optional
          inputs.emplace_back(InputSpec{"mclblin", gDataOriginTPC, datadesc, 0, Lifetime::Timeframe});
        }
      } else {
        inputs.emplace_back(InputSpec{"mclblin", gDataOriginTPC, "CLNATIVEMCLBL", 0, Lifetime::Timeframe});
      }
    }

    auto tmp = std::move(mergeInputs(inputs, inputIds.size(),
                                     [inputIds](InputSpec& input, size_t index) {
                                       // using unique input names for the moment but want to find
                                       // an input-multiplicity-agnostic way of processing
                                       input.binding += std::to_string(inputIds[index]);
                                       DataSpecUtils::updateMatchingSubspec(input, inputIds[index]);
                                     }));
    if (config.zsDecoder) {
      // We add this after the mergeInputs, since we need to keep the subspecification
      tmp.emplace_back(InputSpec{"zsraw", ConcreteDataTypeMatcher{"TPC", "RAWDATA"}, Lifetime::Timeframe});
    }
    return tmp;
  };

  auto createOutputSpecs = [&config]() {
    std::vector<OutputSpec> outputSpecs{
      OutputSpec{{"outTracks"}, gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe},
      OutputSpec{{"outClusRefs"}, gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe},
    };
    if (!config.outputTracks) {
      // this case is the less unlikely one, that's why the logic this way
      outputSpecs.clear();
    }
    if (config.processMC && config.outputTracks) {
      OutputLabel label{"mclblout"};
      constexpr o2::header::DataDescription datadesc("TRACKMCLBL");
      outputSpecs.emplace_back(label, gDataOriginTPC, datadesc, 0, Lifetime::Timeframe);
    }
    if (config.outputCompClusters) {
      outputSpecs.emplace_back(gDataOriginTPC, "COMPCLUSTERS", 0, Lifetime::Timeframe);
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
