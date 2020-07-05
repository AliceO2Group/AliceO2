// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fmt/format.h>
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include <vector>
#include <string>
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::framework;
using RDHUtils = o2::raw::RDHUtils;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("calib-pedestal", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"input-spec", VariantType::String, "A:TPC/RAWDATA", {"selection string input specs"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCCalibPedestal.FirstTimeBin=10;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

using RDH = o2::header::RAWDataHeader;

void printHeader()
{
  LOGP(debug, "{:>5} {:>4} {:>4} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1} {:>10}", "PkC", "pCnt", "fId", "Mem", "CRU", "GLID", "HBOrbit", "HB BC", "s", "Trg");
}

void printRDH(const RDH& rdh)
{
  const int globalLinkID = int(RDHUtils::getLinkID(rdh)) + (((rdh.word1 >> 32) >> 28) * 12);

  LOGP(debug, "{:>5} {:>4} {:>4} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1} {:#010X}", (uint64_t)RDHUtils::getPacketCounter(rdh), (uint64_t)RDHUtils::getPageCounter(rdh),
       (uint64_t)RDHUtils::getFEEID(rdh), (uint64_t)RDHUtils::getMemorySize(rdh), (uint64_t)RDHUtils::getCRUID(rdh), (uint64_t)globalLinkID,
       (uint64_t)RDHUtils::getHeartBeatOrbit(rdh), (uint64_t)RDHUtils::getHeartBeatBC(rdh), (uint64_t)RDHUtils::getStop(rdh), (uint64_t)RDHUtils::getTriggerType(rdh));
}

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpccalibration_configuration.ini");

  struct ProcessAttributes {
    CalibPedestal calibPedestal;
    rawreader::RawReaderCRUManager rawReader;
    uint32_t lastOrbit{0};
    uint64_t lastTFID{0};
    uint32_t maxEvents{100};
    bool quit{false};
    bool dumped{false};
  };

  auto initFunction = [](InitContext& ic) {
    auto processAttributes = std::make_shared<ProcessAttributes>();
    // set up calibration
    // TODO:
    // it is a bit ugly to use the RawReaderCRUManager for this is.
    // At some point the raw reader code should be cleaned up and modularized
    {
      auto& pedestal = processAttributes->calibPedestal;
      pedestal.init(); // initialize configuration via configKeyValues
      processAttributes->rawReader.createReader("");
      processAttributes->rawReader.setADCDataCallback([&pedestal](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> Int_t {
        Int_t timeBins = pedestal.update(padROCPos, cru, data);
        pedestal.setNumberOfProcessedTimeBins(std::max(pedestal.getNumberOfProcessedTimeBins(), size_t(timeBins)));
        return timeBins;
      });
      processAttributes->maxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    }

    auto processingFct = [processAttributes](ProcessingContext& pc) {
      // in case the maximum number of events was reached don't do further processing
      if (processAttributes->quit) {
        return;
      }

      if (pc.inputs().isValid("TFID")) {
        auto tfid = pc.inputs().get<uint64_t>("TFID");
        LOGP(info, "TFid: {}", tfid);
        processAttributes->lastTFID = tfid;
      }

      for (auto& input : pc.inputs()) {
        const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);

        // ---| only process RAWDATA, there might be a nicer way to do this |---
        if (dh == nullptr || dh->dataDescription != o2::header::gDataDescriptionRawData) {
          continue;
        }

        // ---| extract hardware information to do the processing |---
        const auto subSpecification = dh->subSpecification;
        const auto cruID = subSpecification >> 16;
        const auto linkID = ((subSpecification + (subSpecification >> 8)) & 0xFF) - 1;
        const auto dataWrapperID = ((subSpecification >> 8) & 0xFF) > 0;
        const auto globalLinkID = linkID + dataWrapperID * 12;

        // ---| update hardware information in the reader |---
        auto& reader = processAttributes->rawReader.getReaders()[0];
        reader->forceCRU(cruID);
        reader->setLink(globalLinkID);

        LOGP(debug, "Specifier: {}/{}/{}", dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>(), dh->subSpecification);
        LOGP(debug, "Payload size: {}", dh->payloadSize);
        LOGP(debug, "CRU: {}; linkID: {}; dataWrapperID: {}; globalLinkID: {}", cruID, linkID, dataWrapperID, globalLinkID);

        printHeader();

        // TODO: exception handling needed?
        try {
          o2::framework::RawParser parser(input.payload, dh->payloadSize);

          // TODO: it would be better to have external event handling and then moving the event processing functionality to CalibRawBase and RawReader to not repeat it in other places
          rawreader::ADCRawData rawData;
          rawreader::GBTFrame gFrame;

          auto& calibPedestal = processAttributes->calibPedestal;

          for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
            auto* rdhPtr = it.get_if<o2::header::RAWDataHeader>();
            if (!rdhPtr) {
              break;
            }
            const auto& rdh = *rdhPtr;
            printRDH(rdh);
            // ===| event handling |===
            //
            // really ugly, better treatment required extension in DPL
            // events are are detected by close by orbit numbers
            // might be possible to change this using the TFID information
            //
            const auto hbOrbit = RDHUtils::getHeartBeatOrbit(rdhPtr);
            const auto lastOrbit = processAttributes->lastOrbit;

            if ((lastOrbit > 0) && (hbOrbit > (lastOrbit + 3))) {
              calibPedestal.incrementNEvents();
              LOGP(info, "Number of processed events: {} ({})", calibPedestal.getNumberOfProcessedEvents(), processAttributes->maxEvents);
              if (calibPedestal.getNumberOfProcessedEvents() >= processAttributes->maxEvents) {
                LOGP(info, "Maximm number of events reached ({}), no more processing will be done", processAttributes->maxEvents);
                processAttributes->quit = true;
                break;
              }
            }

            processAttributes->lastOrbit = hbOrbit;
            const auto size = it.size();
            auto data = it.data();
            //LOGP(info, "Data size: {}", size);

            int iFrame = 0;
            for (int i = 0; i < size; i += 16) {
              gFrame.setFrameNumber(iFrame);
              gFrame.setPacketNumber(iFrame / 508);
              gFrame.readFromMemory(gsl::span<const o2::byte>(data + i, 16));

              // extract the half words from the 4 32-bit words
              gFrame.getFrameHalfWords();

              // debug output
              //if (CHECK_BIT(mDebugLevel, DebugLevel::GBTFrames)) {
              //std::cout << gFrame;
              //}

              gFrame.getAdcValues(rawData);
              gFrame.updateSyncCheck(false);

              ++iFrame;
            }
          }

          reader->runADCDataCallback(rawData);
        } catch (const std::runtime_error& e) {
          LOGP(error, "can not create raw parser form input data");
          o2::header::hexDump("payload", input.payload, dh->payloadSize, 64);
          LOG(ERROR) << e.what();
        }
      }

      // TODO: For the moment simply dump calibration output to file, to check if everything is working as expected
      if (processAttributes->quit && !processAttributes->dumped) {
        LOGP(info, "Dumping output");
        processAttributes->calibPedestal.analyse();
        processAttributes->calibPedestal.dumpToFile("pedestals.root");
        processAttributes->dumped = true;
        //pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
      }
    };

    return processingFct;
  };

  WorkflowSpec workflow;
  workflow.emplace_back(DataProcessorSpec{
    "calib-pedestal",
    select(config.options().get<std::string>("input-spec").c_str()),
    Outputs{},
    AlgorithmSpec{initFunction},
    Options{{"max-events", VariantType::Int, 100, {"maximum number of events to process"}}}});

  return workflow;
}
