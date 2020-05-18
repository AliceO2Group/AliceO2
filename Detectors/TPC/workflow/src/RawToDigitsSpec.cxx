// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCCalibration/DigitDump.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/RawToDigitsSpec.h"
#include "Framework/Logger.h"
#include <vector>
#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

o2::framework::DataProcessorSpec getRawToDigitsSpec(int channel, const std::string_view inputDef, std::vector<int> const& tpcSectors)
{

  struct ProcessAttributes {
    DigitDump digitDump;                      ///< digits creation class
    rawreader::RawReaderCRUManager rawReader; ///< GBT frame decoder
    uint32_t lastOrbit{0};                    ///< last processed orbit number
    uint32_t maxEvents{100};                  ///< maximum number of events to process
    uint64_t activeSectors{0};                ///< bit mask of active sectors
    bool quit{false};                         ///< if workflow is ready to quit
    std::vector<int> tpcSectors{};            ///< tpc sector configuration
  };

  // ===| stateful initialization |=============================================
  //
  auto initFunction = [channel, tpcSectors](InitContext& ic) {
    // ===| create and set up processing attributes |===
    auto processAttributes = std::make_shared<ProcessAttributes>();
    // set up calibration
    {
      auto& digitDump = processAttributes->digitDump;
      digitDump.init();
      digitDump.setInMemoryOnly();
      const auto pedestalFile = ic.options().get<std::string>("pedestal-file");
      LOGP(info, "Setting pedestal file: {}", pedestalFile);
      digitDump.setPedestalAndNoiseFile(pedestalFile);

      processAttributes->rawReader.createReader("");
      processAttributes->rawReader.setADCDataCallback([&digitDump](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> Int_t {
        Int_t timeBins = digitDump.update(padROCPos, cru, data);
        digitDump.setNumberOfProcessedTimeBins(std::max(digitDump.getNumberOfProcessedTimeBins(), size_t(timeBins)));
        return timeBins;
      });
      processAttributes->maxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
      processAttributes->tpcSectors = tpcSectors;
    }

    // ===| data processor |====================================================
    //
    auto processingFct = [processAttributes, channel](ProcessingContext& pc) {
      if (processAttributes->quit) {
        return;
      }

      // ===| digit snapshot |===
      //
      // lambda that snapshots digits to be sent out;
      // prepares and attaches header with sector information
      //
      auto snapshotDigits = [&pc, processAttributes, channel](std::vector<o2::tpc::Digit> const& digits, int sector) {
        o2::tpc::TPCSectorHeader header{sector};
        header.activeSectors = processAttributes->activeSectors;
        // digit for now are transported per sector, not per lane
        // pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe, header},
        pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(sector), Lifetime::Timeframe, header},
                              const_cast<std::vector<o2::tpc::Digit>&>(digits));
      };

      // loop over all inputs
      for (auto& input : pc.inputs()) {
        const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);

        // select only RAW data
        if (dh->dataDescription != o2::header::gDataDescriptionRawData) {
          continue;
        }

        // ===| extract electronics mapping information |===
        const auto subSpecification = dh->subSpecification;
        const auto cruID = subSpecification >> 16;
        const auto linkID = ((subSpecification + (subSpecification >> 8)) & 0xFF) - 1;
        const auto dataWrapperID = ((subSpecification >> 8) & 0xFF) > 0;
        const auto globalLinkID = linkID + dataWrapperID * 12;
        const auto sector = cruID / 10;

        // update active sectors
        processAttributes->activeSectors |= (0x1 << sector);

        // set up mapping information for raw reader
        auto& reader = processAttributes->rawReader.getReaders()[0];
        reader->forceCRU(cruID);
        reader->setLink(globalLinkID);

        LOGP(debug, "Specifier: {}/{}/{}", dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>(), dh->subSpecification);
        LOGP(debug, "Payload size: {}", dh->payloadSize);
        LOGP(debug, "CRU: {}; linkID: {}; dataWrapperID: {}; globalLinkID: {}", cruID, linkID, dataWrapperID, globalLinkID);

        try {
          o2::framework::RawParser parser(input.payload, dh->payloadSize);

          rawreader::ADCRawData rawData;
          rawreader::GBTFrame gFrame;

          for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
            auto* rdhPtr = it.get_if<o2::header::RAWDataHeaderV4>();
            if (!rdhPtr) {
              break;
            }
            const auto& rdh = *rdhPtr;
            //printRDH(rdh);

            // ===| event handling |===
            //
            // really ugly, better treatment requires extension in DPL
            // events are are detected by close by orbit numbers
            //
            const auto hbOrbit = rdh.heartbeatOrbit;
            const auto lastOrbit = processAttributes->lastOrbit;

            if ((lastOrbit > 0) && (hbOrbit > (lastOrbit + 3))) {
              auto& digitDump = processAttributes->digitDump;
              digitDump.incrementNEvents();
              LOGP(info, "Number of processed events: {} ({})", digitDump.getNumberOfProcessedEvents(), processAttributes->maxEvents);
              digitDump.sortDigits();

              // publish digits of all configured sectors
              for (auto isector : processAttributes->tpcSectors) {
                snapshotDigits(digitDump.getDigits(isector), isector);
              }
              digitDump.clearDigits();

              processAttributes->activeSectors = 0;
              if (digitDump.getNumberOfProcessedEvents() >= processAttributes->maxEvents) {
                LOGP(info, "Maximum number of events reached ({}), no more processing will be done", processAttributes->maxEvents);
                processAttributes->quit = true;
                pc.services().get<ControlService>().endOfStream();
                //pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
                pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
                break;
              }
            }

            processAttributes->lastOrbit = hbOrbit;
            const auto size = it.size();
            auto data = it.data();
            LOGP(debug, "Raw data block payload size: {}", size);

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
          LOG(ERROR) << "can not create raw parser form input data";
          o2::header::hexDump("payload", input.payload, dh->payloadSize, 64);
          LOG(ERROR) << e.what();
        }
      }
    };

    return processingFct;
  };

  std::stringstream id;
  id << "TPCDigitizer" << channel;

  std::vector<OutputSpec> outputs; // define channel by triple of (origin, type id of data to be sent on this channel, subspecification)
  for (auto isector : tpcSectors) {
    outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    id.str().c_str(),
    select(inputDef.data()),
    outputs,
    AlgorithmSpec{initFunction},
    Options{
      {"max-events", VariantType::Int, 100, {"maximum number of events to process"}},
      {"pedestal-file", VariantType::String, "", {"file with pedestals and noise for zero suppression"}}}};
}
} // namespace tpc
} // namespace o2
