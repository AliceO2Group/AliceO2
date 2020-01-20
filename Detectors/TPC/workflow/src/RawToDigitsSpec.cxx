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
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCBase/Digit.h"
#include "TPCCalibration/DigitDump.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/RawToDigitsSpec.h"
#include <vector>
#include <string>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

o2::framework::DataProcessorSpec getRawToDigitsSpec(int channel, const std::string_view inputDef)
{

  struct ProcessAttributes {
    DigitDump digitDump;
    rawreader::RawReaderCRUManager rawReader;
    uint32_t lastOrbit{0};
    uint32_t maxEvents{100};
    uint64_t activeSectors{0};
    bool quit{false};
  };

  auto initFunction = [channel](InitContext& ic) {
    auto processAttributes = std::make_shared<ProcessAttributes>();
    // set up calibration
    {
      auto& digitDump = processAttributes->digitDump;
      digitDump.init();
      digitDump.setInMemoryOnly();
      const auto pedestalFile = ic.options().get<std::string>("pedestal-file");
      LOG(INFO) << "Setting pedestal file: " << pedestalFile;
      digitDump.setPedestalAndNoiseFile(pedestalFile);

      processAttributes->rawReader.createReader("");
      processAttributes->rawReader.setADCDataCallback([&digitDump](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> Int_t {
        Int_t timeBins = digitDump.update(padROCPos, cru, data);
        digitDump.setNumberOfProcessedTimeBins(std::max(digitDump.getNumberOfProcessedTimeBins(), size_t(timeBins)));
        return timeBins;
      });
      processAttributes->maxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    }

    auto processingFct = [processAttributes, channel](ProcessingContext& pc) {
      if (processAttributes->quit) {
        return;
      }

      //uint64_t activeSectors = 0;

      // lambda that snapshots digits to be sent out; prepares and attaches header with sector information
      auto snapshotDigits = [&pc, processAttributes, channel](std::vector<o2::tpc::Digit> const& digits, int sector) {
        o2::tpc::TPCSectorHeader header{sector};
        header.activeSectors = processAttributes->activeSectors;
        // note that snapshoting only works with non-const references (to be fixed?)
        pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe, header},
                              const_cast<std::vector<o2::tpc::Digit>&>(digits));
      };

      // loop over all inputs
      for (auto& input : pc.inputs()) {
        const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);

        // select only RAW data
        if (dh->dataDescription != o2::header::gDataDescriptionRawData) {
          continue;
        }

        const auto subSpecification = dh->subSpecification;
        const auto cruID = subSpecification >> 16;
        const auto linkID = ((subSpecification + (subSpecification >> 8)) & 0xFF) - 1;
        const auto dataWrapperID = ((subSpecification >> 8) & 0xFF) > 0;
        const auto globalLinkID = linkID + dataWrapperID * 12;
        const auto sector = cruID / 10;
        processAttributes->activeSectors |= (0x1 << sector);

        auto& reader = processAttributes->rawReader.getReaders()[0];
        reader->forceCRU(cruID);
        reader->setLink(globalLinkID);

        //LOG(INFO) << dh->dataOrigin.as<std::string>() << "/" << dh->dataDescription.as<std::string>() << "/"
        //<< dh->subSpecification << " payload size " << dh->payloadSize;
        //LOG(INFO) << "CRU: " << cruID << " -- linkID: " << linkID << " -- dataWrapperID: " << dataWrapperID << " -- globalLinkID: " << globalLinkID;

        // there is a bug in InpuRecord::get for vectors of simple types, not catched in
        // DataAllocator unit test
        //auto data = inputs.get<std::vector<char>>(input.spec->binding.c_str());
        //LOG(INFO) << "data size " << data.size();
        //printHeader();
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
            // ugly event handling ...
            const auto hbOrbit = rdh.heartbeatOrbit;
            const auto lastOrbit = processAttributes->lastOrbit;

            if ((lastOrbit > 0) && (hbOrbit > (lastOrbit + 3))) {
              auto& digitDump = processAttributes->digitDump;
              digitDump.incrementNEvents();
              LOG(INFO) << fmt::format("Number of processed events: {} ({})", digitDump.getNumberOfProcessedEvents(), processAttributes->maxEvents);
              digitDump.sortDigits();
              // add publish here
              for (int isector = 0; isector < Sector::MAXSECTOR; ++isector) {
                if (processAttributes->activeSectors & (0x1 << isector)) {
                  snapshotDigits(digitDump.getDigits(isector), isector);
                }
              }
              digitDump.clearDigits();

              processAttributes->activeSectors = 0;
              if (digitDump.getNumberOfProcessedEvents() >= processAttributes->maxEvents) {
                LOG(INFO) << fmt::format("Maximm number of events reached ({}), no more processing will be done", processAttributes->maxEvents);
                processAttributes->quit = true;
                pc.services().get<ControlService>().endOfStream();
                pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
                //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
                break;
              }
            }

            processAttributes->lastOrbit = hbOrbit;
            const auto size = it.size();
            auto data = it.data();
            //LOG(INFO) << fmt::format("Data size: {}", size);

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
  outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);

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
