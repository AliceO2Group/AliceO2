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

#include <fmt/format.h>
#include "Framework/WorkflowSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/CRU.h"
#include "TPCBase/PadSecPos.h"
#include "TPCBase/Mapper.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/LinkZSToDigitsSpec.h"
#include <vector>
#include <string>
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

o2::framework::DataProcessorSpec getLinkZSToDigitsSpec(int channel, const std::string_view inputDef, std::vector<int> const& tpcSectors)
{

  static constexpr int MaxNumberOfBunches = 3564;

  struct ProcessAttributes {
    uint32_t lastOrbit{0};                                         ///< last processed orbit number
    uint32_t firstOrbit{0};                                        ///< first orbit number, required for time bin calculation
    uint32_t firstBC{0};                                           ///< first bunch crossing number, required for time bin calculation
    uint32_t maxEvents{100};                                       ///< maximum number of events to process
    uint32_t processedEvents{0};                                   ///< number of processed events
    uint64_t activeSectors{0};                                     ///< bit mask of active sectors
    bool isContinuous{false};                                      ///< if data is triggered or continuous
    bool quit{false};                                              ///< if workflow is ready to quit
    std::vector<int> tpcSectors{};                                 ///< tpc sector configuration
    std::array<std::vector<Digit>, Sector::MAXSECTOR> digitsAll{}; ///< digit vector to be stored inside the file

    /// cleanup of digits
    void clearDigits()
    {
      for (auto& digits : digitsAll) {
        digits.clear();
      }
    }

    /// Digit sorting according to expected output from simulation
    void sortDigits()
    {
      // sort digits
      for (auto& digits : digitsAll) {
        std::sort(digits.begin(), digits.end(), [](const auto& a, const auto& b) {
          if (a.getTimeStamp() < b.getTimeStamp()) {
            return true;
          }
          if ((a.getTimeStamp() == b.getTimeStamp()) && (a.getRow() < b.getRow())) {
            return true;
          }
          return false;
        });
      }
    }
  };

  // ===| stateful initialization |=============================================
  //
  auto initFunction = [channel, tpcSectors](InitContext& ic) {
    auto processAttributes = std::make_shared<ProcessAttributes>();
    // ===| create and set up processing attributes |===
    {
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
        // pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(channel), header},
        pc.outputs().snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(sector), header},
                              const_cast<std::vector<o2::tpc::Digit>&>(digits));
      };

      auto& mapper = Mapper::instance();

      // loop over all inputs
      for (auto& input : pc.inputs()) {
        const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
        auto payloadSize = DataRefUtils::getPayloadSize(input);

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
        const CRU cru(cruID);
        const int fecLinkOffsetCRU = (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2;
        const int fecInPartition = (globalLinkID % 12) + (globalLinkID > 11) * fecLinkOffsetCRU;
        const int regionIter = cruID % 2;

        const int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
        const int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};

        processAttributes->activeSectors |= (0x1 << sector);

        LOGP(debug, "Specifier: {}/{}/{}", dh->dataOrigin, dh->dataDescription, dh->subSpecification);
        LOGP(debug, "Payload size: {}", payloadSize);
        LOGP(debug, "CRU: {}; linkID: {}; dataWrapperID: {}; globalLinkID: {}", cruID, linkID, dataWrapperID, globalLinkID);

        try {
          o2::framework::RawParser parser(input.payload, payloadSize);

          for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
            auto* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
            if (!rdhPtr) {
              break;
            }

            const auto& rdh = *rdhPtr;

            // ===| only accept physics triggers |===
            if (o2::raw::RDHUtils::getTriggerType(rdhPtr) != 0x10) {
              continue;
            }

            // ===| event handling |===
            //
            // really ugly, better treatment required extension in DPL
            // events are are detected by close by orbit numbers
            //
            const auto hbOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhPtr);
            const auto lastOrbit = processAttributes->lastOrbit;

            if (!processAttributes->firstOrbit) {
              processAttributes->firstOrbit = hbOrbit;
              if (!processAttributes->isContinuous) {
                processAttributes->firstBC = o2::raw::RDHUtils::getHeartBeatBC(rdhPtr);
              }
            }

            const auto globalBCoffset = ((hbOrbit - processAttributes->firstOrbit) * MaxNumberOfBunches - processAttributes->firstBC); // To be calculated

            if ((lastOrbit > 0) && (hbOrbit > (lastOrbit + 3))) {
              ++processAttributes->processedEvents;
              LOG(info) << fmt::format("Number of processed events: {} ({})", processAttributes->processedEvents, processAttributes->maxEvents);
              processAttributes->sortDigits();

              // publish digits of all configured sectors
              for (auto isector : processAttributes->tpcSectors) {
                snapshotDigits(processAttributes->digitsAll[isector], isector);
              }
              processAttributes->clearDigits();

              processAttributes->activeSectors = 0;
              if (processAttributes->processedEvents >= processAttributes->maxEvents) {
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

            // cast raw data pointer to link based zero suppression definition
            zerosupp_link_based::ContainerZS* zsdata = (zerosupp_link_based::ContainerZS*)data;
            const zerosupp_link_based::ContainerZS* const zsdataEnd = (zerosupp_link_based::ContainerZS*)(data + size);

            while (zsdata < zsdataEnd) {
              const auto channelBits = zsdata->cont.header.getChannelBits();
              const uint32_t numberOfWords = zsdata->cont.header.numWordsPayload;
              assert((channelBits.count() - 1) / 10 == numberOfWords - 1);

              std::size_t processedChannels = 0;
              for (std::size_t ichannel = 0; ichannel < channelBits.size(); ++ichannel) {
                if (!channelBits[ichannel]) {
                  continue;
                }

                // adc value
                const auto adcValue = zsdata->getADCValueFloat(processedChannels);

                // pad mapping
                // TODO: verify the assumptions of the channel mapping!
                // assumes the following sorting (s_chn is the channel on the sampa),
                // in this case for even regiona (lower half fec)
                //   chn# SAMPA s_chn
                //      0     0     0
                //      1     0     1
                //      2     0    16
                //      3     0    17
                //      4     1     0
                //      5     1     1
                //      6     1    16
                //      7     1    17
                //      8     2     0
                //      9     2     1
                //
                //     10     0     2
                //     11     0     3
                //     12     0    18
                //     13     0    19
                //     14     1     2
                //     15     1     3
                //     16     1    18
                //     17     1    19
                //     18     2     2
                //     19     2     3
                //
                //     20     0     4
                //     21     0     5
                //     22     0    20
                //     23     0    21
                //     ...
                //     For the uneven regions (upper half fec), the sampa ordering
                //     is 3, 3, 3, 3, 4, 4, 4, 4, 2, 2
                const int istreamm = ((ichannel % 10) / 2);
                const int partitionStream = istreamm + regionIter * 5;
                const int sampaOnFEC = sampaMapping[partitionStream];
                const int channel = (ichannel % 2) + 2 * (ichannel / 10);
                const int channelOnSAMPA = channel + channelOffset[partitionStream];

                const auto padSecPos = mapper.padSecPos(cru, fecInPartition, sampaOnFEC, channelOnSAMPA);
                const auto& padPos = padSecPos.getPadPos();
                int timebin = (globalBCoffset + zsdata->cont.header.bunchCrossing) / 8; // To be calculated

                // add digit
                processAttributes->digitsAll[sector].emplace_back(cruID, adcValue, padPos.getRow(), padPos.getPad(), timebin);
                ++processedChannels;
              }

              // go to next time bin
              zsdata = zsdata->next();
            }
          }

        } catch (const std::runtime_error& e) {
          LOG(alarm) << "can not create raw parser form input data";
          o2::header::hexDump("payload", input.payload, payloadSize, 64);
          LOG(alarm) << e.what();
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
