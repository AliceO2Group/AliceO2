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

#include <array>
#include <chrono>
#include <fmt/format.h>
#include <fmt/chrono.h>

#include "CommonConstants/LHCConstants.h"
#include "Framework/Logger.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/Constants.h"

#include "TPCReconstruction/RawProcessingHelpers.h"

using namespace o2::tpc;

//______________________________________________________________________________
bool raw_processing_helpers::processZSdata(const char* data, size_t size, rdh_utils::FEEIDType feeId, uint32_t orbit, uint32_t referenceOrbit, uint32_t syncOffsetReference, ADCCallback fillADC)
{
  const auto& mapper = Mapper::instance();

  const auto link = rdh_utils::getLink(feeId);
  const auto cruID = rdh_utils::getCRU(feeId);
  const auto endPoint = rdh_utils::getEndPoint(feeId);
  const CRU cru(cruID);
  const int fecLinkOffsetCRU = (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2;
  int fecInPartition = link + endPoint * fecLinkOffsetCRU;

  // temporarily store the sync offset until it is available in the ZS header
  // WARNING: This only works until the TB counter wrapped afterwards the alignment might change
  const int globalLinkID = int(link) + int(endPoint * 12);
  const int tpcGlobalLinkID = cruID * 24 + globalLinkID;
  static std::array<uint32_t, 360 * 24> syncOffsetLinks;

  const uint32_t maxBunches = (uint32_t)o2::constants::lhc::LHCMaxBunches;
  const int globalBCOffset = int(orbit - referenceOrbit) * o2::constants::lhc::LHCMaxBunches;
  static int triggerBCOffset = 0;

  bool hasData{false};

  zerosupp_link_based::ContainerZS* zsdata = (zerosupp_link_based::ContainerZS*)data;
  const zerosupp_link_based::ContainerZS* const zsdataEnd = (zerosupp_link_based::ContainerZS*)(data + size);

  int zsVersion = -1;
  int timeOffset = 0;

  while (zsdata < zsdataEnd) {
    const auto& header = zsdata->cont.header;

    // align to header word if needed
    if (!header.hasCorrectMagicWord()) {
      zsdata = (zerosupp_link_based::ContainerZS*)((const char*)zsdata + sizeof(zerosupp_link_based::Header));
      if (!header.isFillWord()) {
        LOGP(error, "Bad LinkZS magic word (0x{:08x}), for feeId 0x{:05x} (CRU: {:3}, link: {:2}, EP {}), orbit {} , skipping data block", header.magicWord, feeId, cruID, link, endPoint, orbit);
        LOGP(error, "Full 128b word is: 0x{:016x}{:016x}", header.word1, header.word0);
      }
      continue;
    }

    // set trigger offset and skip trigger info
    if (header.isTriggerInfo()) {
      // for the moment only skip the trigger info
      const auto triggerInfo = (zerosupp_link_based::TriggerContainer*)zsdata;
      const auto triggerOrbit = triggerInfo->triggerInfo.getOrbit();
      const auto triggerBC = triggerInfo->triggerInfo.bunchCrossing;
      triggerBCOffset = (int(triggerOrbit) - int(referenceOrbit)) * maxBunches + triggerBC;
      LOGP(debug, "orbit: {}, triggerOrbit: {}, triggerBC: {}, triggerBCOffset: {}", orbit, triggerOrbit, triggerBC, triggerBCOffset);
      zsdata = zsdata->next();
      continue;
    } else if (header.isTriggerInfoV2()) {
      // for the moment only skip the trigger info
      const auto triggerInfo = (zerosupp_link_based::TriggerInfoV2*)zsdata;
      const auto triggerOrbit = triggerInfo->orbit;
      const auto triggerBC = triggerInfo->bunchCrossing;
      triggerBCOffset = (int(triggerOrbit) - int(referenceOrbit)) * maxBunches + triggerBC;
      LOGP(debug, "orbit: {}, triggerOrbit: {}, triggerBC: {}, triggerBCOffset: {}", orbit, triggerOrbit, triggerBC, triggerBCOffset);
      zsdata = zsdata->next();
      continue;
    } else if (header.isMetaHeader()) {
      const auto& metaHDR = *((TPCZSHDRV2*)zsdata);
      zsVersion = metaHDR.version;
      timeOffset = metaHDR.timeOffset;

      const auto& triggerInfo = *(zerosupp_link_based::TriggerInfoV3*)((const char*)&metaHDR + sizeof(metaHDR));
      if (triggerInfo.hasTrigger()) {
        const auto triggerBC = triggerInfo.getFirstBC();
        const auto triggerOrbit = orbit;
        triggerBCOffset = (int(triggerOrbit) - int(referenceOrbit)) * maxBunches + triggerBC;
      }

      zsdata = (zerosupp_link_based::ContainerZS*)((const char*)&triggerInfo + sizeof(triggerInfo));
      continue;
    }

    const auto channelBits = zsdata->getChannelBits();
    const uint32_t expectedWords = std::ceil(channelBits.count() * 0.1f);
    const uint32_t numberOfWords = zsdata->getDataWords();
    assert(expectedWords == numberOfWords);

    const auto bunchCrossingHeader = int(zsdata->getBunchCrossing());
    const auto syncOffset = int(header.syncOffsetBC);

    // in case of old data, alignment must be done in software
    if (zsVersion < 0) {
      timeOffset = syncOffsetReference - syncOffset;
    }

    const int bcOffset = timeOffset + globalBCOffset + bunchCrossingHeader - triggerBCOffset;
    if (bcOffset < 0) {
      using namespace std::literals::chrono_literals;
      static std::chrono::time_point<std::chrono::steady_clock> lastReport = std::chrono::steady_clock::now();
      const auto now = std::chrono::steady_clock::now();
      static size_t reportedErrors = 0;
      const size_t MAXERRORS = 10;
      const auto sleepTime = 10min;

      if ((now - lastReport) < sleepTime) {
        if (reportedErrors < MAXERRORS) {
          ++reportedErrors;
          std::string sleepInfo;
          if (reportedErrors == MAXERRORS) {
            sleepInfo = fmt::format(", maximum error count ({}) reached, not reporting for the next {}", MAXERRORS, sleepTime);
          }
          LOGP(warning, "skipping time bin with negative BC offset timeOffset {} + globalBCoffset (({} - {}) * {} = {}) + bunchCrossingHeader ({}) - triggerBCOffset({}) = {}{}",
               timeOffset, orbit, referenceOrbit, o2::constants::lhc::LHCMaxBunches, globalBCOffset, bunchCrossingHeader, triggerBCOffset, bcOffset, sleepInfo);
          lastReport = now;
        }
      } else {
        lastReport = now;
        reportedErrors = 0;
      }

      // go to next time bin
      zsdata = zsdata->next();
      continue;
    }

    const int timebin = bcOffset / constants::LHCBCPERTIMEBIN;
    if (zsVersion == ZSVersionLinkBasedWithMeta) {
      fecInPartition = header.fecInPartition;
    }

    std::size_t processedChannels = 0;
    for (std::size_t ichannel = 0; ichannel < channelBits.size(); ++ichannel) {
      if (!channelBits[ichannel]) {
        continue;
      }

      // adc value
      const auto adcValue = zsdata->getADCValueFloat(processedChannels);

      // mapping to row, pad sector
      int sampaOnFEC{}, channelOnSAMPA{};
      Mapper::getSampaAndChannelOnFEC(cruID, ichannel, sampaOnFEC, channelOnSAMPA);
      const auto padSecPos = mapper.padSecPos(cru, fecInPartition, sampaOnFEC, channelOnSAMPA);
      const auto& padPos = padSecPos.getPadPos();

      // add digit using callback
      fillADC(int(cruID), int(padPos.getRow()), int(padPos.getPad()), timebin, adcValue);

      ++processedChannels;
      hasData = true;
    }

    // go to next time bin
    zsdata = zsdata->next();
  }

  return hasData;
}
