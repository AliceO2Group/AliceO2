// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <array>

#include "Framework/Logger.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"

#include "TPCReconstruction/RawProcessingHelpers.h"

using namespace o2::tpc;

//______________________________________________________________________________
bool raw_processing_helpers::processZSdata(const char* data, size_t size, rdh_utils::FEEIDType feeId, uint32_t globalBCoffset, ADCCallback fillADC, bool useTimeBin)
{
  const auto& mapper = Mapper::instance();

  const auto link = rdh_utils::getLink(feeId);
  const auto cruID = rdh_utils::getCRU(feeId);
  const auto endPoint = rdh_utils::getEndPoint(feeId);
  const CRU cru(cruID);
  const int fecLinkOffsetCRU = (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2;
  const int fecInPartition = link + endPoint * fecLinkOffsetCRU;

  // temporarily store the sync offset until it is available in the ZS header
  // WARNING: This only works until the TB counter wrapped afterwards the alignment might change
  const int globalLinkID = int(link) + int(endPoint * 12);
  const int tpcGlobalLinkID = cruID * 24 + globalLinkID;
  static std::array<uint32_t, 360 * 24> syncOffsetLinks;

  bool hasData{false};

  zerosupp_link_based::ContainerZS* zsdata = (zerosupp_link_based::ContainerZS*)data;
  const zerosupp_link_based::ContainerZS* const zsdataEnd = (zerosupp_link_based::ContainerZS*)(data + size);

  while (zsdata < zsdataEnd) {
    const auto& header = zsdata->cont.header;

    // align to header word if needed
    if (!header.hasCorrectMagicWord()) {
      zsdata = (zerosupp_link_based::ContainerZS*)((const char*)zsdata + sizeof(zerosupp_link_based::Header));
      if (!header.isFillWord()) {
        LOGP(error, "Bad LinkZS magic word (0x{:08x}), for feeId 0x{:05x} (CRU: {:3}, link: {:2}, EP {}) , skipping data block", header.magicWord, feeId, rdh_utils::getCRU(feeId), rdh_utils::getLink(feeId), rdh_utils::getEndPoint(feeId));
        LOGP(error, "Full 128b word is: 0x{:016x}{:016x}", header.word1, header.word0);
      }
      continue;
    }

    const auto channelBits = zsdata->getChannelBits();
    const uint32_t expectedWords = std::ceil(channelBits.count() * 0.1f);
    const uint32_t numberOfWords = zsdata->getDataWords();
    assert(expectedWords == numberOfWords);

    const uint32_t bunchCrossingHeader = zsdata->getBunchCrossing();
    uint32_t syncOffset = header.syncOffsetBC % 16;

    if (useTimeBin) {
      const uint32_t timebinHeader = (header.syncOffsetCRUCycles << 8) | header.syncOffsetBC;
      if (syncOffsetLinks[tpcGlobalLinkID] == 0) {
        syncOffsetLinks[tpcGlobalLinkID] = (bunchCrossingHeader + 3564 - (timebinHeader * 8) % 3564) % 3564 % 16;
      }
      syncOffset = syncOffsetLinks[tpcGlobalLinkID];
    }

    const int timebin = (int(globalBCoffset) + int(bunchCrossingHeader) - int(syncOffset)) / 8;
    if (timebin < 0) {
      LOGP(info, "skipping negative time bin with (globalBCoffset ({}) + bunchCrossingHeader ({}) - syncOffset({})) / 8 = {}", globalBCoffset, bunchCrossingHeader, syncOffset, timebin);

      // go to next time bin
      zsdata = zsdata->next();
      continue;
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
