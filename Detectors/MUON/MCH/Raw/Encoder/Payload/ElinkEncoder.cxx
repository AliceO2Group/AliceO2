// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Assertions.h"
#include "ElinkEncoder.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawElecMap/DsElecId.h"
#include "NofBits.h"

namespace
{
uint16_t chipAddress(uint8_t elinkId, uint8_t chId)
{
  auto opt = o2::mch::raw::indexFromElinkId(elinkId);
  if (!opt.has_value()) {
    throw std::invalid_argument(fmt::format("elinkId {} is not valid", elinkId));
  }
  return opt.value() * 2 + (chId > 31);
}
} // namespace

namespace o2::mch::raw
{

SampaHeader buildHeader(uint8_t elinkId, uint8_t chId, const std::vector<SampaCluster>& data)
{
  impl::assertIsInRange("chId", chId, 0, 63);

  SampaHeader header;
  header.packetType(SampaPacketType::Data);

  uint16_t n10{0};
  for (const auto& s : data) {
    n10 += s.nof10BitWords();
  }
  impl::assertNofBits("nof10BitWords", n10, 10);
  header.nof10BitWords(n10);
  header.chipAddress(chipAddress(elinkId, chId));
  header.channelAddress(chId % 32);
  header.hammingCode(computeHammingCode(header.uint64()));
  header.headerParity(computeHeaderParity(header.uint64()));

  //header.bunchCrossingCounter(mLocalBunchCrossing); //FIXME: how is this one evolving ?
  // FIXME: compute payload parity

  return header;
}

} // namespace o2::mch::raw
