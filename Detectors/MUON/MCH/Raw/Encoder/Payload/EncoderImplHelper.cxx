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
#include "EncoderImplHelper.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawElecMap/DsElecId.h"
#include "NofBits.h"

namespace o2::mch::raw::impl
{

uint16_t chipAddress(uint8_t elinkId, uint8_t chId)
{
  auto opt = o2::mch::raw::indexFromElinkId(elinkId);
  if (!opt.has_value()) {
    throw std::invalid_argument(fmt::format("elinkId {} is not valid", elinkId));
  }
  return opt.value() * 2 + (chId > 31);
}

SampaHeader buildSampaHeader(uint8_t elinkId, uint8_t chId, gsl::span<const SampaCluster> data)
{
  assertIsInRange("chId", chId, 0, 63);

  SampaHeader header;
  header.packetType(SampaPacketType::Data);

  uint16_t n10{0};
  for (const auto& s : data) {
    n10 += s.nof10BitWords();
  }
  assertNofBits("nof10BitWords", n10, 10);
  header.nof10BitWords(n10);
  header.chipAddress(chipAddress(elinkId, chId));
  header.channelAddress(chId % 32);
  header.hammingCode(computeHammingCode(header.uint64()));
  header.headerParity(computeHeaderParity(header.uint64()));

  //header.bunchCrossingCounter(mLocalBunchCrossing); //FIXME: how is this one evolving ?
  // FIXME: compute payload parity
  return header;
}

uint64_t build64(uint16_t a10, uint16_t b10 = 0, uint16_t c10 = 0, uint16_t d10 = 0, uint16_t e10 = 0)
{
  impl::assertIsInRange("a10", a10, 0, 1023);
  impl::assertIsInRange("b10", a10, 0, 1023);
  impl::assertIsInRange("c10", a10, 0, 1023);
  impl::assertIsInRange("d10", a10, 0, 1023);
  impl::assertIsInRange("e10", a10, 0, 1023);
  return (static_cast<uint64_t>(a10) << 40) |
         (static_cast<uint64_t>(b10) << 30) |
         (static_cast<uint64_t>(c10) << 20) |
         (static_cast<uint64_t>(d10) << 10) |
         (static_cast<uint64_t>(e10));
}

// ensures the buffer size is a multiple of 5
void addPadding(std::vector<uint10_t>& b10)
{
  while (b10.size() % 5) {
    b10.emplace_back(0);
  }
}

void b10to64(std::vector<uint10_t> b10, std::vector<uint64_t>& b64, uint16_t prefix14)
{
  uint64_t prefix = prefix14;
  prefix <<= 50;

  addPadding(b10);

  for (auto i = 0; i < b10.size(); i += 5) {
    uint64_t v = build64(b10[i + 4], b10[i + 3], b10[i + 2], b10[i + 1], b10[i + 0]);
    b64.emplace_back(v | prefix);
  }
}

void bufferizeClusters(gsl::span<const SampaCluster> clusters, std::vector<uint10_t>& b10)
{
  for (auto& c : clusters) {
    b10.emplace_back(c.nofSamples());
    b10.emplace_back(c.timestamp);
    if (c.isClusterSum()) {
      b10.emplace_back(c.chargeSum & 0x3FF);
      b10.emplace_back((c.chargeSum & 0xFFC00) >> 10);
    } else {
      for (auto& s : c.samples) {
        b10.emplace_back(s);
      }
    }
  }
}

void append(std::vector<uint10_t>& b10, uint50_t value)
{
  b10.emplace_back((value & 0x3FF));
  b10.emplace_back((value & 0xFFC00) >> 10);
  b10.emplace_back((value & 0x3FF00000) >> 20);
  b10.emplace_back((value & 0xFFC0000000) >> 30);
  b10.emplace_back((value & 0x3FF0000000000) >> 40);
}

void fillUserLogicBuffer10(std::vector<uint10_t>& b10,
                           gsl::span<const SampaCluster> clusters,
                           uint8_t elinkId,
                           uint8_t chId,
                           bool addSync)
{
  auto sh = buildSampaHeader(elinkId, chId, clusters);
  if (addSync) {
    addPadding(b10);
    append(b10, sampaSyncWord);
  }
  append(b10, sh.uint64());
  bufferizeClusters(clusters, b10);
}

} // namespace o2::mch::raw::impl
