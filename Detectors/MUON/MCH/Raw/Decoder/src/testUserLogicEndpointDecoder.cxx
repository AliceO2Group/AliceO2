// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#define BOOST_TEST_MODULE Test MCHRaw UserLogicElinkDecoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <fmt/printf.h>
#include "UserLogicEndpointDecoder.h"
#include "MCHRawCommon/SampaHeader.h"
#include "Assertions.h"
#include "MoveBuffer.h"
#include "DumpBuffer.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/SampaChannelHandler.h"
#include "MCHRawDecoder/PageDecoder.h"

using namespace o2::mch::raw;
using o2::header::RAWDataHeaderV4;

using uint10_t = uint16_t;
using uint50_t = uint64_t;

SampaChannelHandler handlePacket(std::string& result)
{
  return [&result](DsElecId dsId, uint8_t channel, SampaCluster sc) {
    result += fmt::format("{}-ch-{}-ts-{}-q", asString(dsId), channel, sc.timestamp);
    if (sc.isClusterSum()) {
      result += fmt::format("-{}", sc.chargeSum);
    } else {
      for (auto s : sc.samples) {
        result += fmt::format("-{}", s);
      }
    }
    result += "\n";
  };
}

SampaHeader createHeader(const std::vector<SampaCluster>& clusters)
{
  uint16_t n10{0};
  for (auto c : clusters) {
    n10 += c.nof10BitWords();
  }
  SampaHeader sh;
  sh.nof10BitWords(n10);
  sh.packetType(SampaPacketType::Data);
  sh.hammingCode(computeHammingCode(sh.uint64()));
  sh.headerParity(computeHeaderParity(sh.uint64()));
  return sh;
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

void dumpb10(const std::vector<uint10_t>& b10)
{
  for (auto i = 0; i < b10.size(); i++) {
    if (i % 5 == 0) {
      std::cout << "\nB10";
    }
    std::cout << fmt::format("{:4d} ", b10[i]);
  }
  std::cout << "\n";
}

std::vector<uint64_t> b10to64(std::vector<uint10_t> b10, uint16_t prefix14)
{
  uint64_t prefix = prefix14;
  prefix <<= 50;
  std::vector<uint64_t> b64;

  while (b10.size() % 5) {
    b10.emplace_back(0);
  }
  for (auto i = 0; i < b10.size(); i += 5) {
    uint64_t v = build64(b10[i + 4], b10[i + 3], b10[i + 2], b10[i + 1], b10[i + 0]);
    b64.emplace_back(v | prefix);
  }
  return b64;
}

void bufferizeClusters(const std::vector<SampaCluster>& clusters, std::vector<uint10_t>& b10)
{
  for (auto& c : clusters) {
    std::cout << "c=" << c << "\n";
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

std::vector<uint10_t> createBuffer10(const std::vector<SampaCluster>& clusters,
                                     uint8_t chip,
                                     uint8_t ch,
                                     bool sync)
{
  auto sh = createHeader(clusters);
  sh.chipAddress(chip);
  sh.channelAddress(ch);
  std::vector<uint10_t> b10;
  if (sync) {
    append(b10, sampaSyncWord);
  }
  append(b10, sh.uint64());
  bufferizeClusters(clusters, b10);
  return b10;
}

template <typename CHARGESUM>
void decodeBuffer(typename UserLogicEndpointDecoder<CHARGESUM>::ElinkDecoder& dec, const std::vector<uint64_t>& b64)
{
  std::vector<std::byte> b8;
  impl::copyBuffer(b64, b8);
  impl::dumpBuffer<o2::mch::raw::UserLogicFormat>(b8);
  constexpr uint64_t FIFTYBITSATONE = (static_cast<uint64_t>(1) << 50) - 1;
  for (auto b : b64) {
    uint64_t data50 = b & FIFTYBITSATONE;
    dec.append(data50, 0);
  }
}

constexpr uint8_t CHIP = 5;
constexpr uint8_t CHANNEL = 31;
constexpr uint16_t PREFIX = 24;
// exact value not relevant as long as it is non-zero,
// and is not setting the error bits that are used (for the moment only bit 52)
// Idea being to populate bits 50-63 with some data to ensure
// the decoder is only using the lower 50 bits to get the sync and
// header values, for instance.

std::vector<uint10_t> createBuffer10(const std::vector<SampaCluster>& clustersFirstChannel,
                                     const std::vector<SampaCluster>& clustersSecondChannel = {})
{
  bool sync{true};
  auto b10 = createBuffer10(clustersFirstChannel, CHIP, CHANNEL, sync);
  if (clustersSecondChannel.size()) {
    auto chip2 = CHIP;
    auto ch2 = CHANNEL / 2;
    auto b10_2 = createBuffer10(clustersSecondChannel, chip2, ch2, !sync);
    b10.insert(b10.end(), b10_2.begin(), b10_2.end());
  }
  return b10;
}

template <typename CHARGESUM>
std::string testPayloadDecode(const std::vector<SampaCluster>& clustersFirstChannel,
                              const std::vector<SampaCluster>& clustersSecondChannel = {},
                              std::optional<size_t> insertSync = std::nullopt)
{
  auto b10 = createBuffer10(clustersFirstChannel, clustersSecondChannel);
  auto b64 = b10to64(b10, PREFIX);
  if (insertSync.has_value() && insertSync.value() < b64.size()) {
    b64.insert(b64.begin() + insertSync.value(), (static_cast<uint64_t>(PREFIX) << 50) | sampaSyncWord);
  }
  std::string results;

  uint16_t dummySolarId{0};
  uint8_t dummyGroup{0};
  uint8_t index = (CHIP - (CHANNEL > 32)) / 2;
  DsElecId dsId{dummySolarId, dummyGroup, index};
  typename UserLogicEndpointDecoder<CHARGESUM>::ElinkDecoder dec(dsId, handlePacket(results));
  decodeBuffer<CHARGESUM>(dec, b64);
  return results;
}

template <typename CHARGESUM>
void testDecode(const std::vector<SampaCluster>& clustersFirstChannel,
                const std::vector<SampaCluster>& clustersSecondChannel = {})
{
  auto b10 = createBuffer10(clustersFirstChannel, clustersSecondChannel);
  auto b64 = b10to64(b10, PREFIX);

  std::vector<std::byte> b8;
  impl::copyBuffer(b64, b8);

  std::vector<std::byte> buffer;
  uint8_t linkIDforUL = 15; // must be 15
  uint16_t feeId = 18;
  uint16_t cruId = feeId / 2;
  uint8_t endpoint = 0;
  auto rdh = createRDH<RAWDataHeaderV4>(cruId, endpoint, linkIDforUL, feeId, 12, 34, b8.size());
  appendRDH(buffer, rdh);
  buffer.insert(buffer.end(), b8.begin(), b8.end());
  const auto handlePacket = [](DsElecId dsId, uint8_t channel, SampaCluster sc) {
    std::cout << fmt::format("testDecode:{}-{}\n", asString(dsId), asString(sc));
  };

  impl::dumpBuffer<o2::mch::raw::UserLogicFormat>(buffer);

  gsl::span<const std::byte> page(reinterpret_cast<const std::byte*>(buffer.data()), buffer.size());
  auto decoder = createPageDecoder(page, handlePacket);
  decoder(page);
}

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(userlogicdsdecoder)

BOOST_AUTO_TEST_CASE(SampleModeSimplest)
{
  // only one channel with one very small cluster
  // fitting within one 64-bits word
  SampaCluster cl(345, {123, 456});
  auto r = testPayloadDecode<SampleMode>({cl});
  BOOST_CHECK_EQUAL(r, "S0-J0-DS2-ch-63-ts-345-q-123-456\n");
}

BOOST_AUTO_TEST_CASE(SampleModeSimple)
{
  // only one channel with one cluster, but the cluster
  // spans 2 64-bits words.
  SampaCluster cl(345, {123, 456, 789, 901, 902});
  auto r = testPayloadDecode<SampleMode>({cl});
  BOOST_CHECK_EQUAL(r, "S0-J0-DS2-ch-63-ts-345-q-123-456-789-901-902\n");
}

BOOST_AUTO_TEST_CASE(SampleModeTwoChannels)
{
  // 2 channels with one cluster
  SampaCluster cl(345, {123, 456, 789, 901, 902});
  SampaCluster cl2(346, {1001, 1002, 1003, 1004, 1005, 1006, 1007});
  auto r = testPayloadDecode<SampleMode>({cl}, {cl2});
  BOOST_CHECK_EQUAL(r,
                    "S0-J0-DS2-ch-63-ts-345-q-123-456-789-901-902\n"
                    "S0-J0-DS2-ch-47-ts-346-q-1001-1002-1003-1004-1005-1006-1007\n");
}

BOOST_AUTO_TEST_CASE(ChargeSumModeSimplest)
{
  // only one channel with one cluster
  // (hence fitting within one 64 bits word)
  SampaCluster cl(345, 123456);
  auto r = testPayloadDecode<ChargeSumMode>({cl});
  BOOST_CHECK_EQUAL(r, "S0-J0-DS2-ch-63-ts-345-q-123456\n");
}

BOOST_AUTO_TEST_CASE(ChargeSumModeSimple)
{
  // only one channel with 2 clusters
  // (hence spanning 2 64-bits words)
  SampaCluster cl1(345, 123456);
  SampaCluster cl2(346, 789012);
  auto r = testPayloadDecode<ChargeSumMode>({cl1, cl2});
  BOOST_CHECK_EQUAL(r,
                    "S0-J0-DS2-ch-63-ts-345-q-123456\n"
                    "S0-J0-DS2-ch-63-ts-346-q-789012\n");
}

BOOST_AUTO_TEST_CASE(ChargeSumModeTwoChannels)
{
  // two channels with 2 clusters
  SampaCluster cl1(345, 123456);
  SampaCluster cl2(346, 789012);
  SampaCluster cl3(347, 1357);
  SampaCluster cl4(348, 791);
  auto r = testPayloadDecode<ChargeSumMode>({cl1, cl2}, {cl3, cl4});
  BOOST_CHECK_EQUAL(r,
                    "S0-J0-DS2-ch-63-ts-345-q-123456\n"
                    "S0-J0-DS2-ch-63-ts-346-q-789012\n"
                    "S0-J0-DS2-ch-47-ts-347-q-1357\n"
                    "S0-J0-DS2-ch-47-ts-348-q-791\n");
}

BOOST_AUTO_TEST_CASE(DecodeSampleModeSimplest)
{
  // only one channel with one very small cluster
  // fitting within one 64-bits word
  SampaCluster cl(345, {123, 456});
  testDecode<SampleMode>({cl});
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(SyncInTheMiddleChargeSumModeTwoChannels)
{
  // Insert a sync word in the middle of
  // the TwoChannels case and check the decoder is handling this fine
  // (by just returning to wait for sync mode, i.e. dropping the 2nd part
  // of the communication until a second sync)
  SampaCluster cl1(345, 123456);
  SampaCluster cl2(346, 789012);
  SampaCluster cl3(347, 1357);
  SampaCluster cl4(348, 791);
  auto r = testPayloadDecode<ChargeSumMode>({cl1, cl2}, {cl3, cl4},
                                            5);
  BOOST_CHECK_EQUAL(r,
                    "S0-J0-DS2-ch-63-ts-345-q-123456\n"
                    "S0-J0-DS2-ch-63-ts-346-q-789012\n");
}

std::string asBinary(uint64_t value)
{
  std::string s;
  uint64_t one = 1;
  int space{0};
  for (auto i = 63; i >= 0; i--) {
    if (value & (one << i)) {
      s += "1";
    } else {
      s += "0";
    }
    if (i % 4 == 0 && i) {
      s += " ";
    }
  }
  return s;
}

std::string binaryRule(bool top, std::vector<int> stops)
{
  std::string s;
  for (auto i = 63; i >= 0; i--) {
    if (std::find(stops.begin(), stops.end(), i) != stops.end()) {
      s += fmt::format("{}", (top ? i / 10 : i % 10));
    } else {
      s += " ";
    }
    if (i % 4 == 0) {
      s += " ";
    }
  }
  return s;
}

void dump(uint64_t value, std::vector<int> stops = {63, 47, 31, 15, 0})
{
  std::cout << asBinary(value) << "\n";
  std::cout << binaryRule(true, stops) << "\n";
  std::cout << binaryRule(false, stops) << "\n";
}

BOOST_AUTO_TEST_CASE(Prefix)
{
  std::vector<uint10_t> b10;
  append(b10, sampaSyncWord);
  auto b64 = b10to64(b10, 24);
  auto w = b64[0];
  BOOST_CHECK_EQUAL(asBinary(w), "0000 0000 0110 0001 0101 0101 0101 0101 0100 0000 1111 0000 0000 0001 0001 0011");
}

BOOST_AUTO_TEST_CASE(Binary)
{
  dump(static_cast<uint64_t>(7) << 50, {50, 51, 52, 0});
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
