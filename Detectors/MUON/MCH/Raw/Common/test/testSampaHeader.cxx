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

#define BOOST_TEST_MODULE Test MCHRaw SampaHeader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawCommon/SampaHeader.h"

using namespace o2::mch::raw;

uint64_t allones = 0x3FFFFFFFFFFFF;
uint64_t allbutbx = 0x200001FFFFFFF;
uint64_t allbut10bits = 0x3FFFFFFF003FF;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(sampaheader)

BOOST_AUTO_TEST_CASE(SampaHeaderCtorBunchCrossingCounter)
{
  SampaHeader expected(static_cast<uint8_t>(0x3F),
                       true,
                       SampaPacketType::DataTriggerTooEarlyNumWords,
                       static_cast<uint16_t>(0x3FF),
                       static_cast<uint16_t>(0xF),
                       static_cast<uint16_t>(0x1F),
                       static_cast<uint32_t>(0),
                       true);
  BOOST_CHECK_EQUAL(expected.uint64(), allbutbx);
}

BOOST_AUTO_TEST_CASE(SampaHeaderCtorNof10BitsWords)
{
  SampaHeader expected(static_cast<uint8_t>(0x3F),
                       true,
                       SampaPacketType::DataTriggerTooEarlyNumWords,
                       static_cast<uint16_t>(0),
                       static_cast<uint16_t>(0xF),
                       static_cast<uint16_t>(0x1F),
                       static_cast<uint32_t>(0xFFFFF),
                       true);
  BOOST_CHECK_EQUAL(expected.uint64(), allbut10bits);
}

BOOST_AUTO_TEST_CASE(SampaHeaderEqualityOperators)
{
  // comparison is full comparison (i.e. equality of 50bits)

  SampaHeader h(sampaSync());

  BOOST_CHECK(h == sampaSync());

  SampaHeader h2(UINT64_C(0x1fffff5f0007f));

  BOOST_CHECK(h2 != sampaSync());
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetHamming)
{
  SampaHeader sh;

  sh.hammingCode(0x3F);
  BOOST_CHECK_EQUAL(sh.hammingCode(), 0X3F);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetHeaderParity)
{
  SampaHeader sh;

  sh.headerParity(true);
  BOOST_CHECK_EQUAL(sh.headerParity(), true);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetPacketType)
{
  SampaHeader sh;

  sh.packetType(SampaPacketType::DataTriggerTooEarlyNumWords);
  BOOST_CHECK(sh.packetType() == SampaPacketType::DataTriggerTooEarlyNumWords);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetNumberOf10BitsWords)
{
  SampaHeader sh;

  sh.nof10BitWords(0x3FF);
  BOOST_CHECK_EQUAL(sh.nof10BitWords(), 0x3FF);
  sh.nof10BitWords(0);
  BOOST_CHECK_EQUAL(sh.nof10BitWords(), 0);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetChipAddress)
{
  SampaHeader sh;

  sh.chipAddress(0xF);
  BOOST_CHECK_EQUAL(sh.chipAddress(), 0xF);
  sh.chipAddress(0);
  BOOST_CHECK_EQUAL(sh.chipAddress(), 0);
  sh.chipAddress(1);
  BOOST_CHECK_EQUAL(sh.chipAddress(), 1);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetChannelAddress)
{
  SampaHeader sh;

  sh.channelAddress(0x1F);
  BOOST_CHECK_EQUAL(sh.channelAddress(), 0x1F);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetBunchCrossingCounter)
{
  SampaHeader sh;

  sh.bunchCrossingCounter(0xFFFFF);
  BOOST_CHECK_EQUAL(sh.bunchCrossingCounter(), 0xFFFFF);
}

BOOST_AUTO_TEST_CASE(SampaHeaderSetPayloadParity)
{
  SampaHeader sh;

  sh.payloadParity(true);
  BOOST_CHECK_EQUAL(sh.payloadParity(), true);
}

BOOST_AUTO_TEST_CASE(SampaHeaderLessThanOperatorComparesBx)
{
  SampaHeader h1;
  SampaHeader h2;

  h1.bunchCrossingCounter(1);
  h2.bunchCrossingCounter(2);

  SampaHeader h10{h1};

  BOOST_CHECK(h1 == h10);

  BOOST_CHECK_EQUAL(h1 > h2, false);
  BOOST_CHECK_EQUAL(h1 < h2, true);
  BOOST_CHECK_EQUAL(h1 <= h10, true);
  BOOST_CHECK_EQUAL(h1 >= h10, true);
}

BOOST_AUTO_TEST_CASE(SampaHeaderCtorWithMoreThan50BitsShouldThrow)
{
  BOOST_CHECK_THROW(SampaHeader(static_cast<uint64_t>(1) << 50), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(SampaHeaderCtorWithInvalidBitsIsNotAHeartbeat)
{
  uint64_t h = 0x3FFFFEAFFFFFF; // completely invalid value to start with
  uint64_t one = 1;

  // - bits 7-9 must be zero
  // - bits 10-19 must be zero
  // - bits 24,26,28 must be one
  // - bits 25,27 must be zero
  // - bit 49 must be zero

  std::vector<int> zeros = {7, 8, 9, 24, 26, 28, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 27, 49};
  std::vector<int> ones = {24, 26, 28};

  BOOST_CHECK_EQUAL(SampaHeader(h).isHeartbeat(), false);
  for (auto ix : zeros) {
    h &= ~(one << ix);
    BOOST_CHECK_EQUAL(SampaHeader(h).isHeartbeat(), false);
  }
  for (auto ix : ones) {
    h |= (one << ix);
  }
  BOOST_CHECK_EQUAL(SampaHeader(h).isHeartbeat(), true);
}

BOOST_AUTO_TEST_CASE(CheckSampaSyncIsExpectedValue)
{
  SampaHeader h(0x1555540f00113);
  BOOST_CHECK(sampaSync() == h);
}

BOOST_AUTO_TEST_CASE(SetChannelAddressTwice)
{
  SampaHeader h;

  h.packetType(SampaPacketType::Data);
  h.channelAddress(1);
  BOOST_CHECK_EQUAL(h.channelAddress(), 1);
  h.channelAddress(31);
  BOOST_CHECK_EQUAL(h.channelAddress(), 31);
  h.channelAddress(5);
  BOOST_CHECK_EQUAL(h.channelAddress(), 5);
}

BOOST_AUTO_TEST_CASE(ComputeHammingCode)
{
  BOOST_CHECK_EQUAL(computeHammingCode(0x3722e80103208), 0x8);  // 000100 P0
  BOOST_CHECK_EQUAL(computeHammingCode(0x1722e9f00327d), 0x3D); // 101101 P1
  BOOST_CHECK_EQUAL(computeHammingCode(0x1722e8090322f), 0x2F); // 111101 P0
}

BOOST_AUTO_TEST_CASE(ComputeHammingCode2)
{
  BOOST_CHECK_EQUAL(computeHammingCode2(0x3722e80103208), 0x8);  // 000100 P0
  BOOST_CHECK_EQUAL(computeHammingCode2(0x1722e9f00327d), 0x3D); // 101101 P1
  BOOST_CHECK_EQUAL(computeHammingCode2(0x1722e8090322f), 0x2F); // 111101 P0
}

BOOST_AUTO_TEST_CASE(ComputeHammingCode3)
{
  BOOST_CHECK_EQUAL(computeHammingCode3(0x3722e80103208), 0x8);  // 000100 P0
  BOOST_CHECK_EQUAL(computeHammingCode3(0x1722e9f00327d), 0x3D); // 101101 P1
  BOOST_CHECK_EQUAL(computeHammingCode3(0x1722e8090322f), 0x2F); // 111101 P0
}

BOOST_AUTO_TEST_CASE(ComputeHammingCode4)
{
  BOOST_CHECK_EQUAL(computeHammingCode4(0x3722e80103208), 0x8);  // 000100 P0
  BOOST_CHECK_EQUAL(computeHammingCode4(0x1722e9f00327d), 0x3D); // 101101 P1
  BOOST_CHECK_EQUAL(computeHammingCode4(0x1722e8090322f), 0x2F); // 111101 P0
}

BOOST_AUTO_TEST_CASE(CheckHammingCodeError)
{
  uint64_t v = 0x1722e9f00327d;
  int expected = 0x3D;

  auto ref = computeHammingCode(v);
  BOOST_CHECK_EQUAL(ref, expected);

  const uint64_t one{1};
  // flip a data bit
  v ^= (one << 34);
  auto h = computeHammingCode(v);
  BOOST_CHECK_NE(ref, h);
}

BOOST_AUTO_TEST_CASE(CheckHeaderParity)
{
  BOOST_CHECK_EQUAL(computeHeaderParity(0x3722e80103208), 0); // 000100 P0
  BOOST_CHECK_EQUAL(computeHeaderParity(0x1722e8090322f), 0); // 111101 P0
  BOOST_CHECK_EQUAL(computeHeaderParity(0x1722e9f00327d), 1); // 101101 P1
}

BOOST_AUTO_TEST_CASE(CheckHeaderParity2)
{
  BOOST_CHECK_EQUAL(computeHeaderParity2(0x3722e80103208), 0); // 000100 P0
  BOOST_CHECK_EQUAL(computeHeaderParity2(0x1722e8090322f), 0); // 111101 P0
  BOOST_CHECK_EQUAL(computeHeaderParity2(0x1722e9f00327d), 1); // 101101 P1
}

BOOST_AUTO_TEST_CASE(CheckHeaderParity3)
{
  BOOST_CHECK_EQUAL(computeHeaderParity3(0x3722e80103208), 0); // 000100 P0
  BOOST_CHECK_EQUAL(computeHeaderParity3(0x1722e8090322f), 0); // 111101 P0
  BOOST_CHECK_EQUAL(computeHeaderParity3(0x1722e9f00327d), 1); // 101101 P1
}

BOOST_AUTO_TEST_CASE(CheckHeaderParity4)
{
  BOOST_CHECK_EQUAL(computeHeaderParity4(0x3722e80103208), 0); // 000100 P0
  BOOST_CHECK_EQUAL(computeHeaderParity4(0x1722e8090322f), 0); // 111101 P0
  BOOST_CHECK_EQUAL(computeHeaderParity4(0x1722e9f00327d), 1); // 101101 P1
}

BOOST_AUTO_TEST_CASE(CreateHearbeat)
{
  SampaHeader h = sampaHeartbeat(0, 0);
  BOOST_CHECK_EQUAL(h.isHeartbeat(), true);
  h = sampaHeartbeat(39, 12345);
  BOOST_CHECK_EQUAL(h.isHeartbeat(), true);
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
