// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw Decoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <fmt/printf.h>
#include "MCHRawCommon/RDHManip.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "Headers/RAWDataHeader.h"
#include "RefBuffers.h"
#include "BareGBTDecoder.h"
#include <random>

using namespace o2::mch::raw;
using o2::header::RAWDataHeaderV4;

namespace o2::header
{
extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);
}

SampaChannelHandler handlePacketStoreAsVec(std::vector<std::string>& result)
{
  return [&result](DsElecId dsId, uint8_t channel, SampaCluster sc) {
    result.emplace_back(fmt::format("{}-ch-{}-ts-{}-q-{}", asString(dsId), channel, sc.timestamp, sc.chargeSum));
  };
}

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(decoder)

BOOST_AUTO_TEST_CASE(Test1)
{
  std::vector<std::byte> buffer(1024);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 0xFF);
  std::generate(buffer.begin(), buffer.end(), [&] { return std::byte{dis(gen)}; });

  uint16_t dummyCruId{0};
  uint8_t dummyEndpoint{0};
  uint8_t dummyLinkId{0};
  uint16_t dummyFeeId{0};
  uint32_t orbit{12};
  uint16_t bunchCrossing{34};

  auto rdh = createRDH<RAWDataHeaderV4>(dummyCruId, dummyEndpoint, dummyLinkId, dummyFeeId,
                                        orbit, bunchCrossing, buffer.size());

  std::vector<std::byte> testBuffer;

  // duplicate the (rdh+payload) to fake a 3 rdhs buffer
  int nrdhs{3};
  for (auto i = 0; i < nrdhs; i++) {
    appendRDH(testBuffer, rdh);
    std::copy(begin(buffer), end(buffer), std::back_inserter(testBuffer));
  }

  int n = countRDHs<RAWDataHeaderV4>(testBuffer);

  BOOST_CHECK_EQUAL(n, nrdhs);

  //   showRDHs<RAWDataHeaderV4>(testBuffer);
}

bool testDecode(gsl::span<const std::byte> testBuffer)
{
  std::vector<std::string> result;
  std::vector<std::string> expected{

    "S728-J1-DS0-ch-3-ts-0-q-13",
    "S728-J1-DS0-ch-13-ts-0-q-133",
    "S728-J1-DS0-ch-23-ts-0-q-163",

    "S361-J0-DS4-ch-0-ts-0-q-10",
    "S361-J0-DS4-ch-1-ts-0-q-20",
    "S361-J0-DS4-ch-2-ts-0-q-30",
    "S361-J0-DS4-ch-3-ts-0-q-40",

    "S448-J6-DS2-ch-22-ts-0-q-420",
    "S448-J6-DS2-ch-23-ts-0-q-430",
    "S448-J6-DS2-ch-24-ts-0-q-440",
    "S448-J6-DS2-ch-25-ts-0-q-450",
    "S448-J6-DS2-ch-26-ts-0-q-460",
    "S448-J6-DS2-ch-42-ts-0-q-420"};

  auto pageDecoder = createPageDecoder(testBuffer, handlePacketStoreAsVec(result));

  auto parser = createPageParser(testBuffer);

  parser(testBuffer, pageDecoder);

  bool sameSize = result.size() == expected.size();
  bool permutation = std::is_permutation(begin(result), end(result), begin(expected));
  BOOST_CHECK_EQUAL(sameSize, true);
  BOOST_CHECK_EQUAL(permutation, true);
  if (!permutation || !sameSize) {
    std::cout << "Got:\n";
    for (auto s : result) {
      std::cout << s << "\n";
    }
    std::cout << "Expected:\n";
    for (auto s : expected) {
      std::cout << s << "\n";
    }
    return false;
  }
  return true;
}

BOOST_AUTO_TEST_CASE(TestBareDecoding)
{
  auto testBuffer = REF_BUFFER_CRU<BareFormat, ChargeSumMode>();
  int n = countRDHs<RAWDataHeaderV4>(testBuffer);
  BOOST_CHECK_EQUAL(n, 28);
  testDecode(testBuffer);
}

BOOST_AUTO_TEST_CASE(TestUserLogicDecoding)
{
  auto testBuffer = REF_BUFFER_CRU<UserLogicFormat, ChargeSumMode>();
  int n = countRDHs<RAWDataHeaderV4>(testBuffer);
  BOOST_CHECK_EQUAL(n, 4);
  testDecode(testBuffer);
}

BOOST_AUTO_TEST_CASE(BareGBTDecoderFromBuffer)
{
  std::vector<std::string> result;
  BareGBTDecoder<ChargeSumMode> dec(0, handlePacketStoreAsVec(result));
  auto testBuffer = REF_BUFFER_GBT<BareFormat, ChargeSumMode>();
  dec.append(testBuffer);
  std::vector<std::string> expected{
    "S0-J0-DS3-ch-63-ts-12-q-163",
    "S0-J0-DS3-ch-33-ts-12-q-133",
    "S0-J0-DS3-ch-13-ts-12-q-13",
    "S0-J0-DS0-ch-31-ts-12-q-160",
    "S0-J0-DS0-ch-0-ts-12-q-10"};
  BOOST_CHECK_EQUAL(result.size(), expected.size());
  BOOST_CHECK(std::is_permutation(begin(result), end(result), begin(expected)));
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
