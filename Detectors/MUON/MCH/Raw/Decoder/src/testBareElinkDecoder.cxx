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

#define BOOST_TEST_MODULE Test MCHRaw BareElinkDecoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "BareElinkDecoder.h"
#include <boost/test/unit_test.hpp>
#include <fmt/printf.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace o2::mch::raw;
using ::operator<<;

SampaChannelHandler handlePacketPrint(std::string_view msg)
{
  return [msg](DsElecId dsId, DualSampaChannelId channel, SampaCluster sc) {
    std::stringstream s;
    s << dsId;
    std::cout << fmt::format("{} {} ch={:2d} ", msg, s.str(), channel);
    std::cout << sc << "\n";
  };
}

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(elinkdecoder)

BOOST_AUTO_TEST_CASE(Decoding10)
{
  int npackets{0};
  auto helper = handlePacketPrint("Decoding10:");

  auto hp = [&npackets, helper](DsElecId dsId, DualSampaChannelId channel, SampaCluster sh) {
    npackets++;
    helper(dsId, channel, sh);
  };

  DecodedDataHandlers rh;
  rh.sampaChannelHandler = hp;
  BareElinkDecoder<SampleMode> e(DsElecId{0, 0, 0}, rh);

  std::string enc("1100100010000000000011110000001010101010101010101011111010011100000000000010000000000000000000000000100000000000101000000010100000010000100100100000000000101000000000000000000000000100000000001001100000100110001010011000111110100110100000000000101100000000000000000000001100000000001000001000100000101010000010011000001001001000010110000000000011111000000000000000000000001000000000110110010011011001101101100101110110011111011001");

  for (int i = 0; i < enc.size() - 1; i += 2) {
    e.append(enc[i] == '1', enc[i + 1] == '1');
  }

  BOOST_CHECK_EQUAL(npackets, 4);
}

BOOST_AUTO_TEST_CASE(Decoding20)
{
  int npackets{0};
  auto helper = handlePacketPrint("Decoding20:");

  auto hp = [&npackets, helper](DsElecId dsId, DualSampaChannelId channel, SampaCluster sh) {
    npackets++;
    helper(dsId, channel, sh);
  };

  DecodedDataHandlers rh;
  rh.sampaChannelHandler = hp;
  BareElinkDecoder<ChargeSumMode> e(DsElecId{0, 0, 0}, rh);
  std::string enc("11001000100000000000111100000010101010101010101010110110100100100000000000100000000000000000000000001000000000001010000010100110000000000000010000100100100000000000101000000000000000000000001000000000001001100010011111100000000000000110100100100000000000101100000000000000000000001000000000001000001010000100101000000000110110100100100000000000111110000000000000000000001000000000001101100111011100001100000000");

  for (int i = 0; i < enc.size() - 1; i += 2) {
    e.append(enc[i] == '1', enc[i + 1] == '1');
  }

  BOOST_CHECK_EQUAL(npackets, 4);

  // same thing but with a decoder without a channel handler
  // so we don't "see" any packet in this case
  npackets = 0;
  BareElinkDecoder<ChargeSumMode> e2(DsElecId{0, 0, 0}, DecodedDataHandlers{});
  for (int i = 0; i < enc.size() - 1; i += 2) {
    e2.append(enc[i] == 1, enc[i + 1] == 1);
  }

  BOOST_CHECK_EQUAL(npackets, 0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
