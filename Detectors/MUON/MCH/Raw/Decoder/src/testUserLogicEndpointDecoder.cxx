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
#include "Assertions.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DumpBuffer.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawDecoder/SampaChannelHandler.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MoveBuffer.h"
#include "RDHManip.h"
#include "UserLogicEndpointDecoder.h"
#include <fmt/printf.h>
#include <fstream>
#include <iostream>

using namespace o2::mch::raw;
using o2::header::RAWDataHeaderV4;

SampaChannelHandler handlePacket(std::string& result)
{
  return [&result](DsElecId dsId, uint8_t channel, SampaCluster sc) {
    result += fmt::format("{}-ch-{}-ts-{}-q", asString(dsId), channel, sc.sampaTime);
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

std::vector<std::byte> convertBuffer2PayloadBuffer(gsl::span<const std::byte> buffer,
                                                   std::optional<size_t> insertSync = std::nullopt)
{
  // some gym to go from a buffer coming from the encoder,
  // which holds (DataBlockHeader,Payload) pairs in a byte buffer
  // to a buffer that holds just Payload information
  // to be fed to the decoder

  // strip the headers
  std::vector<std::byte> b8;
  forEachDataBlockRef(
    buffer, [&](const DataBlockRef& r) {
      auto& b = r.block;
      b8.insert(b8.end(), b.payload.begin(), b.payload.end());
    });

  // convert to a 64-bits buffer to be able to insert sync if needed
  std::vector<uint64_t> b64;
  impl::copyBuffer(b8, b64);

  // insert a sync at the given position if required
  if (insertSync.has_value() && insertSync.value() < b64.size()) {
    uint64_t prefix = b64[0] & 0xFFFC00000000000F;
    b64.insert(b64.begin() + insertSync.value(), prefix | sampaSyncWord);
  }

  impl::dumpBuffer<o2::mch::raw::UserLogicFormat>(b64);

  // get back to byte buffer to return
  std::vector<std::byte> bytes;
  impl::copyBuffer(b64, bytes);
  return bytes;
}

template <typename CHARGESUM>
std::string decodeBuffer(int feeId, gsl::span<const std::byte> buffer)
{
  std::string results;
  auto fee2solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  UserLogicEndpointDecoder<CHARGESUM> dec(feeId, fee2solar, handlePacket(results));
  dec.append(buffer);
  return results;
}

template <typename CHARGESUM>
std::string testPayloadDecode(DsElecId ds1,
                              int ch1,
                              const std::vector<SampaCluster>& clustersFirstChannel,
                              DsElecId ds2 = DsElecId{0, 0, 0},
                              int ch2 = 47,
                              const std::vector<SampaCluster>& clustersSecondChannel = {},
                              std::optional<size_t> insertSync = std::nullopt)
{
  auto encoder = createPayloadEncoder<UserLogicFormat, CHARGESUM, true>();

  encoder->startHeartbeatFrame(0, 0);

  auto solar2feelink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();

  uint16_t feeId{0};

  auto f1 = solar2feelink(ds1.solarId());
  if (!f1.has_value()) {
    throw std::invalid_argument("invalid solarId for ds1");
  }
  if (!clustersSecondChannel.empty()) {
    auto f2 = solar2feelink(ds2.solarId());
    if (!f2.has_value()) {
      throw std::invalid_argument("invalid solarId for ds2");
    }
    if (f2->feeId() != f1->feeId()) {
      throw std::invalid_argument("this test is only meant to work with 2 solars in the same cru endpoint");
    }
  }

  feeId = f1->feeId();

  encoder->addChannelData(ds1, ch1, clustersFirstChannel);
  if (!clustersSecondChannel.empty()) {
    encoder->addChannelData(ds2, ch2, clustersSecondChannel);
  }

  std::vector<std::byte> buffer;
  encoder->moveToBuffer(buffer);
  auto payloadBuffer = convertBuffer2PayloadBuffer(buffer, insertSync);

  return decodeBuffer<CHARGESUM>(feeId, payloadBuffer);
}

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(userlogicdsdecoder)

BOOST_AUTO_TEST_CASE(SampleModeSimplest)
{
  // only one channel with one very small cluster
  // fitting within one 64-bits word
  SampaCluster cl(345, 6789, {123, 456});
  auto r = testPayloadDecode<SampleMode>(DsElecId{728, 1, 0}, 63, {cl});
  BOOST_CHECK_EQUAL(r, "S728-J1-DS0-ch-63-ts-345-q-123-456\n");
}

BOOST_AUTO_TEST_CASE(SampleModeSimple)
{
  // only one channel with one cluster, but the cluster
  // spans 2 64-bits words.
  SampaCluster cl(345, 6789, {123, 456, 789, 901, 902});
  auto r = testPayloadDecode<SampleMode>(DsElecId{448, 6, 4}, 63, {cl});
  BOOST_CHECK_EQUAL(r, "S448-J6-DS4-ch-63-ts-345-q-123-456-789-901-902\n");
}

BOOST_AUTO_TEST_CASE(SampleModeTwoChannels)
{
  // 2 channels with one cluster
  SampaCluster cl(345, 6789, {123, 456, 789, 901, 902});
  SampaCluster cl2(346, 6789, {1001, 1002, 1003, 1004, 1005, 1006, 1007});
  auto r = testPayloadDecode<SampleMode>(DsElecId{361, 6, 2}, 63, {cl}, DsElecId{361, 6, 2}, 47, {cl2});
  BOOST_CHECK_EQUAL(r,
                    "S361-J6-DS2-ch-63-ts-345-q-123-456-789-901-902\n"
                    "S361-J6-DS2-ch-47-ts-346-q-1001-1002-1003-1004-1005-1006-1007\n");
}

BOOST_AUTO_TEST_CASE(ChargeSumModeSimplest)
{
  // only one channel with one cluster
  // (hence fitting within one 64 bits word)
  SampaCluster cl(345, 6789, 123456);
  auto r = testPayloadDecode<ChargeSumMode>(DsElecId{728, 1, 0}, 63, {cl});
  BOOST_CHECK_EQUAL(r, "S728-J1-DS0-ch-63-ts-345-q-123456\n");
}

BOOST_AUTO_TEST_CASE(ChargeSumModeSimple)
{
  // only one channel with 2 clusters
  // (hence spanning 2 64-bits words)
  SampaCluster cl1(345, 6789, 123456);
  SampaCluster cl2(346, 6789, 789012);
  auto r = testPayloadDecode<ChargeSumMode>(DsElecId{448, 6, 4}, 63, {cl1, cl2});
  BOOST_CHECK_EQUAL(r,
                    "S448-J6-DS4-ch-63-ts-345-q-123456\n"
                    "S448-J6-DS4-ch-63-ts-346-q-789012\n");
}

BOOST_AUTO_TEST_CASE(ChargeSumModeTwoChannels)
{
  // two channels with 2 clusters
  SampaCluster cl1(345, 6789, 123456);
  SampaCluster cl2(346, 6789, 789012);
  SampaCluster cl3(347, 6789, 1357);
  SampaCluster cl4(348, 6789, 7912);
  auto r = testPayloadDecode<ChargeSumMode>(DsElecId{361, 6, 2}, 63, {cl1, cl2}, DsElecId{361, 6, 2}, 47, {cl3, cl4});
  BOOST_CHECK_EQUAL(r,
                    "S361-J6-DS2-ch-63-ts-345-q-123456\n"
                    "S361-J6-DS2-ch-63-ts-346-q-789012\n"
                    "S361-J6-DS2-ch-47-ts-347-q-1357\n"
                    "S361-J6-DS2-ch-47-ts-348-q-7912\n");
}

BOOST_AUTO_TEST_CASE(SyncInTheMiddleChargeSumModeTwoChannels)
{
  // Insert a sync word in the middle of
  // the TwoChannels case and check the decoder is handling this fine
  // (by just returning to wait for sync mode, i.e. dropping the 2nd part
  // of the communication until a second sync)
  SampaCluster cl1(345, 6789, 123456);
  SampaCluster cl2(346, 6789, 789012);
  SampaCluster cl3(347, 6789, 1357);
  SampaCluster cl4(348, 6789, 7912);
  auto r = testPayloadDecode<ChargeSumMode>(
    DsElecId{361, 6, 2}, 63, {cl1, cl2},
    DsElecId{361, 6, 2}, 47, {cl3, cl4},
    5);
  BOOST_CHECK_EQUAL(r,
                    "S361-J6-DS2-ch-63-ts-345-q-123456\n"
                    "S361-J6-DS2-ch-63-ts-346-q-789012\n");
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
