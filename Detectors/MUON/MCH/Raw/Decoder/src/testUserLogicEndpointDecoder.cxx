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
#include "MCHRawDecoder/DecodedDataHandlers.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MoveBuffer.h"
#include "RDHManip.h"
#include "UserLogicEndpointDecoder.h"
#include <fmt/printf.h>
#include <fstream>
#include <iostream>
#include <boost/test/data/test_case.hpp>
#include <boost/mpl/list.hpp>

using namespace o2::mch::raw;
namespace bdata = boost::unit_test::data;

const uint64_t CruPageOK[] = {
  0x00000A0000124006ul,
  0x000C4C0F00A000A0ul,
  0x010E853D00000570ul,
  0x0000000000000000ul,
  0x0000000000006000ul,
  0x0000000000000000ul,
  0x0000000000000000ul,
  0x0000000000000000ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1555540F00113ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x3F04ECA103E5Cul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x0000040215C0Dul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x00000C0301004ul,
  ((0x0204ul << 50) & 0xFFFC000000000000ul) + 0x0000000000400ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1555540F00113ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1F080CA100E4Dul,
  ((0x0204ul << 50) & 0xFFFC000000000000ul) + 0x00044C0100001ul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul};

const uint64_t CruPageBadClusterSize[] = {
  0x00000A0000124006ul,
  0x000C4C0F00A000A0ul,
  0x010E853D00000570ul,
  0x0000000000000000ul,
  0x0000000000006000ul,
  0x0000000000000000ul,
  0x0000000000000000ul,
  0x0000000000000000ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1555540F00113ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x3F04ECA103E5Cul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x0000040215C0Eul, // <== the cluster size is increased from 13 (0xD) to 14 (0xE)
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x00000C0301004ul, // now the cluster size does not match anymore with the
  ((0x0204ul << 50) & 0xFFFC000000000000ul) + 0x0000000000400ul, // number of 10-bit words in the SAMPA header, which will trigger
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1555540F00113ul, // a ErrorBadClusterSize error.
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1F080CA100E4Dul,
  ((0x0204ul << 50) & 0xFFFC000000000000ul) + 0x00044C0100001ul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul};

const uint64_t CruPageBadN10bitWords[] = {
  0x00000A0000124006ul,
  0x000C4C0F00A000A0ul,
  0x010E853D00000570ul,
  0x0000000000000000ul,
  0x0000000000006000ul,
  0x0000000000000000ul,
  0x0000000000000000ul,
  0x0000000000000000ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1555540F00113ul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x3F04ECA103E5Cul,
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x0000040215C08ul, // <== the cluster size is decreased from 13 (0xD) to 8 (0x8)
  //((0x0200ul<<50)&0xFFFC000000000000ul) + 0x00000C0301004ul, // and one 50-bit word is removed. In this case the cluster
  ((0x0204ul << 50) & 0xFFFC000000000000ul) + 0x0000000000400ul, // size matches the number of samples in the data, but the
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1555540F00113ul, // end of the SAMPA packet arrives too early with respect to
  ((0x0200ul << 50) & 0xFFFC000000000000ul) + 0x1F080CA100E4Dul, // the number of 10-bit words in the SAMPA header. This will
  ((0x0204ul << 50) & 0xFFFC000000000000ul) + 0x00044C0100001ul, // trigger a ErrorBadIncompleteWord error.
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul,
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul, // <== a word is added at the end in order to match the
  ((0x3FBBul << 50) & 0xFFFC000000000000ul) + 0x1DEEDFEEDDEEDul  // payload size in the RDH
};

SampaChannelHandler handlePacket(std::string& result)
{
  return [&result](DsElecId dsId, DualSampaChannelId channel, SampaCluster sc) {
    result += fmt::format("{}-ch-{}-ts-{}-q", asString(dsId), channel, sc.sampaTime);
    if (sc.isClusterSum()) {
      result += fmt::format("-{}-cs-{}", sc.chargeSum, sc.clusterSize);
    } else {
      for (auto s : sc.samples) {
        result += fmt::format("-{}", s);
      }
    }
    result += "\n";
  };
}

SampaErrorHandler handleError(std::string& result)
{
  return [&result](DsElecId dsId, int8_t chip, uint32_t error) {
    result += fmt::format("{}-chip-{}-error-{}", asString(dsId), chip, error);
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

  // get back to byte buffer to return
  std::vector<std::byte> bytes;
  impl::copyBuffer(b64, bytes);
  return bytes;
}

template <typename CHARGESUM, int VERSION>
std::string decodeBuffer(int feeId, gsl::span<const std::byte> buffer)
{
  std::string results;
  auto fee2solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  DecodedDataHandlers handlers;
  handlers.sampaChannelHandler = handlePacket(results);
  handlers.sampaErrorHandler = handleError(results);
  UserLogicEndpointDecoder<CHARGESUM, VERSION> dec(feeId, fee2solar, handlers);
  dec.append(buffer);
  return results;
}

template <typename CHARGESUM, int VERSION>
std::string testPayloadDecode(DsElecId ds1,
                              DualSampaChannelId ch1,
                              const std::vector<SampaCluster>& clustersFirstChannel,
                              DsElecId ds2 = DsElecId{0, 0, 0},
                              DualSampaChannelId ch2 = 47,
                              const std::vector<SampaCluster>& clustersSecondChannel = {},
                              std::optional<size_t> insertSync = std::nullopt)
{

  auto solar2feelink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();

  auto encoder = createPayloadEncoder(solar2feelink, true, VERSION, isChargeSumMode<CHARGESUM>::value);

  encoder->startHeartbeatFrame(0, 0);

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

  return decodeBuffer<CHARGESUM, VERSION>(feeId, payloadBuffer);
}

template <int VERSION>
std::vector<uint64_t> convert(gsl::span<const uint64_t> page);

template <>
std::vector<uint64_t> convert<0>(gsl::span<const uint64_t> page)
{
  return {page.begin(), page.end()};
}

template <>
std::vector<uint64_t> convert<1>(gsl::span<const uint64_t> page)
{
  // convert the 14 MSB bits of page, expressed using V0 spec,
  // to match the V1 spec
  std::vector<uint64_t> pagev1{page.begin(), page.end()};
  constexpr int rdhSize{8};
  for (int i = rdhSize; i < pagev1.size(); i++) {
    if (pagev1[i] == 0xFEEDDEEDFEEDDEED || pagev1[i] == 0) {
      // do not mess with padding words
      continue;
    }
    ULHeaderWord<0> v0{pagev1[i]};
    ULHeaderWord<1> v1;
    v1.data = v0.data;
    v1.error = v0.error;
    v1.incomplete = v0.incomplete;
    v1.dsID = v0.dsID;
    v1.linkID = v0.linkID;
    pagev1[i] = v1.word;
  }
  return pagev1;
}

template <int VERSION = 0>
std::string testPayloadDecodeCruPages(gsl::span<const uint64_t> ipage)
{
  std::vector<uint64_t> page = convert<VERSION>(ipage);

  const void* rdhP = reinterpret_cast<const void*>(page.data());
  uint16_t feeId = o2::raw::RDHUtils::getFEEID(rdhP);
  auto rdhSize = o2::raw::RDHUtils::getHeaderSize(rdhP);
  auto payloadSize = o2::raw::RDHUtils::getMemorySize(rdhP) - rdhSize;

  gsl::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(page.data()), page.size() * 8);
  gsl::span<const std::byte> payloadBuffer = buffer.subspan(rdhSize, payloadSize);

  o2::mch::raw::FEEID f{feeId};

  if (f.chargeSum) {
    return decodeBuffer<ChargeSumMode, VERSION>(f.id, payloadBuffer);
  } else {
    return decodeBuffer<SampleMode, VERSION>(f.id, payloadBuffer);
  }
}

struct V0 {
  static constexpr int value = 0;
};
struct V1 {
  static constexpr int value = 1;
};

typedef boost::mpl::list<V0, V1> testTypes;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(userlogicdsdecoder)

BOOST_AUTO_TEST_CASE_TEMPLATE(SampleModeSimplest, V, testTypes)
{
  // only one channel with one very small cluster
  // fitting within one 64-bits word
  SampaCluster cl(345, 6789, {123, 456});
  auto r = testPayloadDecode<SampleMode, V::value>(DsElecId{728, 1, 0}, 63, {cl});
  BOOST_CHECK_EQUAL(r, "S728-J1-DS0-ch-63-ts-345-q-123-456\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SampleModeSimple, V, testTypes)
{
  // only one channel with one cluster, but the cluster
  // spans 2 64-bits words.
  SampaCluster cl(345, 6789, {123, 456, 789, 901, 902});
  auto r = testPayloadDecode<SampleMode, V::value>(DsElecId{448, 6, 4}, 63, {cl});
  BOOST_CHECK_EQUAL(r, "S448-J6-DS4-ch-63-ts-345-q-123-456-789-901-902\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SampleModeTwoChannels, V, testTypes)
{
  // 2 channels with one cluster
  SampaCluster cl(345, 6789, {123, 456, 789, 901, 902});
  SampaCluster cl2(346, 6789, {1001, 1002, 1003, 1004, 1005, 1006, 1007});
  auto r = testPayloadDecode<SampleMode, V::value>(DsElecId{361, 6, 2}, 63, {cl}, DsElecId{361, 6, 2}, 47, {cl2});
  BOOST_CHECK_EQUAL(r,
                    "S361-J6-DS2-ch-63-ts-345-q-123-456-789-901-902\n"
                    "S361-J6-DS2-ch-47-ts-346-q-1001-1002-1003-1004-1005-1006-1007\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ChargeSumModeSimplest, V, testTypes)
{
  // only one channel with one cluster
  // (hence fitting within one 64 bits word)
  SampaCluster cl(345, 6789, 123456, 789);
  auto r = testPayloadDecode<ChargeSumMode, V::value>(DsElecId{728, 1, 0}, 63, {cl});
  BOOST_CHECK_EQUAL(r, "S728-J1-DS0-ch-63-ts-345-q-123456-cs-789\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ChargeSumModeSimple, V, testTypes)
{
  // only one channel with 2 clusters
  // (hence spanning 2 64-bits words)
  SampaCluster cl1(345, 6789, 123456, 789);
  SampaCluster cl2(346, 6789, 789012, 345);
  auto r = testPayloadDecode<ChargeSumMode, V::value>(DsElecId{448, 6, 4}, 63, {cl1, cl2});
  BOOST_CHECK_EQUAL(r,
                    "S448-J6-DS4-ch-63-ts-345-q-123456-cs-789\n"
                    "S448-J6-DS4-ch-63-ts-346-q-789012-cs-345\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ChargeSumModeTwoChannels, V, testTypes)
{
  // two channels with 2 clusters
  SampaCluster cl1(345, 6789, 123456, 789);
  SampaCluster cl2(346, 6789, 789012, 345);
  SampaCluster cl3(347, 6789, 1357, 890);
  SampaCluster cl4(348, 6789, 7912, 345);
  auto r = testPayloadDecode<ChargeSumMode, V::value>(DsElecId{361, 6, 2}, 63, {cl1, cl2}, DsElecId{361, 6, 2}, 47, {cl3, cl4});
  BOOST_CHECK_EQUAL(r,
                    "S361-J6-DS2-ch-63-ts-345-q-123456-cs-789\n"
                    "S361-J6-DS2-ch-63-ts-346-q-789012-cs-345\n"
                    "S361-J6-DS2-ch-47-ts-347-q-1357-cs-890\n"
                    "S361-J6-DS2-ch-47-ts-348-q-7912-cs-345\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SyncInTheMiddleChargeSumModeTwoChannels, V, testTypes)
{
  // Insert a sync word in the middle of
  // the TwoChannels case and check the decoder is handling this fine
  // (by just returning to wait for sync mode, i.e. dropping the 2nd part
  // of the communication until a second sync)
  SampaCluster cl1(345, 6789, 123456, 789);
  SampaCluster cl2(346, 6789, 789012, 345);
  SampaCluster cl3(347, 6789, 1357, 890);
  SampaCluster cl4(348, 6789, 7912, 345);
  auto r = testPayloadDecode<ChargeSumMode, V::value>(
    DsElecId{361, 6, 2}, 63, {cl1, cl2},
    DsElecId{361, 6, 2}, 47, {cl3, cl4},
    5);
  BOOST_CHECK_EQUAL(r,
                    "S361-J6-DS2-ch-63-ts-345-q-123456-cs-789\n"
                    "S361-J6-DS2-ch-63-ts-346-q-789012-cs-345\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(TestCruPageOK, V, testTypes)
{
  gsl::span<const uint64_t> page = CruPageOK;
  std::string r = testPayloadDecodeCruPages<V::value>(page);
  BOOST_CHECK_EQUAL(r,
                    "S81-J0-DS0-ch-42-ts-87-q-2-1-0-4-4-3-3-0-0-1-0-0-0\n"
                    "S81-J0-DS0-ch-42-ts-0-q-1\n");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(TestCruPageBadClusterSize, V, testTypes)
{
  gsl::span<const uint64_t> page = CruPageBadClusterSize;
  std::string r = testPayloadDecodeCruPages<V::value>(page);
  BOOST_CHECK_EQUAL(r,
                    fmt::format("S81-J0-DS0-chip-1-error-{}\nS81-J0-DS0-ch-42-ts-0-q-1\n", ErrorBadClusterSize));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(TestCruPageBadN10bitWords, V, testTypes)
{
  gsl::span<const uint64_t> page = CruPageBadN10bitWords;
  std::string r = testPayloadDecodeCruPages<V::value>(page);
  std::string expected =
    fmt::format("S81-J0-DS0-ch-42-ts-87-q-2-1-0-0-1-0-0-0\nS81-J0-DS0-chip-1-error-{}\nS81-J0-DS0-ch-42-ts-0-q-1\n",
                ErrorBadIncompleteWord);
  BOOST_CHECK_EQUAL(r, expected);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
