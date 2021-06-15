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
///
/// In those tests we are mainly concerned about testinng
/// whether the payloads are actually properly simulated.
///
#include "MCHRawElecMap/Mapper.h"
#define BOOST_TEST_MODULE Test MCHRaw Encoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "BareElinkEncoder.h"
#include "BareElinkEncoderMerger.h"
#include "EncoderImplHelper.h"
#include "GBTEncoder.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "UserLogicElinkEncoder.h"
#include "UserLogicElinkEncoderMerger.h"
#include <boost/mpl/list.hpp>
#include <cstdint>
#include <fmt/printf.h>
#include <vector>

using namespace o2::mch::raw;

template <typename ELECMAP, typename FORMAT, int VERSION = 0>
std::vector<std::byte> makeBuffer(int norbit = 1,
                                  uint32_t firstOrbit = 12345,
                                  uint16_t firstBC = 678,
                                  bool withHB = false)
{
  const DsElecId ds1{728, 1, 0};
  const DsElecId ds2{361, 0, 4};
  const DsElecId ds3{448, 6, 2};
  const DsElecId ds4{728, 1, 2};

  std::set<DsElecId> dsElecIds;
  if (withHB) {
    dsElecIds.insert(ds1);
    dsElecIds.insert(ds2);
    dsElecIds.insert(ds3);
    if (norbit > 1) {
      dsElecIds.insert(ds4);
    }
  }
  auto encoder = createPayloadEncoder(createSolar2FeeLinkMapper<ELECMAP>(),
                                      isUserLogicFormat<FORMAT>::value,
                                      VERSION,
                                      isChargeSumMode<ChargeSumMode>::value);

  uint16_t sampaTime{24};
  uint32_t bunchCrossing = 567;
  uint16_t bc(firstBC);

  encoder->startHeartbeatFrame(firstOrbit, bc);
  if (withHB) {
    encoder->addHeartbeatHeaders(dsElecIds);
  }

  encoder->addChannelData(ds1, 3, {SampaCluster(sampaTime, bunchCrossing, 13, 14)});
  encoder->addChannelData(ds1, 13, {SampaCluster(sampaTime, bunchCrossing, 133, 134)});
  encoder->addChannelData(ds1, 23, {SampaCluster(sampaTime, bunchCrossing, 163, 164)});

  encoder->addChannelData(ds2, 0, {SampaCluster(sampaTime, bunchCrossing, 10, 11)});
  encoder->addChannelData(ds2, 1, {SampaCluster(sampaTime, bunchCrossing, 20, 21)});
  encoder->addChannelData(ds2, 2, {SampaCluster(sampaTime, bunchCrossing, 30, 31)});
  encoder->addChannelData(ds2, 3, {SampaCluster(sampaTime, bunchCrossing, 40, 41)});

  encoder->addChannelData(ds3, 22, {SampaCluster(sampaTime, bunchCrossing, 420, 421)});
  encoder->addChannelData(ds3, 23, {SampaCluster(sampaTime, bunchCrossing, 430, 431)});
  encoder->addChannelData(ds3, 24, {SampaCluster(sampaTime, bunchCrossing, 440, 441)});
  encoder->addChannelData(ds3, 25, {SampaCluster(sampaTime, bunchCrossing, 450, 451)});
  encoder->addChannelData(ds3, 26, {SampaCluster(sampaTime, bunchCrossing, 460, 461)});
  encoder->addChannelData(ds3, 42, {SampaCluster(sampaTime, bunchCrossing, 420, 421)});

  if (norbit > 1) {
    encoder->startHeartbeatFrame(firstOrbit + 1, bc);
    encoder->addChannelData(ds4, 0, {SampaCluster(sampaTime, bunchCrossing, 10, 11)});
    encoder->addChannelData(ds4, 1, {SampaCluster(sampaTime, bunchCrossing, 10, 11)});
    encoder->addChannelData(ds2, 0, {SampaCluster(sampaTime, bunchCrossing, 10, 11)});
    encoder->addChannelData(ds2, 1, {SampaCluster(sampaTime, bunchCrossing, 20, 21)});
    encoder->addChannelData(ds2, 2, {SampaCluster(sampaTime, bunchCrossing, 30, 31)});
    encoder->addChannelData(ds2, 3, {SampaCluster(sampaTime, bunchCrossing, 40, 41)});
  }

  if (norbit > 2) {
    encoder->startHeartbeatFrame(firstOrbit + 2, bc);
    encoder->addChannelData(ds3, 12, {SampaCluster(sampaTime, bunchCrossing, 420, 421)});
  }

  std::vector<std::byte> buffer;
  encoder->moveToBuffer(buffer);

  return buffer;
}

template <typename ELECMAP, typename FORMAT>
std::unique_ptr<PayloadEncoder> defaultEncoder()
{
  return createPayloadEncoder(createSolar2FeeLinkMapper<ELECMAP>(),
                              isUserLogicFormat<FORMAT>::value,
                              0,
                              isChargeSumMode<SampleMode>::value /* i.e. use sample mode */);
}

struct BareGen {
  using format = BareFormat;
  using elecmap = ElectronicMapperGenerated;
  static constexpr int version = 0;
};

struct UserLogicGen {
  using format = UserLogicFormat;
  using elecmap = ElectronicMapperGenerated;
  static constexpr int version = 0;
};

struct UserLogicGen1 {
  using format = UserLogicFormat;
  using elecmap = ElectronicMapperGenerated;
  static constexpr int version = 1;
};

struct BareDummy {
  using format = BareFormat;
  using elecmap = ElectronicMapperDummy;
  static constexpr int version = 0;
};

struct UserLogicDummy {
  using format = UserLogicFormat;
  using elecmap = ElectronicMapperDummy;
  static constexpr int version = 0;
};

struct UserLogicDummy1 {
  using format = UserLogicFormat;
  using elecmap = ElectronicMapperDummy;
  static constexpr int version = 1;
};

typedef boost::mpl::list<BareGen, UserLogicGen, UserLogicGen1, BareDummy, UserLogicDummy, UserLogicDummy1> testTypes;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(encoder)

BOOST_AUTO_TEST_CASE_TEMPLATE(StartHBFrameBunchCrossingMustBe12Bits, T, testTypes)
{
  auto encoder = defaultEncoder<typename T::elecmap, typename T::format>();
  BOOST_CHECK_THROW(encoder->startHeartbeatFrame(0, 1 << 12), std::invalid_argument);
  BOOST_CHECK_NO_THROW(encoder->startHeartbeatFrame(0, 0xFFF));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EmptyEncoderHasEmptyBufferIfPhaseIsZero, T, testTypes)
{
  srand(time(nullptr));
  auto encoder = defaultEncoder<typename T::elecmap, typename T::format>();
  encoder->startHeartbeatFrame(12345, 123);
  std::vector<std::byte> buffer;
  encoder->moveToBuffer(buffer);
  BOOST_CHECK_EQUAL(buffer.size(), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MultipleOrbitsWithNoDataIsAnEmptyBufferIfPhaseIsZero, T, testTypes)
{
  srand(time(nullptr));
  auto encoder = defaultEncoder<typename T::elecmap, typename T::format>();
  encoder->startHeartbeatFrame(12345, 123);
  encoder->startHeartbeatFrame(12345, 125);
  encoder->startHeartbeatFrame(12345, 312);
  std::vector<std::byte> buffer;
  encoder->moveToBuffer(buffer);
  BOOST_CHECK_EQUAL(buffer.size(), 0);
}

int estimateUserLogicSize(int nofDS, int maxNofChPerDS)
{
  // counting first in number of 10 bits

  size_t sync = 5;
  size_t sampaHeaders = nofDS * maxNofChPerDS * 5;
  size_t sampaData = nofDS * maxNofChPerDS * 4;

  size_t n10 = sync + sampaHeaders + sampaData;
  while (n10 % 5) {
    n10++;
  }

  return (n10 / 5) * (64 / 8) + sizeof(DataBlockHeader);
}

int estimateBareSize(int nofDS, int maxNofChPerGBT)
{
  size_t headerSize = 2; // equivalent to 2 64-bits words
  size_t nbits = nofDS * 50 + maxNofChPerGBT * 90;
  size_t n128bitwords = nbits / 2;
  size_t n64bitwords = n128bitwords * 2;
  return 8 * (n64bitwords + headerSize); // size in bytes
}

template <typename FORMAT>
int estimateSize(bool withHB = false);

template <>
int estimateSize<BareFormat>(bool withHB)
{
  int size = estimateBareSize(1, 3) +
             estimateBareSize(1, 4) +
             estimateBareSize(1, 6);
  if (withHB) {
    size += 3 * 50 * 8;
  }
  return size; // in bytes
}

template <>
int estimateSize<UserLogicFormat>(bool withHB)
{
  int size = estimateUserLogicSize(1, 3) +
             estimateUserLogicSize(1, 4) +
             estimateUserLogicSize(1, 6) +
             2 * 8; // 8 bytes per FEE (to ensure the payload size is a multiple of 128 bits = 1 GBT word)
  if (withHB) {
    size += 6 * 8;
  }
  return size; // in bytes
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckNumberOfPayloadHeaders, T, testTypes)
{
  auto buffer = makeBuffer<typename T::elecmap, typename T::format, T::version>();
  int nheaders = o2::mch::raw::countHeaders(buffer);
  BOOST_CHECK_EQUAL(nheaders, 3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckSize, T, testTypes)
{
  auto buffer = makeBuffer<typename T::elecmap, typename T::format, T::version>();
  size_t expectedSize = estimateSize<typename T::format>();
  BOOST_CHECK_EQUAL(buffer.size(), expectedSize);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckSizeWithHB, T, testTypes)
{
  auto buffer = makeBuffer<typename T::elecmap, typename T::format, T::version>(1, 12345, 678, true);
  size_t expectedSize = estimateSize<typename T::format>(true);
  BOOST_CHECK_EQUAL(buffer.size(), expectedSize);
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
  impl::append(b10, sampaSyncWord);
  std::vector<uint64_t> b64;
  impl::b10to64(b10, b64, 24);
  auto w = b64[0];
  BOOST_CHECK_EQUAL(asBinary(w), "0000 0000 0110 0001 0101 0101 0101 0101 0100 0000 1111 0000 0000 0001 0001 0011");
}

BOOST_AUTO_TEST_CASE(Binary)
{
  dump(static_cast<uint64_t>(7) << 50, {50, 51, 52, 0});
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
