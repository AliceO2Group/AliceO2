// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw Closure
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include <fmt/format.h>
#include <iostream>
#include <boost/mpl/list.hpp>
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::mch::raw;

const char* sampaClusterFormat = "{}-CH{}-{}";

// Create a vector of SampaCluster from a string d
// where d is of the form ts-#-bc-#-cs-#-q-# or
// ts-#-bc-#-cs-#-q-#-#-# ...
// d is expected to be a valid SampaCluster string representation
// see the asString function in SampaCluster
std::vector<SampaCluster> getSampaClusters(const std::string& d)
{
  std::vector<SampaCluster> clusters;

  std::cout << "d: " << d << std::endl;

  auto index = d.find("ts-");
  auto ts = std::stoi(d.substr(index + 3));
  index = d.find("bc-");
  auto bc = std::stoi(d.substr(index + 3));
  index = d.find("cs-");
  auto cs = std::stoi(d.substr(index + 3));
  index = d.find("q-");
  auto q = d.substr(index + 2);
  index = q.find("-");
  if (index != std::string::npos) {
    std::vector<uint10_t> charges;
    std::istringstream ss(q);
    std::string adc;
    while (std::getline(ss, adc, '-')) {
      charges.emplace_back(std::stoi(adc) & 0x3FF);
    }
    clusters.emplace_back(SampaCluster(ts, bc, charges));
  } else {
    clusters.emplace_back(SampaCluster(ts, bc, std::stoi(q), cs));
  }
  return clusters;
}

// create a raw data buffer from a list of strings
// where each string is of the form
// S#-J#-DS#-CH#-ts-#-bc-#-q-#
//
// it's a two steps process :
//
// - first create a buffer of payloads using a PayloadEncoder
// - then create the raw data itself (with proper RDHs) from the payload buffer
//
template <typename FORMAT, typename CHARGESUM>
std::vector<std::byte> createBuffer(gsl::span<std::string> data,
                                    uint32_t orbit = 12345, uint16_t bc = 678)
{
  auto encoder = createPayloadEncoder<FORMAT, CHARGESUM, true>();
  encoder->startHeartbeatFrame(orbit, bc);
  for (auto d : data) {
    auto dsElecId = decodeDsElecId(d);
    if (!dsElecId) {
      std::cout << "Could not get dsElecId for " << d << "\n";
    }
    auto channel = decodeChannelId(d);
    if (!channel) {
      std::cout << "Could not get channel for " << d << "\n";
    }
    auto sampaClusters = getSampaClusters(d);
    encoder->addChannelData(dsElecId.value(), channel.value(), sampaClusters);
  }
  std::vector<std::byte> buffer;
  encoder->moveToBuffer(buffer);

  const o2::raw::HBFUtils& hbfutils = o2::raw::HBFUtils::Instance();
  o2::conf::ConfigurableParam::setValue<uint32_t>("HBFUtils", "orbitFirst", orbit);
  std::vector<std::byte> out = o2::mch::raw::paginate(buffer,
                                                      isUserLogicFormat<FORMAT>::value,
                                                      isChargeSumMode<CHARGESUM>::value,
                                                      fmt::format("mch-closure-codec-{}-{}.raw",
                                                                  orbit, bc));
  return out;
}

// method that is called by the decoder each time a SampaCluster is decoded.
SampaChannelHandler handlePacketStoreAsVec(std::vector<std::string>& result)
{
  return [&result](DsElecId dsId, uint8_t channel, SampaCluster sc) {
    result.emplace_back(fmt::format(sampaClusterFormat, asString(dsId), channel, asString(sc)));
  };
}

// decode the buffer and check its content against the expected vector of strings
bool testDecode(gsl::span<const std::byte> testBuffer, gsl::span<std::string> expected)
{
  std::vector<std::string> result;

  DecodedDataHandlers handlers;
  handlers.sampaChannelHandler = handlePacketStoreAsVec(result);
  auto pageDecoder = createPageDecoder(testBuffer, handlers);

  auto parser = createPageParser();

  parser(testBuffer, pageDecoder);

  bool sameSize = result.size() == expected.size();
  bool permutation = std::is_permutation(begin(result), end(result), begin(expected));
  BOOST_CHECK_EQUAL(sameSize, true);
  BOOST_CHECK_EQUAL(permutation, true);
  if (!permutation || !sameSize) {
    std::cout << "Got " << result.size() << " results:\n";
    for (auto s : result) {
      std::cout << s << "\n";
    }
    std::cout << "Expected " << expected.size() << ":\n";
    for (auto s : expected) {
      std::cout << s << "\n";
    }
    return false;
  }
  return true;
}

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(closure)

std::vector<std::string> chargeSumInput = {
  "S728-J1-DS0-CH3-ts-24-bc-0-cs-14-q-13",
  "S728-J1-DS0-CH13-ts-24-bc-0-cs-134-q-133",
  "S728-J1-DS0-CH23-ts-24-bc-0-cs-164-q-163",

  "S361-J0-DS4-CH0-ts-24-bc-0-cs-11-q-10",
  "S361-J0-DS4-CH1-ts-24-bc-0-cs-21-q-20",
  "S361-J0-DS4-CH2-ts-24-bc-0-cs-31-q-30",
  "S361-J0-DS4-CH3-ts-24-bc-0-cs-41-q-40",

  "S448-J6-DS2-CH22-ts-24-bc-0-cs-421-q-420",
  "S448-J6-DS2-CH23-ts-24-bc-0-cs-431-q-430",
  "S448-J6-DS2-CH24-ts-24-bc-0-cs-441-q-440",
  "S448-J6-DS2-CH25-ts-24-bc-0-cs-451-q-450",
  "S448-J6-DS2-CH26-ts-24-bc-0-cs-461-q-460",
  "S448-J6-DS2-CH42-ts-24-bc-0-cs-421-q-420"};

typedef boost::mpl::list<BareFormat, UserLogicFormat> testTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(ClosureChargeSum, FORMAT, testTypes)
{
  auto buffer = createBuffer<FORMAT, ChargeSumMode>(chargeSumInput);
  testDecode(buffer, chargeSumInput);
}

std::vector<std::string> sampleInput = {
  "S728-J1-DS0-CH3-ts-24-bc-0-cs-3-q-13-15-17",
  "S728-J1-DS0-CH13-ts-24-bc-0-cs-3-q-133-135-137",
  "S728-J1-DS0-CH23-ts-24-bc-0-cs-2-q-163-165",

  "S361-J0-DS4-CH0-ts-24-bc-0-cs-3-q-10-12-14",
  "S361-J0-DS4-CH1-ts-24-bc-0-cs-3-q-20-22-24",
  "S361-J0-DS4-CH2-ts-24-bc-0-cs-3-q-30-32-34",
  "S361-J0-DS4-CH3-ts-24-bc-0-cs-3-q-40-42-44",

  "S448-J6-DS2-CH22-ts-24-bc-0-cs-3-q-420-422-424",
  "S448-J6-DS2-CH23-ts-24-bc-0-cs-3-q-430-432-434",
  "S448-J6-DS2-CH24-ts-24-bc-0-cs-3-q-440-442-444",
  "S448-J6-DS2-CH25-ts-24-bc-0-cs-3-q-450-452-454",
  "S448-J6-DS2-CH26-ts-24-bc-0-cs-3-q-460-462-464",
  "S448-J6-DS2-CH42-ts-24-bc-0-cs-5-q-420-422-424-426-428"};

BOOST_AUTO_TEST_CASE_TEMPLATE(ClosureSample, FORMAT, testTypes)
{
  auto buffer = createBuffer<FORMAT, SampleMode>(sampleInput);
  testDecode(buffer, sampleInput);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
