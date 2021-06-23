// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawElecMap/Mapper.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#define BOOST_TEST_MODULE Test MCHRaw Closure
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "Framework/Logger.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include <boost/mpl/list.hpp>
#include <fmt/format.h>
#include <iostream>

using namespace o2::mch::raw;

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

template <typename ELECMAP, typename FORMAT, typename CHARGESUM, int VERSION>
std::vector<std::byte> paginate(gsl::span<const std::byte> buffer, const std::string& tmpbasename)
{
  fair::Logger::SetConsoleSeverity("nolog");
  o2::raw::RawFileWriter fw;

  fw.setVerbosity(1);
  fw.setDontFillEmptyHBF(true);

  auto solar2LinkInfo = createSolar2LinkInfo<ELECMAP, FORMAT, CHARGESUM, VERSION>();

  // only use the solarIds that are actually in this test buffer
  // (to speed up the test)
  std::set<LinkInfo> links;
  for (auto solarId : {361, 448, 728}) {
    links.insert(solar2LinkInfo(solarId).value());
  }

  registerLinks(fw, tmpbasename, links, false);

  paginate(fw, buffer, links, solar2LinkInfo);

  fw.close();

  auto filename = fmt::format("{:s}.raw", tmpbasename);
  std::ifstream in(filename, std::ifstream::binary);
  if (in.fail()) {
    throw std::runtime_error(fmt::format("could not open ", filename));
  }
  // get length of file:
  in.seekg(0, in.end);
  int length = in.tellg();
  in.seekg(0, in.beg);
  std::vector<std::byte> pages(length);

  // read data as a block:
  in.read(reinterpret_cast<char*>(&pages[0]), length);

  return pages;
}

const char* sampaClusterFormat = "{}-CH{}-{}";

// Create a vector of SampaCluster from a string d
// where d is of the form ts-#-bc-#-cs-#-q-# or
// ts-#-bc-#-cs-#-q-#-#-# ...
// d is expected to be a valid SampaCluster string representation
// see the asString function in SampaCluster
std::vector<SampaCluster> getSampaClusters(const std::string& d)
{
  std::vector<SampaCluster> clusters;

  //std::cout << "d: " << d << std::endl;

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

std::set<DsElecId> getDs(gsl::span<std::string> data)
{
  std::set<DsElecId> dsids;
  for (auto d : data) {
    auto dsElecId = decodeDsElecId(d);
    if (!dsElecId) {
      std::cout << "Could not get dsElecId for " << d << "\n";
      continue;
    }
    dsids.insert(dsElecId.value());
  }
  return dsids;
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
template <typename ELECMAP, typename FORMAT, typename CHARGESUM, int VERSION>
std::vector<std::byte> createBuffer(gsl::span<std::string> data,
                                    uint32_t orbit = 12345, uint16_t bc = 678)
{
  const o2::raw::HBFUtils& hbfutils = o2::raw::HBFUtils::Instance();
  o2::conf::ConfigurableParam::setValue<uint32_t>("HBFUtils", "orbitFirst", orbit);

  auto encoder = createPayloadEncoder(createSolar2FeeLinkMapper<ELECMAP>(),
                                      isUserLogicFormat<FORMAT>::value,
                                      VERSION,
                                      isChargeSumMode<CHARGESUM>::value);
  encoder->startHeartbeatFrame(orbit, bc);
  std::set<DsElecId> dsElecIds = getDs(data);
  encoder->addHeartbeatHeaders(dsElecIds);
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

  o2::conf::ConfigurableParam::setValue<uint32_t>("HBFUtils", "orbitFirst", orbit);
  o2::conf::ConfigurableParam::setValue<uint32_t>("HBFUtils", "orbitFirstSampled", orbit);

  std::vector<std::byte> out =
    paginate<ELECMAP, FORMAT, CHARGESUM, VERSION>(buffer,
                                                  fmt::format("mch-closure-codec-{}-{}.raw",
                                                              orbit, bc));
  return out;
}

// method that is called by the decoder each time a SampaCluster is decoded.
SampaChannelHandler handlePacketStoreAsVec(std::vector<std::string>& result)
{
  return [&result](DsElecId dsId, DualSampaChannelId channel, SampaCluster sc) {
    result.emplace_back(fmt::format(sampaClusterFormat, asString(dsId), channel, asString(sc)));
  };
}

// decode the buffer and check its content against the expected vector of strings
template <typename ELECMAP>
bool testDecode(gsl::span<const std::byte> testBuffer, gsl::span<std::string> expected)
{
  std::vector<std::string> result;

  DecodedDataHandlers handlers;
  handlers.sampaChannelHandler = handlePacketStoreAsVec(result);
  auto pageDecoder = createPageDecoder(testBuffer, handlers,
                                       createFeeLink2SolarMapper<ELECMAP>());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ClosureChargeSum, T, testTypes)
{
  auto buffer = createBuffer<typename T::elecmap,
                             typename T::format,
                             ChargeSumMode,
                             T::version>(chargeSumInput);
  testDecode<typename T::elecmap>(buffer, chargeSumInput);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ClosureSample, T, testTypes)
{
  auto buffer = createBuffer<typename T::elecmap,
                             typename T::format,
                             SampleMode,
                             T::version>(sampleInput);
  testDecode<typename T::elecmap>(buffer, sampleInput);
}
