// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw ClosureDigit
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <fstream>
#include "MCHRawDecoder/DataDecoder.h"
#include "MCHRawEncoderDigit/DigitRawEncoder.h"
#include "DataFormatsMCH/Digit.h"
#include "Framework/Logger.h"
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <array>
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawCommon/CoDecParam.h"

using namespace o2::mch::raw;

constexpr const char* sampaClusterFormat = "{}-CH{}-{}";
const bool useDummyElecMap = true;

struct DePadId {
  int deid;
  int padid;
  bool operator==(const DePadId& other) const
  {
    return deid == other.deid && padid == other.padid;
  }
  bool operator<(const DePadId& other) const
  {
    if (deid == other.padid) {
      return padid < other.padid;
    }
    return deid < other.padid;
  }
};

std::ostream& operator<<(std::ostream& os, const DePadId& dpi)
{
  os << fmt::format("DE {:4d} PADID {:4d}", dpi.deid, dpi.padid);
  return os;
}

template <typename T>
SampaChannelHandler handlePacketStoreAsVec(std::vector<T>& result);

// method that is called by the decoder each time a SampaCluster is decoded.
template <>
SampaChannelHandler handlePacketStoreAsVec(std::vector<std::string>& result)
{
  return [&result](DsElecId dsId, DualSampaChannelId channel, SampaCluster sc) {
    result.emplace_back(fmt::format(sampaClusterFormat, asString(dsId), channel, asString(sc)));
  };
}

template <>
SampaChannelHandler handlePacketStoreAsVec(std::vector<DePadId>& result)
{
  auto elec2det = useDummyElecMap ? o2::mch::raw::createElec2DetMapper<ElectronicMapperDummy>() : o2::mch::raw::createElec2DetMapper<ElectronicMapperGenerated>();
  return [&result, elec2det](DsElecId dsId, DualSampaChannelId channel, SampaCluster sc) {
    auto dsDet = elec2det(dsId);
    auto deId = dsDet->deId();
    auto seg = o2::mch::mapping::segmentation(deId);
    auto padId = seg.findPadByFEE(dsDet->dsId(), channel);
    result.emplace_back(DePadId{dsDet->deId(), padId});
  };
}

void writeDigits()
{
  std::cout << fmt::format("BEGIN writeDigits({})\n", useDummyElecMap);
  fair::Logger::SetConsoleSeverity("nolog");
  {
    std::vector<o2::mch::Digit> digits;
    digits.emplace_back(923, 3959, 959, 123, 1);
    digits.emplace_back(923, 3974, 974, 123, 1);
    digits.emplace_back(100, 6664, 664, 123, 1);

    DigitRawEncoderOptions opts;
    opts.splitMode = OutputSplit::None; // to get only one file
    opts.noGRP = true;                  // as we don't have a GRP at hand
    opts.noEmptyHBF = true;             // as we don't want to create big files
    opts.writeHB = false;               // as we'd like to keep it simple
    opts.userLogicVersion = 1;          // test only the latest version
    opts.dummyElecMap = useDummyElecMap;

    DigitRawEncoder dre(opts);

    uint32_t orbit{0};
    uint16_t bc{3456};

    fair::Logger::SetConsoleSeverity("info");
    dre.encodeDigits(digits, orbit, bc);
    //dre.addHeartbeats(std::set<DsElecId> dsElecIds, uint32_t orbit);
  }
  std::cout << fmt::format("END writeDigits({})\n", useDummyElecMap);
}

std::vector<std::byte> getBuffer(const char* filename)
{
  std::vector<std::byte> buffer;
  std::ifstream is(filename, std::ifstream::binary);

  is.seekg(0, is.end);
  int length = is.tellg();
  is.seekg(0, is.beg);

  buffer.resize(length);

  is.read(reinterpret_cast<char*>(&buffer[0]), length);
  is.close();
  return buffer;
}

template <typename T>
std::vector<T> readDigits()
{
  std::vector<T> result;
  DataDecoder dd(handlePacketStoreAsVec<T>(result), nullptr, "", "", false, false, useDummyElecMap);

  auto buffer = getBuffer("MCH.raw");
  dd.decodeBuffer(buffer);
  return result;
}

BOOST_AUTO_TEST_CASE(WrittenAndReadBackDigitsShouldBeTheSameStringVersion)
{
  std::vector<std::string> expected = {
    "S481-J5-DS1-CH58-ts-0-bc-3456-cs-1-q-959",
    "S481-J5-DS1-CH11-ts-0-bc-3456-cs-1-q-974",
    "S394-J5-DS3-CH35-ts-0-bc-3456-cs-1-q-664"};
  if (useDummyElecMap) {
    expected = std::vector<std::string>{
      "S727-J6-DS4-CH58-ts-0-bc-3456-cs-1-q-959",
      "S727-J6-DS4-CH11-ts-0-bc-3456-cs-1-q-974",
      "S363-J4-DS4-CH35-ts-0-bc-3456-cs-1-q-664"};
  }
  writeDigits();
  auto result = readDigits<std::string>();

  bool sameSize = result.size() == expected.size();
  BOOST_CHECK_EQUAL(sameSize, true);
  bool permutation = false;
  if (sameSize) {
    permutation = std::is_permutation(begin(result), end(result), begin(expected));
    BOOST_CHECK_EQUAL(permutation, true);
  }
  if (!permutation || !sameSize) {
    std::cout << "Got " << result.size() << " results:\n";
    for (auto s : result) {
      std::cout << s << "\n";
    }
    std::cout << "Expected " << expected.size() << ":\n";
    for (auto s : expected) {
      std::cout << s << "\n";
    }
  }
}

BOOST_AUTO_TEST_CASE(WrittenAndReadBackDigitsShouldBeTheSame)
{
  std::vector<DePadId> expected = {
    DePadId{923, 3959},
    DePadId{923, 3974},
    DePadId{100, 6664}};
  writeDigits();
  auto result = readDigits<DePadId>();

  bool sameSize = result.size() == expected.size();
  BOOST_CHECK_EQUAL(sameSize, true);
  bool permutation = false;
  if (sameSize) {
    permutation = std::is_permutation(begin(result), end(result), begin(expected));
    BOOST_CHECK_EQUAL(permutation, true);
  }
  if (!permutation || !sameSize) {
    std::cout << "Got " << result.size() << " results:\n";
    for (auto s : result) {
      std::cout << s << "\n";
    }
    std::cout << "Expected " << expected.size() << ":\n";
    for (auto s : expected) {
      std::cout << s << "\n";
    }
  }
}
