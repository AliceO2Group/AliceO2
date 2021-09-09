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

/// \file   MID/Filtering/test/testFilter.cxx
/// \brief  Test Filtering device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 March 2018

#define BOOST_TEST_MODULE midFiltering
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include <cstdint>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include "DataFormatsMID/ColumnData.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDFiltering/ChannelScalers.h"
#include "MIDFiltering/FetToDead.h"
#include "MIDFiltering/MaskMaker.h"

namespace o2
{
namespace mid
{

std::vector<ColumnData> generateData(size_t nData = 10)
{

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<uint8_t> deIds(0, 71);
  std::uniform_int_distribution<uint8_t> colIds(0, 6);
  std::uniform_int_distribution<uint16_t> patterns(0, 0xFFFF);

  std::vector<o2::mid::ColumnData> data;
  for (size_t idata = 0; idata < nData; ++idata) {
    ColumnData col;
    col.deId = deIds(mt);
    col.columnId = colIds(mt);
    for (int iline = 0; iline < 4; ++iline) {
      col.setBendPattern(patterns(mt), iline);
    }
    col.setNonBendPattern(patterns(mt));
    data.emplace_back(col);
  }
  return data;
}

BOOST_AUTO_TEST_CASE(mask)
{
  o2::mid::ColumnData col1;
  col1.deId = 71;
  col1.columnId = 6;
  col1.setNonBendPattern(0x8000);

  ChannelMasksHandler masksHandler;
  masksHandler.switchOffChannels(col1);
  auto maskVec = masksHandler.getMasks();
  BOOST_TEST(maskVec.size() == 1);
  for (auto mask : maskVec) {
    for (int iline = 0; iline < 4; ++iline) {
      BOOST_TEST(mask.getBendPattern(iline) == static_cast<uint16_t>(~col1.getBendPattern(iline)));
    }
    BOOST_TEST(mask.getNonBendPattern() == static_cast<uint16_t>(~col1.getNonBendPattern()));
  }
}

BOOST_AUTO_TEST_CASE(scalers)
{

  o2::mid::ColumnData col1;
  col1.deId = 71;
  col1.columnId = 6;
  col1.setNonBendPattern(0x8000);

  o2::mid::ChannelScalers cs;
  cs.count(col1);
  auto sc1 = cs.getScalers();
  BOOST_REQUIRE(sc1.size() == 1);
  for (auto& sc : sc1) {
    BOOST_TEST(cs.getDeId(sc.first) = col1.deId);
    BOOST_TEST(cs.getColumnId(sc.first) = col1.columnId);
    BOOST_TEST(cs.getLineId(sc.first) == 0);
    BOOST_TEST(cs.getCathode(sc.first) == 1);
    BOOST_TEST(cs.getStrip(sc.first) == 15);
  }
  cs.reset();

  o2::mid::ColumnData col2;
  col2.deId = 25;
  col2.columnId = 3;
  col2.setBendPattern(0x0100, 3);
  cs.count(col2);
  auto sc2 = cs.getScalers();
  BOOST_REQUIRE(sc2.size() == 1);
  for (auto& sc : sc2) {
    BOOST_TEST(cs.getDeId(sc.first) = col2.deId);
    BOOST_TEST(cs.getColumnId(sc.first) = col2.columnId);
    BOOST_TEST(cs.getLineId(sc.first) == 3);
    BOOST_TEST(cs.getCathode(sc.first) == 0);
    BOOST_TEST(cs.getStrip(sc.first) == 8);
  }
}

BOOST_AUTO_TEST_CASE(maskMaker)
{
  auto data = generateData();
  o2::mid::ChannelScalers cs;
  std::vector<o2::mid::ColumnData> refMasks{};
  for (auto col : data) {
    cs.reset();
    cs.count(col);
    auto masks = o2::mid::makeMasks(cs, 1, 0., refMasks);
    BOOST_TEST(masks.size() == 1);
    for (auto& mask : masks) {
      for (int iline = 0; iline < 4; ++iline) {
        BOOST_TEST(mask.getBendPattern(iline) == static_cast<uint16_t>(~col.getBendPattern(iline)));
      }
      BOOST_TEST(mask.getNonBendPattern() == static_cast<uint16_t>(~col.getNonBendPattern()));
    }
  }
}

BOOST_AUTO_TEST_CASE(FETConversion)
{
  /// Tests the conversion of Fet data to dead channels

  // Use the masks as FET: all channels should answer
  auto fets = makeDefaultMasks();

  // We now modify one FET data
  uint16_t fullPattern = 0xFFFF;
  for (auto& col : fets) {
    if (col.deId == 3 && col.columnId == 0) {
      col.setBendPattern(fullPattern & ~0x1, 0);
      col.setBendPattern(fullPattern, 1);
      col.setBendPattern(fullPattern, 2);
      // The following does not exist in deId 3, col 0
      col.setBendPattern(fullPattern & ~0x1, 3);
      col.setNonBendPattern(fullPattern);
    }
  }

  FetToDead fetToDead;
  auto inverted = fetToDead.process(fets);

  BOOST_TEST(inverted.size() == 1);
  BOOST_TEST(inverted.back().deId == 3);
  BOOST_TEST(inverted.back().columnId == 0);
  BOOST_TEST(inverted.back().getBendPattern(0) == 1);
  BOOST_TEST(inverted.back().getBendPattern(1) == 0);
  BOOST_TEST(inverted.back().getBendPattern(2) == 0);
  // Test that the non-existing pattern was not converted
  // Since the mask in the FET conversion takes care of removing it
  BOOST_TEST(inverted.back().getBendPattern(3) == 0);
  BOOST_TEST(inverted.back().getNonBendPattern() == 0);
}

} // namespace mid
} // namespace o2
