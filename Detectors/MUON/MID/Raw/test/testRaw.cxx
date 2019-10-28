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
/// @author  Diego Stocco

#define BOOST_TEST_MODULE Test MID raw
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include <iostream>
#include <vector>
#include <map>
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/Encoder.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/RawUnit.h"

BOOST_AUTO_TEST_SUITE(o2_mid_raw)

o2::mid::ColumnData getColData(uint8_t deId, uint8_t columnId, uint16_t nbp = 0, uint16_t bp1 = 0, uint16_t bp2 = 0, uint16_t bp3 = 0, uint16_t bp4 = 0)
{
  o2::mid::ColumnData col;
  col.deId = deId;
  col.columnId = columnId;
  col.setNonBendPattern(nbp);
  col.setBendPattern(bp1, 0);
  col.setBendPattern(bp2, 1);
  col.setBendPattern(bp3, 2);
  col.setBendPattern(bp4, 3);
  return col;
}

void doTest(const o2::mid::EventType& inEventType, const std::map<uint16_t, std::vector<o2::mid::ColumnData>>& inData, const std::vector<o2::mid::ROFRecord>& rofRecords, const std::vector<o2::mid::ColumnData>& data)
{
  BOOST_TEST(rofRecords.size() == inData.size());
  auto inItMap = inData.begin();
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    BOOST_TEST(static_cast<int>(rofIt->eventType) == static_cast<int>(inEventType));
    BOOST_TEST(rofIt->interactionRecord.bc == inItMap->first);
    BOOST_TEST(rofIt->nEntries == inItMap->second.size());
    for (size_t icol = rofIt->firstEntry; icol < rofIt->firstEntry + rofIt->nEntries; ++icol) {
      size_t inCol = icol - rofIt->firstEntry;
      BOOST_TEST(data[icol].deId == inItMap->second[inCol].deId);
      BOOST_TEST(data[icol].columnId == inItMap->second[inCol].columnId);
      BOOST_TEST(data[icol].getNonBendPattern() == inItMap->second[inCol].getNonBendPattern());
      for (int iline = 0; iline < 4; ++iline) {
        BOOST_TEST(data[icol].getBendPattern(iline) == inItMap->second[inCol].getBendPattern(iline));
      }
    }
    ++inItMap;
  }
}

BOOST_AUTO_TEST_CASE(TestRawSmall)
{
  std::map<uint16_t, std::vector<o2::mid::ColumnData>> inData;
  // Small standard event
  uint16_t localClock = 100;
  inData[localClock].emplace_back(getColData(2, 4, 0, 0xFFFF));
  inData[localClock].emplace_back(getColData(2, 5, 0xFFFF, 0));
  inData[localClock].emplace_back(getColData(4, 6, 0xFF0F, 0, 0xF0FF, 0xF));
  localClock = 200;
  inData[localClock].emplace_back(getColData(34, 3, 0xFF00, 0xFF));
  o2::mid::EventType inEventType = o2::mid::EventType::Standard;
  o2::mid::Encoder encoder;
  o2::mid::Decoder decoder;
  decoder.init();
  uint32_t orbit = 20;
  for (auto& item : inData) {
    encoder.newHeader(40, ++orbit, 0);
    encoder.process(item.second, item.first, inEventType);
  }
  decoder.process(encoder.getBuffer());
  doTest(inEventType, inData, decoder.getROFRecords(), decoder.getData());
}

BOOST_AUTO_TEST_CASE(TestRawBig)
{
  std::map<uint16_t, std::vector<o2::mid::ColumnData>> inData;
  // Big event that should pass the 8kB
  for (int irepeat = 0; irepeat < 4; ++irepeat) {
    uint16_t localClock = 100 + irepeat;
    for (int ide = 0; ide < 72; ++ide) {
      for (int icol = 0; icol < 7; ++icol) {
        inData[localClock].emplace_back(getColData(ide, icol, 0xFF00, 0xFFFF));
      }
    }
  }
  o2::mid::EventType inEventType = o2::mid::EventType::Standard;
  o2::mid::Encoder encoder;
  uint32_t orbit = 20;
  encoder.newHeader(40, ++orbit, 0);
  for (auto& item : inData) {
    encoder.process(item.second, item.first, inEventType);
  }
  o2::mid::Decoder decoder;
  decoder.init();
  decoder.process(encoder.getBuffer());
  doTest(inEventType, inData, decoder.getROFRecords(), decoder.getData());
}

BOOST_AUTO_TEST_CASE(TestRawHeaderOnly)
{
  o2::mid::Encoder encoder;
  for (uint32_t iorbit = 1; iorbit < 10; ++iorbit) {
    encoder.newHeader(40, iorbit, 0);
  }
  // End of data
  encoder.newHeader(0, 0, 1);
  o2::mid::Decoder decoder;
  decoder.init();
  decoder.process(encoder.getBuffer());
  BOOST_TEST(decoder.getROFRecords().size() == 0);
}

BOOST_AUTO_TEST_SUITE_END()
