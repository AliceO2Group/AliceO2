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
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/Mapping.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/Encoder.h"
#include "MIDRaw/CRUUserLogicDecoder.h"
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

std::vector<o2::mid::ColumnData> sortData(const std::vector<o2::mid::ColumnData>& data, size_t first, size_t last)
{
  std::vector<o2::mid::ColumnData> sortedData(data.begin() + first, data.begin() + last);
  std::sort(sortedData.begin(), sortedData.end(), [](o2::mid::ColumnData& a, o2::mid::ColumnData& b) { if (a.deId == b.deId ) return (a.columnId < b.columnId); return (a.deId < b.deId); });
  return sortedData;
}

void doTest(const o2::mid::EventType& inEventType, const std::map<uint16_t, std::vector<o2::mid::ColumnData>>& inData, const std::vector<o2::mid::ROFRecord>& rofRecords, const std::vector<o2::mid::ColumnData>& data)
{
  BOOST_TEST(rofRecords.size() == inData.size());
  auto inItMap = inData.begin();
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    BOOST_TEST(static_cast<int>(rofIt->eventType) == static_cast<int>(inEventType));
    BOOST_TEST(rofIt->interactionRecord.bc == inItMap->first);
    BOOST_TEST(rofIt->nEntries == inItMap->second.size());
    auto sortedIn = sortData(inItMap->second, 0, inItMap->second.size());
    auto sortedOut = sortData(data, rofIt->firstEntry, rofIt->firstEntry + rofIt->nEntries);
    for (size_t icol = 0; icol < sortedOut.size(); ++icol) {
      // size_t inCol = icol - rofIt->firstEntry;
      BOOST_TEST(sortedOut[icol].deId == sortedIn[icol].deId);
      BOOST_TEST(sortedOut[icol].columnId == sortedIn[icol].columnId);
      BOOST_TEST(sortedOut[icol].getNonBendPattern() == sortedIn[icol].getNonBendPattern());
      for (int iline = 0; iline < 4; ++iline) {
        BOOST_TEST(sortedOut[icol].getBendPattern(iline) == sortedIn[icol].getBendPattern(iline));
      }
    }
    ++inItMap;
  }
}

BOOST_AUTO_TEST_CASE(RawBuffer)
{
  std::vector<o2::InteractionRecord> HBIRVec;
  o2::raw::HBFUtils hbfUtils;
  o2::InteractionRecord irFrom = hbfUtils.getFirstIR();
  o2::InteractionRecord ir(5, 4);
  hbfUtils.fillHBIRvector(HBIRVec, irFrom, ir);
  std::vector<uint8_t> bytes;
  unsigned int memSize = 0;
  for (auto& hbIr : HBIRVec) {
    auto rdh = hbfUtils.createRDH<o2::header::RAWDataHeader>(hbIr);
    rdh.offsetToNext = (hbIr.orbit == ir.orbit) ? 0x2000 : rdh.headerSize;
    rdh.memorySize = (hbIr.orbit == ir.orbit) ? 0x1000 : rdh.headerSize;
    memSize = rdh.memorySize - rdh.headerSize;
    auto rdhBuf = reinterpret_cast<const uint8_t*>(&rdh);
    for (size_t ii = 0; ii < rdh.headerSize; ++ii) {
      bytes.emplace_back(rdhBuf[ii]);
    }
    for (size_t ii = 0; ii < rdh.memorySize - rdh.headerSize; ++ii) {
      bytes.emplace_back(ii);
    }
    for (size_t ii = 0; ii < rdh.offsetToNext - rdh.memorySize; ++ii) {
      bytes.emplace_back(0);
    }

    if (bytes.size() < rdh.offsetToNext) {
      o2::mid::RawBuffer<uint8_t> rb;
      rb.setBuffer(bytes);
      BOOST_TEST(rb.getRDH()->word0 == rdh.word0);
    }
  }

  o2::mid::RawBuffer<uint8_t> rb;
  size_t nHeaders = 0;
  rb.setBuffer(bytes);
  // Reads only the headers
  while (rb.nextHeader()) {
    ++nHeaders;
  }
  BOOST_TEST(nHeaders == HBIRVec.size());

  // Set buffer again after full reset
  rb.setBuffer(bytes, o2::mid::RawBuffer<uint8_t>::ResetMode::all);
  rb.next();
  BOOST_TEST(static_cast<int>(rb.next()) == 1);

  if (!rb.hasNext(memSize - 1)) {
    // Set buffer but keep unconsumed
    rb.setBuffer(bytes);
    // This should come from the last unconsumed buffer
    BOOST_TEST(static_cast<int>(rb.next()) == 2);

    for (int ibyte = 0; ibyte < memSize; ++ibyte) {
      rb.next();
    }
    // And this comes from the new buffer
    BOOST_TEST(static_cast<int>(rb.next()) == 3);
  }
}

BOOST_AUTO_TEST_CASE(CRUUserLogicDecoder)
{
  /// Event with just one link fired

  std::map<uint16_t, std::vector<o2::mid::ColumnData>> inData;
  uint16_t bc = 100;
  // Crate 5 link 0
  inData[bc].emplace_back(getColData(2, 4, 0, 0xFFFF));

  bc = 200;
  inData[bc].emplace_back(getColData(3, 4, 0xFF00, 0xFF));
  o2::mid::Encoder encoder;
  uint32_t orbit = 0;
  for (auto& item : inData) {
    o2::InteractionRecord ir(item.first, orbit);
    encoder.process(item.second, ir, o2::mid::EventType::Standard);
  }
  o2::mid::CRUUserLogicDecoder CRUUserLogicDecoder;
  CRUUserLogicDecoder.process(encoder.getBuffer());
  BOOST_TEST(CRUUserLogicDecoder.getROFRecords().size() == inData.size());
}

BOOST_AUTO_TEST_CASE(SmallSample)
{
  std::map<uint16_t, std::vector<o2::mid::ColumnData>> inData;
  // Small standard event
  uint16_t bc = 100;

  // Crate 5 link 0
  inData[bc].emplace_back(getColData(2, 4, 0, 0xFFFF));
  inData[bc].emplace_back(getColData(11, 4, 0, 0xFFFF));

  // Crate 1 link 1 and crate 2 link 0
  inData[bc].emplace_back(getColData(5, 1, 0xFFFF, 0, 0xF, 0xF0));
  // Crate 10 link 1 and crate 11 link 1
  inData[bc].emplace_back(getColData(41, 2, 0xFF0F, 0, 0xF0FF, 0xF));
  bc = 200;
  // Crate 12 link 1
  inData[bc].emplace_back(getColData(70, 3, 0xFF00, 0xFF));
  o2::mid::EventType inEventType = o2::mid::EventType::Standard;
  o2::mid::Encoder encoder;
  uint32_t orbit = 0;
  for (auto& item : inData) {
    o2::InteractionRecord ir(item.first, orbit);
    encoder.process(item.second, ir, o2::mid::EventType::Standard);
  }
  o2::mid::Decoder decoder;
  decoder.process(encoder.getBuffer());
  doTest(inEventType, inData, decoder.getROFRecords(), decoder.getData());
}

BOOST_AUTO_TEST_CASE(LargeBufferSample)
{
  o2::mid::Mapping mapping;
  std::map<uint16_t, std::vector<o2::mid::ColumnData>> inData;
  // Big event that should pass the 8kB
  for (int irepeat = 0; irepeat < 150; ++irepeat) {
    uint16_t bc = 1 + irepeat;
    for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
      // Since we have 1 RDH per GBT, we can put data only on 1 column
      int icol = 4;
      // for (int icol = mapping.getFirstColumn(ide); icol < 7; ++icol) {
      if (mapping.getFirstBoardBP(icol, ide) != 0) {
        continue;
      }
      inData[bc].emplace_back(getColData(ide, icol, 0xFF00, 0xFFFF));
    }
  }
  o2::mid::EventType inEventType = o2::mid::EventType::Standard;
  o2::mid::Encoder encoder;
  uint32_t orbit = 0;
  for (auto& item : inData) {
    o2::InteractionRecord ir(item.first, orbit);
    encoder.process(item.second, ir, inEventType);
  }
  o2::mid::Decoder decoder;
  decoder.process(encoder.getBuffer());
  doTest(inEventType, inData, decoder.getROFRecords(), decoder.getData());
}

BOOST_AUTO_TEST_CASE(RawHeaderOnly)
{
  o2::mid::Encoder encoder;
  std::vector<o2::mid::ColumnData> data;
  for (uint32_t iorbit = 1; iorbit < 10; ++iorbit) {
    o2::InteractionRecord ir(0, iorbit);
    encoder.process(data, ir, o2::mid::EventType::Standard);
  }
  o2::mid::Decoder decoder;
  decoder.process(encoder.getBuffer());
  BOOST_TEST(decoder.getROFRecords().size() == 0);
}

BOOST_AUTO_TEST_SUITE_END()
