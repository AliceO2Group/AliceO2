// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/test/testRaw.cxx
/// \brief  Test MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 March 2018

#define BOOST_TEST_MODULE Test MID raw
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include <vector>
#include <map>
#include "Framework/Logger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/RawFileReader.h"
#include "DataFormatsMID/ColumnData.h"
#include "Headers/RAWDataHeader.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/Mapping.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/DecodedDataAggregator.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/Encoder.h"
#include "MIDRaw/GBTUserLogicEncoder.h"
#include "MIDRaw/LinkDecoder.h"

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
  std::sort(sortedData.begin(), sortedData.end(), [](o2::mid::ColumnData& a, o2::mid::ColumnData& b) { if (a.deId == b.deId ) { return (a.columnId < b.columnId); 

}return (a.deId < b.deId); });
  return sortedData;
}

void doTest(const std::map<o2::InteractionRecord, std::vector<o2::mid::ColumnData>>& inData, const std::vector<o2::mid::ROFRecord>& rofRecords, const std::vector<o2::mid::ColumnData>& data, const o2::mid::EventType inEventType = o2::mid::EventType::Standard)
{
  BOOST_REQUIRE(rofRecords.size() == inData.size());
  auto inItMap = inData.begin();
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    BOOST_TEST(static_cast<int>(rofIt->eventType) == static_cast<int>(inEventType));
    BOOST_TEST(rofIt->interactionRecord == inItMap->first);
    BOOST_TEST(rofIt->nEntries == inItMap->second.size());
    auto sortedIn = sortData(inItMap->second, 0, inItMap->second.size());
    auto sortedOut = sortData(data, rofIt->firstEntry, rofIt->firstEntry + rofIt->nEntries);
    BOOST_REQUIRE(sortedOut.size() == sortedIn.size());
    for (size_t icol = 0; icol < sortedOut.size(); ++icol) {
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

std::tuple<std::vector<o2::mid::ColumnData>, std::vector<o2::mid::ROFRecord>> encodeDecode(std::map<o2::InteractionRecord, std::vector<o2::mid::ColumnData>> inData, o2::mid::EventType inEventType = o2::mid::EventType::Standard)
{
  auto severity = fair::Logger::GetConsoleSeverity();
  fair::Logger::SetConsoleSeverity(fair::Severity::WARNING);
  std::string tmpFilename0 = "tmp_mid_raw";
  std::string tmpFilename = tmpFilename0 + ".raw";
  o2::mid::Encoder encoder;
  encoder.init(tmpFilename0.c_str());
  std::string tmpConfigFilename = "tmp_MIDConfig.cfg";
  encoder.getWriter().writeConfFile("MID", "RAWDATA", tmpConfigFilename.c_str(), false);
  for (auto& item : inData) {
    encoder.process(item.second, item.first, inEventType);
  }
  encoder.finalize();

  o2::raw::RawFileReader rawReader(tmpConfigFilename.c_str());
  rawReader.init();
  std::vector<char> buffer;
  for (size_t itf = 0; itf < rawReader.getNTimeFrames(); ++itf) {
    rawReader.setNextTFToRead(itf);
    for (size_t ilink = 0; ilink < rawReader.getNLinks(); ++ilink) {
      auto& link = rawReader.getLink(ilink);
      auto tfsz = link.getNextTFSize();
      if (!tfsz) {
        continue;
      }
      std::vector<char> linkBuffer(tfsz);
      link.readNextTF(linkBuffer.data());
      buffer.insert(buffer.end(), linkBuffer.begin(), linkBuffer.end());
    }
  }
  fair::Logger::SetConsoleSeverity(severity);

  std::remove(tmpFilename.c_str());
  std::remove(tmpConfigFilename.c_str());

  o2::mid::Decoder decoder;
  gsl::span<const uint8_t> data(reinterpret_cast<uint8_t*>(buffer.data()), buffer.size());
  decoder.process(data);

  o2::mid::DecodedDataAggregator aggregator;
  aggregator.process(decoder.getData(), decoder.getROFRecords());

  return std::make_tuple(aggregator.getData(), aggregator.getROFRecords());
}

BOOST_AUTO_TEST_CASE(ColumnDataConverter)
{
  std::map<o2::InteractionRecord, std::vector<o2::mid::ColumnData>> inData;
  o2::InteractionRecord ir(100, 0);
  // Crate 5 link 0
  inData[ir].emplace_back(getColData(2, 4, 0x1, 0xFFFF));

  ir.bc = 200;
  inData[ir].emplace_back(getColData(3, 4, 0xFF00, 0xFF));
  inData[ir].emplace_back(getColData(12, 4, 0, 0, 0xFF));

  ir.bc = 400;
  inData[ir].emplace_back(getColData(5, 1, 0xFF00, 0xFF));
  inData[ir].emplace_back(getColData(14, 1, 0, 0, 0, 0xFF));

  std::vector<o2::mid::ROFRecord> rofs;
  std::vector<o2::mid::ROBoard> outData;
  auto inEventType = o2::mid::EventType::Standard;
  o2::mid::ColumnDataToLocalBoard converter;
  converter.setDebugMode(true);
  for (auto& item : inData) {
    converter.process(item.second);
    auto firstEntry = outData.size();
    for (auto& gbtItem : converter.getData()) {
      for (auto& loc : gbtItem.second) {
        outData.emplace_back(loc);
      }
      rofs.push_back({item.first, inEventType, firstEntry, outData.size() - firstEntry});
    }
  }

  o2::mid::DecodedDataAggregator aggregator;
  aggregator.process(outData, rofs);

  doTest(inData, aggregator.getROFRecords(), aggregator.getData());
}

BOOST_AUTO_TEST_CASE(GBTUserLogicDecoder)
{
  /// Event with just one link fired

  std::map<uint16_t, std::vector<o2::mid::ROBoard>> inData;
  uint16_t bc = 100;
  o2::mid::ROBoard loc;
  // Crate 5 link 0
  loc.statusWord = o2::mid::raw::sSTARTBIT | o2::mid::raw::sCARDTYPE;
  loc.triggerWord = 0;
  loc.boardId = 2;
  loc.firedChambers = 0x1;
  loc.patternsBP[0] = 0xF;
  loc.patternsNBP[0] = 0xF;
  inData[bc].emplace_back(loc);

  bc = 200;
  loc.patternsBP.fill(0);
  loc.patternsNBP.fill(0);
  loc.boardId = 5;
  loc.firedChambers = 0x4;
  loc.patternsBP[2] = 0xF0;
  loc.patternsNBP[2] = 0x5;
  inData[bc].emplace_back(loc);

  uint8_t crateId = 5;
  uint8_t linkInCrate = 0;
  uint16_t gbtUniqueId = o2::mid::crateparams::makeGBTUniqueId(crateId, linkInCrate);
  o2::mid::GBTUserLogicEncoder encoder;
  encoder.setGBTUniqueId(gbtUniqueId);
  for (auto& item : inData) {
    encoder.process(item.second, o2::InteractionRecord(item.first, 0));
  }
  std::vector<char> buf;
  encoder.flush(buf, o2::InteractionRecord());
  o2::header::RAWDataHeader rdh;
  auto memSize = buf.size() + 64;
  rdh.word1 |= (memSize | (memSize << 16));
  // Sets the linkId
  uint16_t feeId = gbtUniqueId / 8;
  rdh.word0 |= (feeId << 16);
  auto decoder = o2::mid::createLinkDecoder(feeId);
  std::vector<o2::mid::ROBoard> data;
  std::vector<o2::mid::ROFRecord> rofs;
  std::vector<uint8_t> convertedBuffer(buf.size());
  memcpy(convertedBuffer.data(), buf.data(), buf.size());
  decoder->process(convertedBuffer, rdh, data, rofs);
  BOOST_REQUIRE(rofs.size() == inData.size());
  auto inItMap = inData.begin();
  for (auto rofIt = rofs.begin(); rofIt != rofs.end(); ++rofIt) {
    BOOST_TEST(rofIt->interactionRecord.bc == inItMap->first);
    BOOST_TEST(rofIt->nEntries == inItMap->second.size());
    auto outLoc = data.begin() + rofIt->firstEntry;
    for (auto inLoc = inItMap->second.begin(); inLoc != inItMap->second.end(); ++inLoc) {
      BOOST_TEST(inLoc->statusWord == outLoc->statusWord);
      BOOST_TEST(inLoc->triggerWord == outLoc->triggerWord);
      BOOST_TEST(o2::mid::raw::makeUniqueLocID(crateId, inLoc->boardId) == outLoc->boardId);
      BOOST_TEST(inLoc->firedChambers == outLoc->firedChambers);
      for (int ich = 0; ich < 4; ++ich) {
        BOOST_TEST(inLoc->patternsBP[ich] == outLoc->patternsBP[ich]);
        BOOST_TEST(inLoc->patternsNBP[ich] == outLoc->patternsNBP[ich]);
      }
      ++outLoc;
    }
    ++inItMap;
  }
}

BOOST_AUTO_TEST_CASE(SmallSample)
{
  /// Event with just one link fired

  std::map<o2::InteractionRecord, std::vector<o2::mid::ColumnData>> inData;
  // Small standard event
  o2::InteractionRecord ir(100, 0);

  // Crate 5 link 0
  inData[ir].emplace_back(getColData(2, 4, 0x1, 0xFFFF));
  inData[ir].emplace_back(getColData(11, 4, 0x3, 0xFFFF));

  // Crate 1 link 1 and crate 2 link 0
  inData[ir].emplace_back(getColData(5, 1, 0xFFFF, 0, 0xF, 0xF0));
  // Crate 10 link 1 and crate 11 link 1
  inData[ir].emplace_back(getColData(41, 2, 0xFF0F, 0, 0xF0FF, 0xF));
  ir.bc = 0xde6;
  ir.orbit = 2;
  // Crate 12 link 1
  inData[ir].emplace_back(getColData(70, 3, 0xFF00, 0xFF));

  ir.bc = 0xdea;
  ir.orbit = 3;
  inData[ir].emplace_back(getColData(70, 3, 0xFF00, 0xFF));

  auto [data, rofs] = encodeDecode(inData);

  doTest(inData, rofs, data);
}

BOOST_AUTO_TEST_CASE(LargeBufferSample)
{
  o2::mid::Mapping mapping;
  std::map<o2::InteractionRecord, std::vector<o2::mid::ColumnData>> inData;
  // Big event that should pass the 8kB
  o2::InteractionRecord ir(0, 1);
  for (int irepeat = 0; irepeat < 4000; ++irepeat) {
    ++ir;
    for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
      // Since we have 1 RDH per GBT, we can put data only on 1 column
      int icol = 4;
      // for (int icol = mapping.getFirstColumn(ide); icol < 7; ++icol) {
      if (mapping.getFirstBoardBP(icol, ide) != 0) {
        continue;
      }
      inData[ir].emplace_back(getColData(ide, icol, 0xFF00, 0xFFFF));
    }
  }

  auto [data, rofs] = encodeDecode(inData);

  doTest(inData, rofs, data);
}

BOOST_AUTO_TEST_SUITE_END()
