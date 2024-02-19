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

/// \file   MID/Calibration/macros/ccdbUtils.cxx
/// \brief  Retrieve or upload MID calibration objects
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 May 2022

#include <string>
#include <map>
#include <vector>
#include "TFile.h"
#include "TObjString.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ROBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"
#include "MIDRaw/DecodedDataAggregator.h"
#include "MIDFiltering/FetToDead.h"

const std::string BadChannelCCDBPath = "MID/Calib/BadChannels";

/// @brief Prints the list of bad channels from the CCDB
/// @param ccdbUrl CCDB url
/// @param timestamp Timestamp
/// @param verbose True for verbose output
void queryBadChannels(const char* ccdbUrl, long timestamp, bool verbose, const std::string path)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  auto* badChannels = api.retrieveFromTFileAny<std::vector<o2::mid::ColumnData>>(path.c_str(), metadata, timestamp);
  if (!badChannels) {
    std::cout << "Error: cannot find " << path << " in " << ccdbUrl << std::endl;
    return;
  }
  std::cout << "number of bad channels = " << badChannels->size() << std::endl;
  if (verbose) {
    std::sort(badChannels->begin(), badChannels->end(), [](const o2::mid::ColumnData& c1, const o2::mid::ColumnData& c2) { return o2::mid::getColumnDataUniqueId(c1.deId, c1.columnId) < o2::mid::getColumnDataUniqueId(c2.deId, c2.columnId); });
    for (const auto& badChannel : *badChannels) {
      std::cout << badChannel << "\n";
    }
  }
}

/// @brief Returns the masks from the DCS CCDB
/// @param ccdbUrl CCDB url
/// @param timestamp Timestamp
/// @param verbose True for verbose output
/// @return Masks as string
std::string queryDCSMasks(const char* ccdbUrl, long timestamp, bool verbose)
{
  o2::ccdb::CcdbApi api;
  std::string maskCCDBPath = "MID/Calib/ElectronicsMasks";
  api.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  auto* masks = api.retrieveFromTFileAny<TObjString>(maskCCDBPath.c_str(), metadata, timestamp);
  if (!masks) {
    std::cout << "Error: cannot find " << maskCCDBPath << " in " << ccdbUrl << std::endl;
    return "";
  }
  if (verbose) {
    std::cout << masks->GetName() << "\n";
  }
  return masks->GetName();
}

/// @brief Writes the masks from the DCS CCDB to a text file
/// @param ccdbUrl DCS CCDB url
/// @param timestamp Timestamp
/// @param outFilename Output text filename
void writeDCSMasks(const char* ccdbUrl, long timestamp, const char* outFilename = "masks.txt")
{
  auto masks = queryDCSMasks(ccdbUrl, timestamp, false);
  std::ofstream outFile(outFilename);
  if (!outFile.is_open()) {
    std::cout << "Error: cannot write to file " << outFilename << std::endl;
    return;
  }
  outFile << masks << std::endl;
  outFile.close();
}

/// @brief Uploads the list of channels provided
/// @param ccdbUrl CCDB url
/// @param timestamp Timestamp
/// @param badChannels List of bad channels. Default is no bad channel
/// @param path Calibration object path
void uploadBadChannels(const char* ccdbUrl, long timestamp, const std::string path, std::vector<o2::mid::ColumnData> channels = {})
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  std::cout << "Storing MID problematic channels (valid from " << timestamp << ") to " << path << "\n";

  api.storeAsTFileAny(&channels, path, md, timestamp, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
}

/// @brief Reads the DCS masks from a file
/// @param filename Root or txt filename
/// @return DCS masks as string
std::string readDCSMasksFile(std::string filename)
{
  std::string out;
  if (filename.find(".root") != std::string::npos) {
    TFile* file = TFile::Open(filename.data());
    auto obj = file->Get("ccdb_object");
    out = obj->GetName();
    delete file;
  } else {
    std::ifstream inFile(filename);
    if (inFile.is_open()) {
      std::stringstream ss;
      ss << inFile.rdbuf();
      out = ss.str();
    }
  }
  return out;
}

/// @brief Returns the list of masked channels from the DCS masks
/// @param masksTxt DCS masks as string
/// @return Vector of bad channels
std::vector<o2::mid::ColumnData> getBadChannelsFromDCSMasks(const char* masksTxt)
{
  std::stringstream ss;
  ss << masksTxt;
  o2::mid::ROBoardConfigHandler cfgHandler(ss);
  auto cfgMap = cfgHandler.getConfigMap();
  std::vector<o2::mid::ROBoard> boards;
  o2::mid::ROBoard board;
  board.statusWord = o2::mid::raw::sCARDTYPE | o2::mid::raw::sACTIVE;
  for (auto& item : cfgMap) {
    board.boardId = item.second.boardId;
    bool isMasked = item.second.configWord & o2::mid::crateconfig::sMonmoff;
    board.firedChambers = 0;
    for (size_t ich = 0; ich < 4; ++ich) {
      board.patternsBP[ich] = isMasked ? item.second.masksBP[ich] : 0xFFFF;
      board.patternsNBP[ich] = isMasked ? item.second.masksNBP[ich] : 0xFFFF;
      if (board.patternsBP[ich] || board.patternsNBP[ich]) {
        board.firedChambers |= 1 << ich;
      }
    }
    boards.emplace_back(board);
  }
  std::vector<o2::mid::ROFRecord> rofs;
  rofs.emplace_back(o2::InteractionRecord{2, 2}, o2::mid::EventType::Standard, 0, boards.size());
  o2::mid::DecodedDataAggregator aggregator;
  aggregator.process(boards, rofs);
  o2::mid::FetToDead ftd;
  return ftd.process(aggregator.getData());
}

/// @brief Uploads the bad channels list from DCS mask to CCDB
/// @param filename DCS mask text filename
/// @param timestamp Timestamp
/// @param ccdbUrl CCDB url
void uploadBadChannelsFromDCSMask(const char* filename, long timestamp, const char* ccdbUrl, bool verbose)
{
  auto masks = readDCSMasksFile(filename);
  auto badChannels = getBadChannelsFromDCSMasks(masks.data());
  if (verbose) {
    for (auto& col : badChannels) {
      std::cout << col << std::endl;
    }
  }
  uploadBadChannels(ccdbUrl, timestamp, BadChannelCCDBPath, badChannels);
}

/// @brief Generates the list of fake dead channels
/// @return Fake dead channels
std::vector<o2::mid::ColumnData> makeFakeDeadChannels()
{
  // In the comments:
  // - the fake deads are obtained from special calibration runs with modified FET phase
  // - expected refers to the expected fake dead channels obtained by Baptiste
  //   by comparing the masked channels with the default masks.
  std::vector<o2::mid::ColumnData> fakeDeads;
  // fakeDeads.push_back({0, 3, 0x81, 0x0, 0x0, 0x0, 0x0}); // 40; X1; expected: 0x0; data 10/2023
  fakeDeads.push_back({6, 5, 0x0, 0x2a00, 0x0, 0x0, 0x0});  // 6c; X1; expected: 0x0; data 12/2023: 0x2e00
  fakeDeads.push_back({7, 5, 0x0, 0x6bff, 0x0, 0x0, 0x0});  // 6e; X1; expected: 0x0; data 10/2023: 0x28ff
  fakeDeads.push_back({9, 5, 0x0, 0x0, 0x0, 0x0, 0x2});     // 60; Y2; expected: 0x0
  fakeDeads.push_back({10, 2, 0x0, 0x0, 0x0, 0x0, 0xe0});   // 31; Y2;
  fakeDeads.push_back({10, 4, 0x805f, 0x0, 0x0, 0x0, 0x0}); // 51; X2; expected: 0x905e; data 12/2023: 0x5f
  fakeDeads.push_back({14, 5, 0x0, 0x0, 0x0, 0x0, 0x80});   // 69; Y2;
  fakeDeads.push_back({16, 2, 0x0, 0x0, 0x0, 0x0, 0xe0});   // 2c; Y2; expected: 0x80; data 10/2023: 0xa0
  fakeDeads.push_back({16, 3, 0x0, 0xf00, 0x0, 0x0, 0x0});  // 4e; X2; expected: 0x0; data 10/2023: 0x300
  fakeDeads.push_back({25, 1, 0x0, 0x0, 0x0, 0x0, 0x63});   // 24; Y3; expected: 0xa
  // fakeDeads.push_back({34, 2, 0x0, 0x0, 0x0, 0x0, 0x20});   // 2c; Y4; expected: 0x0; data 10/2023
  fakeDeads.push_back({42, 5, 0x0, 0x2d00, 0x0, 0x0, 0x0}); // ec; X2; expected: 0x0; data 12/2023: 0x2f00
  fakeDeads.push_back({43, 4, 0x0, 0x0, 0x0, 0x0, 0x20});   // dd; Y1; expected: 0x0
  fakeDeads.push_back({43, 5, 0x0, 0xff, 0x0, 0x0, 0x0});   // ee; X1; expected: 0x0; data 10/2023: 0xf3
  fakeDeads.push_back({44, 3, 0x0, 0x0, 0x0, 0x0, 0x2});    // cf; Y1; expected: 0x0
  fakeDeads.push_back({44, 5, 0x2000, 0x0, 0x0, 0x0, 0x0}); // ef; X1; expected: 0x0
  fakeDeads.push_back({44, 6, 0x0, 0x0, 0x0, 0x0, 0xf});    // f8; Y1; expected: 0x0; data 10/2023: 0x7
  fakeDeads.push_back({46, 1, 0x0, 0x0, 0x0, 0x0, 0xc0});   // 91; Y2; expected: 0xc0; data 12/2023: 0x40
  fakeDeads.push_back({46, 2, 0x0, 0x0, 0x0, 0x0, 0xc0});   // b1; Y2; expected: 0xe0; data 10/2023: 0xe0
  fakeDeads.push_back({46, 3, 0x0, 0x0, 0x0, 0x0, 0xc0});   // c1; Y2;
  fakeDeads.push_back({46, 4, 0x0, 0x0, 0x0, 0x0, 0x58});   // d1; Y2; expected: 0xe8; data 10/2023: 0x60
  fakeDeads.push_back({46, 5, 0x0, 0x0, 0x0, 0x0, 0x60});   // e1; Y2;
  fakeDeads.push_back({47, 3, 0x1, 0x0, 0x0, 0x0, 0x0});    // c3; X2; expected: 0x9d; data 10/2023: 0x15
  // fakeDeads.push_back({47, 5, 0x1, 0x0, 0x0, 0x0, 0x0});  // e3; X2;  0 (expected only)
  fakeDeads.push_back({52, 2, 0x0, 0x0, 0x0, 0x0, 0xf8}); // ac; Y2; expected: 0xb0; data 10/2023: 0xf0
  fakeDeads.push_back({52, 3, 0x0, 0x0, 0x0, 0x0, 0x80}); // cd; Y2; expected: 0x0
  // fakeDeads.push_back({64, 4, 0x0, 0x0, 0x0, 0x0, 0x40}); // d1; Y4; expected: 0x0; data 12/2023
  fakeDeads.push_back({68, 5, 0x0, 0x0, 0x0, 0x0, 0xe2}); // e9; Y4; expected: 0x0; data 10/2023: 0xe0

  return fakeDeads;
}

/// @brief Utility to query or upload bad channels and masks from/to the CCDB
/// @param what Command to be executed
/// @param timestamp Timestamp
/// @param verbose True for verbose output
/// @param ccdbUrl CCDB url
void ccdbUtils(const char* what, long timestamp = 0, const char* inFilename = "mask.txt", bool verbose = false, const char* ccdbUrl = "http://ccdb-test.cern.ch:8080")
{
  if (timestamp == 0) {
    timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::vector<std::string> whats = {"querybad", "uploadbad", "queryfake", "uploadfake", "querymasks", "writemasks", "uploadbadfrommasks"};

  const std::string fakeDeadChannelCCDBPath = "MID/Calib/FakeDeadChannels";

  if (what == whats[0]) {
    queryBadChannels(ccdbUrl, timestamp, verbose, BadChannelCCDBPath);
  } else if (what == whats[1]) {
    uploadBadChannels(ccdbUrl, timestamp, BadChannelCCDBPath);
  } else if (what == whats[2]) {
    queryBadChannels(ccdbUrl, timestamp, verbose, fakeDeadChannelCCDBPath);
  } else if (what == whats[3]) {
    uploadBadChannels(ccdbUrl, timestamp, fakeDeadChannelCCDBPath);
  } else if (what == whats[4]) {
    queryDCSMasks(ccdbUrl, timestamp, verbose);
  } else if (what == whats[5]) {
    writeDCSMasks(ccdbUrl, timestamp);
  } else if (what == whats[6]) {
    uploadBadChannelsFromDCSMask(inFilename, timestamp, ccdbUrl, verbose);
  } else {
    std::cout << "Unimplemented option chosen " << what << std::endl;
    std::cout << "Available:\n";
    for (auto& str : whats) {
      std::cout << str << " ";
    }
    std::cout << std::endl;
  }
}
