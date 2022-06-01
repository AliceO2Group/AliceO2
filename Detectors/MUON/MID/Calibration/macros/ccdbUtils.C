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
void queryBadChannels(const char* ccdbUrl, long timestamp, bool verbose)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  auto* badChannels = api.retrieveFromTFileAny<std::vector<o2::mid::ColumnData>>(BadChannelCCDBPath.c_str(), metadata, timestamp);
  if (!badChannels) {
    std::cout << "Error: cannot find " << BadChannelCCDBPath << " in " << ccdbUrl << std::endl;
    return;
  }
  std::cout << "number of bad channels = " << badChannels->size() << std::endl;
  if (verbose) {
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

/// @brief Uploads the list of bad channels provided
/// @param ccdbUrl CCDB url
/// @param timestamp Timestamp
/// @param badChannels List of bad channels. Default is no bad channel
void uploadBadChannels(const char* ccdbUrl, long timestamp, std::vector<o2::mid::ColumnData> badChannels = {})
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  std::cout << "storing default MID bad channels (valid from " << timestamp << ") to " << BadChannelCCDBPath << "\n";

  api.storeAsTFileAny(&badChannels, BadChannelCCDBPath, md, timestamp, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
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
  uploadBadChannels(ccdbUrl, timestamp, badChannels);
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

  std::vector<std::string> whats = {"querybad", "uploadbad", "querymasks", "writemasks", "uploadbadfrommasks"};

  if (what == whats[0]) {
    queryBadChannels(ccdbUrl, timestamp, verbose);
  } else if (what == whats[1]) {
    uploadBadChannels(ccdbUrl, timestamp);
  } else if (what == whats[2]) {
    queryDCSMasks(ccdbUrl, timestamp, verbose);
  } else if (what == whats[3]) {
    writeDCSMasks(ccdbUrl, timestamp);
  } else if (what == whats[4]) {
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
