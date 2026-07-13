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
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <array>
#include <ranges>

R__LOAD_LIBRARY(libO2CommonUtils)
R__LOAD_LIBRARY(libO2CCDB)
R__LOAD_LIBRARY(libO2DataFormatsFIT)

#include "DataFormatsFIT/LookUpTable.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"
#endif

void saveToRoot(std::vector<o2::fit::EntryFEE>& lut, string_view path);

void convertLUTCSVtoROOT(const std::string csvFilePath, const std::string rootFilePath)
{
  std::vector<o2::fit::EntryFEE> lut;
  std::ifstream lutCSV(csvFilePath);
  if (lutCSV.is_open() == false) {
    LOGP(error, "Failed to open {}", csvFilePath);
    return;
  }

  std::string line;
  std::getline(lutCSV, line);
  std::map<std::string, int> headerMap;

  auto headersView = std::string_view(line) | std::views::split(',');

  int index = 0;
  for (auto&& rng : headersView) {
    std::string columnName(rng.begin(), rng.end());
    headerMap[columnName] = index++;
  }

  while (std::getline(lutCSV, line)) {
    if (line.size() == 0) {
      return;
    }
    o2::fit::EntryFEE entry;
    auto fieldViews = std::string_view(line) | std::views::split(',');
    std::vector<std::string> parsedLine;
    for (auto&& view : fieldViews) {
      parsedLine.emplace_back(view.begin(), view.end());
    }
    if (parsedLine.size() < headerMap.size()) {
      LOGP(error, "Ill-formed line: {}", line);
      return;
    }

    entry.mEntryCRU.mLinkID = std::stoi(parsedLine[headerMap.at("LinkID")]);
    entry.mEntryCRU.mEndPointID = std::stoi(parsedLine[headerMap.at("EndPointID")]);
    entry.mEntryCRU.mCRUID = std::stoi(parsedLine[headerMap.at("CRUID")]);
    entry.mEntryCRU.mFEEID = std::stoi(parsedLine[headerMap.at("FEEID")]);

    entry.mModuleType = parsedLine[headerMap.at("ModuleType")];
    entry.mLocalChannelID = parsedLine[headerMap.at("LocalChannelID")];
    entry.mChannelID = parsedLine[headerMap.at("channel #")];
    entry.mModuleName = parsedLine[headerMap.at("Module")];
    entry.mBoardHV = parsedLine[headerMap.at("HV board")];
    entry.mChannelHV = parsedLine[headerMap.at("HV channel")];
    entry.mSerialNumberMCP = parsedLine[headerMap.at("MCP S/N")];
    entry.mCableHV = parsedLine[headerMap.at("HV cable")];
    entry.mCableSignal = parsedLine[headerMap.at("signal cable")];
    lut.emplace_back(entry);
  }
  saveToRoot(lut, rootFilePath);
}

void saveToRoot(std::vector<o2::fit::EntryFEE>& lut, string_view path)
{
  TFile file(path.data(), "RECREATE");
  if (file.IsOpen() == false) {
    LOGP(fatal, "Failed to open file {}", path.data());
  }

  file.WriteObject(&lut, "LookupTable");
  file.Close();
}