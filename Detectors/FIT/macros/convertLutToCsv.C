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
R__LOAD_LIBRARY(libO2DataFormatsFIT)

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFIT/LookUpTable.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"

#endif

void saveToCSV(const std::vector<o2::fit::EntryFEE>& lut, string_view path);

std::vector<o2::fit::EntryFEE> readLUTFromFile(const std::string filePath, const std::string objectName)
{
  TFile file(filePath.c_str(), "READ");
  if (file.IsOpen() == false) {
    LOGP(fatal, "Failed to open {}", filePath);
  }
  LOGP(info, "Successfully opened {}", filePath);
  std::vector<o2::fit::EntryFEE>* lut = nullptr;
  file.GetObject<std::vector<o2::fit::EntryFEE>>(objectName.c_str(), lut);
  if (lut == nullptr) {
     LOGP(fatal, "Failed to read object {}", objectName);
  }
  LOGP(info, "Successfully get {} object", objectName);
  std::vector<o2::fit::EntryFEE> lutCopy = *lut;
  file.Close();
  return lutCopy;
}

void fetchLUT(const std::string fileName, cosnt std::string objectName, const std::string csvName)
{
  if (fileName.empty() || objectName.empty() || csvName.empty()) {
    return;
  }
  std::vector<o2::fit::EntryFEE> lut = readLUTFromFile(fileA, objectName);
  saveToCSV(lut, csvName);
}

void saveToCSV(const std::vector<o2::fit::EntryFEE>& lut, string_view path)
{
  std::ofstream ofs(path.data());
  if (!ofs.is_open()) {
    LOGP(error, "Cannot open file for writing: {}", path);
    return;
  }
  ofs << "LinkID,EndPointID,CRUID,FEEID,ModuleType,LocalChannelID,channel #,Module,HV board,HV channel,MCP S/N,HV cable,signal cable\n";
  for (const auto& entry : lut) {
    ofs << entry.mEntryCRU.mLinkID << ","
        << entry.mEntryCRU.mEndPointID << ","
        << entry.mEntryCRU.mCRUID << ","
        << entry.mEntryCRU.mFEEID << ","
        << entry.mModuleType << ","
        << entry.mLocalChannelID << ","
        << entry.mChannelID << ","
        << entry.mModuleName << ","
        << entry.mBoardHV << ","
        << entry.mChannelHV << ","
        << entry.mSerialNumberMCP << ","
        << entry.mCableHV << ","
        << entry.mCableSignal << "\n";
  }
  ofs.close();
}
