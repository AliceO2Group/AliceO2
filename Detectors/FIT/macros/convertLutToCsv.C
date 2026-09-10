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

namespace convert_lut_to_csv
{
std::vector<o2::fit::EntryFEE> readLutFromFile(const std::string filePath, const std::string objectName)
{
  TFile file(filePath.c_str(), "READ");
  if (file.IsOpen() == false) {
    std::cerr << "Failed to open " << filePath << std::endl;
    return {};
  }
  std::cout << "Successfully opened " << std::endl
            << filePath;

  std::vector<o2::fit::EntryFEE>* lut = nullptr;
  file.GetObject<std::vector<o2::fit::EntryFEE>>(objectName.c_str(), lut);

  if (lut == nullptr) {
    std::cerr << "Failed to read object " << objectName << std::endl;
    return {};
  }
  std::cout << "Successfully get " << objectName << " object" << std::endl;

  std::vector<o2::fit::EntryFEE> lutCopy = *lut;
  file.Close();

  return std::move(lutCopy);
}
} // namespace convert_lut_to_csv

void saveToCSV(const std::vector<o2::fit::EntryFEE>& lut, const std::string& path)
{
  std::ofstream ofs(path.data());
  if (!ofs.is_open()) {
    std::cerr << "Cannot open file for writing: " << path << std::endl;
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

void convertLutToCsv(const std::string fileName, const std::string objectName, const std::string csvName)
{
  if (fileName.empty() || objectName.empty() || csvName.empty()) {
    return;
  }
  std::vector<o2::fit::EntryFEE> lut = convert_lut_to_csv::readLutFromFile(fileName, objectName);
  saveToCSV(lut, csvName);
}