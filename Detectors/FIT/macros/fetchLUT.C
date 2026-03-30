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

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFIT/LookUpTable.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"

#endif

void saveToCSV(const std::vector<o2::fit::EntryFEE>& lut, string_view path);
void saveToRoot(std::shared_ptr<std::vector<o2::fit::EntryFEE>> lut, string_view path);

void fetchLUT(const std::string ccdbUrl, const std::string detector, long timestamp = -1, const std::string fileName = "", bool asCsv = true)
{
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  const std::string ccdbPath = detector + "/Config/LookupTable";
  std::map<std::string, std::string> metadata;

  if (timestamp == -1) {
    timestamp = o2::ccdb::getCurrentTimestamp();
  }

  std::shared_ptr<std::vector<o2::fit::EntryFEE>> lut(ccdbApi.retrieveFromTFileAny<std::vector<o2::fit::EntryFEE>>(ccdbPath, metadata, timestamp));

  if (!lut) {
    LOGP(error, "LUT object not found in {}/{} for timestamp {}.", ccdbUrl, ccdbPath, timestamp);
    return;
  } else {
    LOGP(info, "Successfully fetched LUT for {} from {}", detector, ccdbUrl);
  }

  std::cout << detector << " lookup table: " << std::endl;
  for (const auto& entry : (*lut)) {
    std::cout << entry << std::endl;
  }

  if (fileName.empty()) {
    return;
  }

  if (asCsv) {
    saveToCSV(*lut, fileName);
  } else {
    saveToRoot(lut, fileName);
  }
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

void saveToRoot(std::shared_ptr<std::vector<o2::fit::EntryFEE>> lut, string_view path)
{
  TFile file(path.data(), "RECREATE");
  if (file.IsOpen() == false) {
    LOGP(fatal, "Failed to open file {}", path.data());
  }

  file.WriteObject(lut.get(), "LookupTable");
  file.Close();
}