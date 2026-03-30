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
#endif


R__LOAD_LIBRARY(libO2CommonUtils)
R__LOAD_LIBRARY(libO2DataFormatsFIT)

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFIT/LookUpTable.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"

std::vector<o2::fit::EntryFEE> readLUTFromFile(const std::string filePath, const std::string objectName)
{
    TFile file(filePath.c_str(), "READ");
    if(file.IsOpen() == false) {
    LOGP(fatal, "Failed to open {}", filePath);
    }
    LOGP(info, "Successfully opened {}", filePath);

    std::vector<o2::fit::EntryFEE>* lut = nullptr;
    file.GetObject<std::vector<o2::fit::EntryFEE>>(objectName.c_str(), lut);

    if(lut == nullptr) {
    LOGP(fatal, "Failed to read object {}", objectName);
    }
    LOGP(info, "Successfully get {} object", objectName);

    std::vector<o2::fit::EntryFEE> lutCopy = *lut;
    file.Close();

    return std::move(lutCopy);
}

inline bool operator==(const o2::fit::EntryFEE& lhs, const o2::fit::EntryFEE& rhs)
{
  auto comparer = [](const o2::fit::EntryFEE& e) { 
    return std::tie(
      e.mEntryCRU.mLinkID, e.mEntryCRU.mEndPointID, e.mEntryCRU.mCRUID, e.mEntryCRU.mFEEID,
      e.mChannelID, e.mLocalChannelID, e.mModuleType, e.mModuleName, 
      e.mBoardHV, e.mChannelHV, e.mSerialNumberMCP, e.mCableHV, e.mCableSignal
    ); 
  };
  return comparer(lhs) == comparer(rhs);
}


void compareLUT(const std::string fileA, const std::string fileB, bool compareEvenForDifferentSize = false, const std::string objectName = "LookupTable")
{
    std::vector<o2::fit::EntryFEE> lutA = readLUTFromFile(fileA, objectName);
    std::vector<o2::fit::EntryFEE> lutB = readLUTFromFile(fileB, objectName);

    bool lutAreSame = true;

    if(lutA.size() != lutB.size()) {
        LOGP(error, "The LUT vary in size: {} for {} vs {} for {}", lutA.size(), fileA, lutB.size(), fileB);
        lutAreSame = false;
        if(compareEvenForDifferentSize == false) {
            return;
        }
    }

    size_t size = (lutA.size() < lutB.size()) ? lutA.size() : lutB.size();
    
    std::cout << "--- Comparision ---" << std::endl;

    for(size_t idx = 0; idx < size; idx++) {
        if(lutA[idx] == lutB[idx]) {
            continue;
        } else {
            std::cout << "Entry " << idx << " in " << fileA << " entry: " << lutA[idx] << std::endl;
            std::cout << "Entry " << idx << " in " << fileB << " entry: " << lutB[idx] << std::endl;
            lutAreSame = false;
        }
    }

    for(size_t idx = size; idx < lutA.size(); idx++) {
        std::cout << "Extra entry " << idx << " in " << fileA << ": " << lutA[idx] << std::endl;
    }

    for(size_t idx = size; idx < lutB.size(); idx++) {
        std::cout << "Extra entry " << idx << " in " << fileB << ": " << lutB[idx] << std::endl;
    }

    if(lutAreSame) {
        std::cout << "LUTs are the same!" << std::endl;
    } else {
        std::cout << "LUTs are different!" << std::endl;
    }
}
