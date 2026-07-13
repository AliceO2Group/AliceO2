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

R__LOAD_LIBRARY(libO2CommonUtils)
R__LOAD_LIBRARY(libO2DataFormatsFIT)

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFIT/LookUpTable.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"
#endif

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

void printLUT(const std::string fileA, const std::string objectName = "ccdb_object")
{
  std::vector<o2::fit::EntryFEE> lut = readLUTFromFile(fileA, objectName);
  const size_t size = lut.size();

  std::cout << "--- Lookup table ---" << std::endl;

  for (size_t idx = 0; idx < size; idx++) {
    std::cout << lut[idx] << std::endl;
  }
}
