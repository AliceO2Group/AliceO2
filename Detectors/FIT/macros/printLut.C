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

namespace print_lut
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
} // namespace print_lut

void printLut(const std::string fileName, const std::string objectName = "ccdb_object")
{
  std::vector<o2::fit::EntryFEE> lut = print_lut::readLutFromFile(fileName, objectName);
  const size_t size = lut.size();

  std::cout << "--- Lookup table ---" << std::endl;

  for (size_t idx = 0; idx < size; idx++) {
    std::cout << lut[idx] << std::endl;
  }
}