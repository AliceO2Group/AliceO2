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

namespace {
void saveToRoot(std::shared_ptr<std::vector<o2::fit::EntryFEE>> lut, const std::string& path)
{
  TFile file(path.data(), "RECREATE");
  if (file.IsOpen() == false) {
    std::cerr << "Failed to open file " << path << std::endl;
  }

  file.WriteObject(lut.get(), "ccdb_object");
  file.Close();
}

void _fetchLut(const std::string ccdbUrl="alice-ccdb.cern.ch", const std::string detector="FT0", long timestamp = -1, const std::string fileName = "o2_lut.root")
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
    std::cerr << "LUT object not found in " << ccdbUrl << "/" << ccdbPath << " for timestamp " << timestamp << std::endl;
    return;
  } else {
    std::cout << "Successfully fetched LUT for " << detector << " from " << ccdbUrl << std::endl;
  }

  if (fileName.empty()) {
    return;
  }

  saveToRoot(lut, fileName);
}
}

void fetchLut(const std::string ccdbUrl="alice-ccdb.cern.ch", const std::string detector="FT0", long timestamp = -1, const std::string fileName = "o2_lut.root")
{
  _fetchLut(ccdbUrl, detector, timestamp, fileName);
}