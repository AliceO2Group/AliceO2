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
#include <string>
#include "TFile.h"
#include <iostream>
#include "FV0Calibration/FV0ChannelTimeCalibrationObject.h"
#include "CCDB/CcdbApi.h"
#endif

// Macro retrieves only latest set of calibrations
int readChannelTimeOffsetFV0CalibObjectFromCCDB(const std::string url = "http://alice-ccdb.cern.ch/")
{
  o2::ccdb::CcdbApi api;
  api.init(url);
  map<string, string> metadata;
  map<string, string> headers;
  auto retrieved = api.retrieveFromTFileAny<o2::fv0::FV0ChannelTimeCalibrationObject>("FV0/Calib/ChannelTimeOffset", metadata, -1, &headers);

  std::cout << "--- HEADERS ---" << std::endl;
  for (const auto& [key, value] : headers) {
    std::cout << key << " = " << value << std::endl;
  }
  std::cout << std::endl
            << "---------------" << std::endl;

  for (uint16_t ich = 0; ich < retrieved->mTimeOffsets.size(); ++ich) {
    std::cout << "Channel: " << ich << "\t|  CalibCoeff: " << retrieved->mTimeOffsets.at(ich) << std::endl;
  }
  std::cout << std::endl;
  return 0;
}
