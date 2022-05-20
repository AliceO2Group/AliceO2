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

int makeChannelTimeOffsetFV0CalibObjectInCCDB(const std::string url = "http://alice-ccdb.cern.ch/")
{
  o2::ccdb::CcdbApi api;
  api.init(url);
  std::map<std::string, std::string> md;
  o2::fv0::FV0ChannelTimeCalibrationObject obj;
  for (auto& dummyCalCoeff : obj.mTimeOffsets) {
    dummyCalCoeff = 0;
  }
  api.storeAsTFileAny(&obj, "FV0/Calib/ChannelTimeOffset", md, 0);
  return 0;
}
