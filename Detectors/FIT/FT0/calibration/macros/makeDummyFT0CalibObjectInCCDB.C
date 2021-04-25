// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <string>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include <iostream>
#include "FT0Calibration/FT0DummyCalibrationObject.h"

int makeDummyFT0CalibObjectInCCDB(const std::string url = "http://localhost:8080")
{

  o2::ccdb::CcdbApi api;
  api.init(url);
  std::map<std::string, std::string> md;
  o2::ft0::FT0DummyNeededCalibrationObject obj;
  api.storeAsTFileAny(&obj, "FT0/Calibration/DummyNeeded", md, 0);

  return 0;
}
