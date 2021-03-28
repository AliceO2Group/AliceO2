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
#include "FT0Calibration/FT0CalibrationObject.h"


int makeDummyFT0CalibObjectInCCDB(const std::string url = "http://localhost:8080")
{

  const char* OBJECT_PATH = "FT0/Calibration/CalibrationObject";

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;

  o2::ft0::FT0CalibrationObject calibObject;
  api.storeAsTFileAny(&calibObject, OBJECT_PATH, md, 0);

  const auto testCalibObject = api.retrieveFromTFileAny<o2::ft0::FT0CalibrationObject>(OBJECT_PATH, md, 0);
  if(testCalibObject){
    for(unsigned int i = 0; i < 208; ++i){
      if(calibObject.mChannelOffsets[i] != testCalibObject->mChannelOffsets[i]){
        std::cout << "=====FAILURE=====" << std::endl;
        std::cout << "Saved and retrieved objects are different!\n";
        return 1;
      }
    }
    std::cout << "=====SUCCESS=====" << std::endl;
    return 0;
  }

  std::cout << "=====FAILURE=====" << std::endl;
  return 1;
}
