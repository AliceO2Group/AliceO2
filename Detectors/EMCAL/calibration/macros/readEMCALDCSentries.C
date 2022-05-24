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

// macro to read the EMC DCS information from CCDB
// default ts is very big: Saturday, November 20, 2286 5:46:39 PM

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "EMCALCalib/ElmbData.h"
#include "EMCALCalibration/EMCDCSProcessor.h"
#include "EMCALCalib/CalibDB.h"

//#include <string>
//#include <unordered_map>
#include <chrono>
#include <bitset>
#endif

using namespace o2::emcal;

typedef std::tuple<int, float, float, float, float> Sensor_t; //{Npoints, mean, rms, min, max}
void printElmbData(std::vector<Sensor_t> data);

void readEMCALDCSentries(long ts = 9999999999990, const char* ccdb = "http://ccdb-test.cern.ch:8080")
{

  o2::ccdb::CcdbApi api;
  api.init(ccdb); // or http://localhost:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  FeeDCS* feeDCS(nullptr);
  feeDCS = api.retrieveFromTFileAny<FeeDCS>(o2::emcal::CalibDB::getCDBPathFeeDCS(), metadata, ts);
  if (!feeDCS) {
    std::cerr << "No FeeDCS object received from CCDB" << std::endl;
  } else {
    std::cout << *feeDCS << std::endl;
  }

  ElmbData* mELMBdata(nullptr);
  mELMBdata = api.retrieveFromTFileAny<ElmbData>(o2::emcal::CalibDB::getCDBPathTemperatureSensor(), metadata, ts);
  if (!mELMBdata) {
    std::cerr << "No Temperature object received from CCDB" << std::endl;
  } else {
    printElmbData(mELMBdata->getData());
  }

  return;
}

void printElmbData(std::vector<Sensor_t> data)
{

  std::cout << "Temperature sensor data\n";
  std::cout << "sensor# index | Npoints | mean | rms | min | max\n";
  for (int i = 0; i < 180; i++) {
    if (get<0>(data[i]) < 1)
      continue;
    std::cout << "sensor# " << i << " | ";
    std::cout << get<0>(data[i]) << " | ";
    std::cout << get<1>(data[i]) << " | ";
    std::cout << get<2>(data[i]) << " | ";
    std::cout << get<3>(data[i]) << " | ";
    std::cout << get<4>(data[i]) << std::endl;
  }
}
