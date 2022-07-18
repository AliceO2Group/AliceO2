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

// macro showing how to read a CCDB object created from DCS data points

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include "TRDCalibration/DCSProcessor.h"

#include <string>
#include <unordered_map>
#include <chrono>
#include <bitset>
#endif

void readTRDDCSentries(std::string ccdb = "http://localhost:8080", long ts = -1)
{

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setURL(ccdb.c_str()); // comment out this line to read from production CCDB instead of a local one, or adapt ccdb string
  if (ts < 0) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }
  ccdbmgr.setTimestamp(ts);

  // first, load the configuration of the DPs
  auto dpConfigTrd = ccdbmgr.get<std::unordered_map<o2::dcs::DataPointIdentifier, std::string>>("TRD/Config/DCSDPconfig");
  std::cout << "Printing first all the DPs which are configured:" << std::endl;
  for (const auto& entry : *dpConfigTrd) {
    std::cout << "id = " << entry.first << std::endl;
    std::cout << "string = " << entry.second << std::endl;
  }
  std::cout << std::endl;

  // now, access the actual calibration object from CCDB
  auto cal = ccdbmgr.get<unordered_map<o2::dcs::DataPointIdentifier, o2::trd::TRDDCSMinMaxMeanInfo>>("TRD/Calib/DCSDPsGas");

  std::cout << "Printing a single object from the map (trd_gasCO2):" << std::endl;
  o2::dcs::DataPointIdentifier dpid; // used as key to access the map
  o2::dcs::DataPointIdentifier::FILL(dpid, "trd_gasCO2", o2::dcs::DeliveryType::DPVAL_DOUBLE);
  auto obj = cal->at(dpid);
  obj.print();
  std::cout << std::endl;

  std::cout << "And lastly print all objects from the map, together with their DataPointIdentifier:" << std::endl;
  for (const auto& entry : *cal) {
    std::cout << entry.first << std::endl;
    entry.second.print();
  }
  std::cout << std::endl;

  return;
}
