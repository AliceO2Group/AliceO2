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

// macro to read the TOF DCS information from CCDB
// default ts is very big: Saturday, November 20, 2286 5:46:39 PM

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "MFTCondition/DCSNameResolver.h"
#include "MFTCondition/MFTDCSProcessor.h"
#include <string>
#include <unordered_map>
#include <chrono>
#include <bitset>
#endif

void readMFTDCSentries(long ts = 9999999999000, std::string ccdb_path = o2::base::NameConf::getCCDBServer())
{

  o2::mft::DCSNameResolver namer;
  namer.init();

  o2::ccdb::CcdbApi api;
  api.init(ccdb_path); // or http://ccdb-test.cern.ch:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::unordered_map<o2::dcs::DataPointIdentifier, o2::mft::MFTDCSinfo>* m = api.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, o2::mft::MFTDCSinfo>>("MFT/Condition/DCSDPs", metadata, ts);
  std::cout << "size of map = " << m->size() << std::endl;
  for (auto& i : *m) {
    std::cout << " PID  =  " << i.first.get_alias() << " (alias)   ======>   " << namer.getFullName(string(i.first.get_alias())) << " (full name)" << std::endl;
    i.second.print();
  }

  return;
}
