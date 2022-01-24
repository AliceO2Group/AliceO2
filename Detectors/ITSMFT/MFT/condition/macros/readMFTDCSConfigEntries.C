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
#include "MFTCondition/DCSConfigInfo.h"
#include "MFTCondition/DCSConfigUtils.h"
#include <string>
#include <unordered_map>
#include <chrono>
#include <bitset>
#endif

void readMFTDCSConfigEntries(long ts = 9999999999000, std::string ccdb_path = o2::base::NameConf::getCCDBServer())
{

  o2::ccdb::CcdbApi api;
  api.init(ccdb_path); // or http://ccdb-test.cern.ch:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::vector<o2::mft::DCSConfigInfo>* obj = api.retrieveFromTFileAny<std::vector<o2::mft::DCSConfigInfo>>("MFT/Config/Params", metadata, ts);

  std::cout << "Stored in CCDB server (o2::mft::DCSConfigInfo)" << endl;
  for (auto& i : *obj) {

    int data, add, type;
    std::string ver;

    data = i.getData();
    type = i.getType();
    add = i.getAdd();
    ver = i.getVersion();

    std::cout << "   Parameter type = " << type << " (RU=0, ALPIDE=1),   Address (decimal) = " << add << ",   Data =  " << data << ",   Config ver. = " << ver << std::endl;
  }

  std::cout << "Human-friendly format (o2::mft::DCSConfigInfo + o2::mft::DCSConfigUtils)" << endl;
  for (auto& i : *obj) {

    auto utils = std::make_unique<o2::mft::DCSConfigUtils>();
    utils->init(i);

    int data, add, type;
    std::string ver;
    std::string typestr, name;

    data = i.getData();
    ver = i.getVersion();
    typestr = utils->getTypeStr();
    name = utils->getName();
    std::cout << "   Parameter type = " << typestr << ",   Name = " << name << ",   Data =  " << data << ",   Config ver. = " << ver << std::endl;
  }

  return;
}
