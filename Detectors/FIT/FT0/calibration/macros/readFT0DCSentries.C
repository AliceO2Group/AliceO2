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

// macro to read the FT0 DCS information from CCDB
// default ts is very big: Saturday, November 20, 2286 5:46:39 PM

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DataFormatsFIT/DCSDPValues.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "FT0Calibration/FT0DCSProcessor.h"

#include <string>
#include <unordered_map>
#include <chrono>
#endif

void readFT0DCSentries(long ts = 9999999999000, const char* ccdb = "http://localhost:8080", const bool printEmpty = false)
{

  o2::ccdb::CcdbApi api;
  api.init(ccdb); // or http://ccdb-test.cern.ch:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>* m = api.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, o2::fit::DCSDPValues>>("FT0/Calib/DCSDPs", metadata, ts);
  std::cout << "size of map = " << m->size() << std::endl;
  if (!printEmpty) {
    std::cout << "Not printing DPs with no values" << std::endl;
  }

  for (auto& i : *m) {
    if (i.second.values.empty() && !printEmpty) {
      continue;
    }
    LOG(info) << "DPID = " << i.first;
    i.second.print();
  }

  return;
}
