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

#include <vector>
#include <string>
#include <map>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "MIDEfficiency/ChamberEfficiency.h"

int makeMIDEfficiencyCCDBEntry(const char* url = "http://localhost:8080")
{
  auto chamberEff = o2::mid::createDefaultChamberEfficiency();
  auto data = chamberEff.getCountersAsVector();
  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  api.storeAsTFileAny(&data, "MID/Calib/ChamberEfficiency", md, 1, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);

  return 0;
}
