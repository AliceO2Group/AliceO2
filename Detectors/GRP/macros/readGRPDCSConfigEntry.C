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
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <unordered_map>
#include <chrono>

using DPID = o2::dcs::DataPointIdentifier;

int readGRPDCSConfigEntry(const char* fileName)
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for GRP with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;

  TFile* f = new TFile(fileName);
  auto* dcsConfig = (std::unordered_map<DPID, std::string>*)f->Get("ccdb_object");
  std::cout << "DCS config for GRP is " << std::endl;
  for (auto const& el : *dcsConfig) {
    std::cout << el.first << ", " << el.second << std::endl;
  }

  return 0;
}
