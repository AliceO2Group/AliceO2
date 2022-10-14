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
#include "DataFormatsParameters/GRPMagField.h"
#include "GRPCalibration/GRPDCSDPsProcessor.h"

#include <string>
#include <unordered_map>
#include <chrono>
#endif

// ccdb can be
// http://localhost:8080 (as by default)
// http://ccdb-test.cern.ch:8080 (test CCDB)
// https://alice-ccdb.cern.ch (production CCDB)
// http://o2-ccdb.internal (production CCDB, only from FLPs and EPNs)

void readGRPCCDB(long ts = 9999999999000, const char* ccdb = "http://localhost:8080")
{
  o2::ccdb::CcdbApi api;
  api.init(ccdb); // or http://ccdb-test.cern.ch:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::unordered_map<o2::dcs::DataPointIdentifier, std::string>* m = api.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, std::string>>("GRP/Config/DCSDPconfig", metadata, ts);
  std::cout << "size of map = " << m->size() << std::endl;

  for (auto& i : *m) {
    std::cout << "id = " << i.first << ", " << i.second << std::endl;
  }

  o2::parameters::GRPMagField* magField = api.retrieveFromTFileAny<o2::parameters::GRPMagField>("GLO/Calib/GRPMagField", metadata, ts);
  std::cout << "\n*** Magnetic field:" << std::endl;
  magField->print();

  o2::grp::GRPLHCInfo* lhcinfo = api.retrieveFromTFileAny<o2::grp::GRPLHCInfo>("GLO/Calib/LHCIF", metadata, ts);
  std::cout << "\n*** LHC info:" << std::endl;
  lhcinfo->print();

  o2::grp::GRPCollimators* collim = api.retrieveFromTFileAny<o2::grp::GRPCollimators>("GLO/Calib/Collimators", metadata, ts);
  std::cout << "\n*** Collimators:" << std::endl;
  collim->print();

  o2::grp::GRPEnvVariables* envVar = api.retrieveFromTFileAny<o2::grp::GRPEnvVariables>("GLO/Calib/EnvVars", metadata, ts);
  std::cout << "\n*** Env Vars:" << std::endl;
  envVar->print();

  return;
}
