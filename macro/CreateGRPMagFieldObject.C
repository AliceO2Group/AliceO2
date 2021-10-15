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

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <ctime>
#include <chrono>
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/CcdbApi.h"
#include "CommonTypes/Units.h"

#endif

using timePoint = std::time_t;
using CcdbApi = o2::ccdb::CcdbApi;
using GRPMagField = o2::parameters::GRPMagField;
using current = o2::units::Current_t;

// Simple macro to exemplify how to fill a GRPMagField object (GRP object containing the information on the magnetic field)

// e.g.
/*
 .L CreateGRPMagFieldObject.C+
 float l3 = 30000 // Ampere
 float dipole = 6000 // Ampere
 std::time_t tStart = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
 std::time_t tEnd = (tStart + 60 * 60 * 24 * 365) * 1000; // 1 year validity, just an example
 bool isUniform = true
 int runNb = 123456
 CreateGRPMagFieldObject(l3, dipole, tStart, runNb, tEnd, isUniform)
*/

void CreateGRPMagFieldObject(current l3, current dipole, timePoint start, int run, timePoint end = -1, bool isUniform = true, std::string ccdbPath = "http://ccdb-test.cern.ch:8080")
{

  GRPMagField grp;
  grp.setL3Current(l3);
  grp.setDipoleCurrent(dipole);
  grp.setFieldUniformity(isUniform);

  CcdbApi api;
  api.init(ccdbPath);
  std::map<std::string, std::string> metadata;
  metadata["responsible"] = "DCS";
  metadata["run_number"] = std::to_string(run);
  //long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (end < 0) {
    end = (start + 60 * 60 * 10) * 1000; // start + 10h, in ms
  }
  api.storeAsTFileAny(&grp, "GLO/Config/GRPMagField", metadata, start * 1000, end); // making it 1-year valid to be sure we have something
}
