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
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CCDB/CcdbApi.h"

#endif

using timePoint = long;
using CcdbApi = o2::ccdb::CcdbApi;
using GRPLHCIFData = o2::parameters::GRPLHCIFData;

// Simple macro to exemplify how to fill a GRPLHCIF object (GRP object containing the information from the LHC)

void CreateGRPLHCIFObject(timePoint start, int egev, int fill, const std::string& injScheme, const std::string& beamAFilling,
                          const std::string& beamCFilling = "", int A1 = 1, int A2 = 1, float crossAngle = 0.f,
                          timePoint end = -1, std::string ccdbPath = "http://ccdb-test.cern.ch:8080")
{
  GRPLHCIFData grp;
  grp.setBeamEnergyPerZWithTime(start, egev);
  grp.setFillNumberWithTime(start, fill);
  grp.setInjectionSchemeWithTime(start, injScheme);
  grp.setAtomicNumberB1WithTime(start, A1);
  grp.setAtomicNumberB2WithTime(start, A2);
  grp.setBeamAZ();
  grp.setCrossingAngleWithTime(start, crossAngle);
  o2::BunchFilling bf(beamAFilling, beamCFilling);

  CcdbApi api;
  api.init(ccdbPath);
  std::map<std::string, std::string> metadata;
  metadata["responsible"] = "LHCIF";
  if (end < 0) {
    end = (start + 60 * 60 * 15) * 1000; // start + 15h, in ms
  }
  api.storeAsTFileAny(&grp, "GLO/Config/GRPLHCIF", metadata, start * 1000, end); // making it 1-year valid to be sure we have something
}
