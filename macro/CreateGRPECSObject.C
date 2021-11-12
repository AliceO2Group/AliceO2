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
#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CCDB/CcdbApi.h"

#endif

using timePoint = std::time_t;
using DetID = o2::detectors::DetID;
using CcdbApi = o2::ccdb::CcdbApi;
using GRPECSObject = o2::parameters::GRPECSObject;

// Simple macro to exemplify how to fill a GRPECS object (GRP object containing the information that come from ECS)

// The list of detectors in the readout, read-out continuously, or in the trigger are passed as DetID::mask_it; this is a std::bitset<32>, with the following meaning per bit:
// {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "ACO", "CTP"}

// e.g.
/*
 .L CreateGRPECSObject.C+
 std::time_t tStart = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
 std::time_t tEnd = (tStart + 60 * 60 * 24 * 365) * 1000; // 1 year validity, just an example
 o2::detectors::DetID::mask_t detsRO = o2::detectors::DetID::getMask("ITS, TPC, TRD, TOF, CPV, PHS, EMC, MID, MFT, MCH, FT0, FV0, FDD, CTP");
 o2::detectors::DetID::mask_t detsContRO = o2::detectors::DetID::getMask("TOF, TPC, ITS")
 o2::detectors::DetID::mask_t detsTrig = o2::detectors::DetID::getMask("FV0")
 CreateGRPECSObject(tStart, 128, detsRO, detsContRO, detsTrig, 123456, "LHC21m", tEnd)
*/

void CreateGRPECSObject(timePoint start, uint32_t nHBPerTF, DetID::mask_t detsReadout, DetID::mask_t detsContinuousRO, DetID::mask_t detsTrigger, int run, std::string dataPeriod, timePoint end = -1, std::string ccdbPath = "http://ccdb-test.cern.ch:8080")
{

  GRPECSObject grpecs;
  grpecs.setTimeStart(start);
  grpecs.setNHBFPerTF(nHBPerTF);
  grpecs.setDetsReadOut(detsReadout);
  grpecs.setDetsContinuousReadOut(detsContinuousRO);
  grpecs.setDetsTrigger(detsTrigger);
  grpecs.setRun(run);
  grpecs.setDataPeriod(dataPeriod);

  CcdbApi api;
  api.init(ccdbPath);
  std::map<std::string, std::string> metadata;
  metadata["responsible"] = "ECS";
  metadata["run_number"] = std::to_string(run);
  //long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (end < 0) {
    end = (start + 60 * 60 * 10) * 1000; // start + 10h, in ms
  }
  api.storeAsTFileAny(&grpecs, "GLO/Config/GRPECS", metadata, start * 1000, end); // making it 1-year valid to be sure we have something
}
