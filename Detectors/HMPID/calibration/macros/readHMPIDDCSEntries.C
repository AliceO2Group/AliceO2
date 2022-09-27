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

// macro to read the HMP DCS information from CCDB
// Reads the vectors of TF1s and saves them in specified folder
// default ts is very big: Saturday, November 20, 2286 5:46:39 PM

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "HMPIDCalibration/HMPIDDCSProcessor.h"

#include "TCanvas.h"
#include "TF1.h"

#include <string>
#include <unordered_map>
#include <chrono>
#include <bitset>
#endif
// in root folder;
// to start listening on localhost: java -jar local.jar
void readHMPIDDCSEntries(long ts = 9999999999000, const char* ccdb = "localhost:8080")
{

  o2::ccdb::CcdbApi api;
  api.init(ccdb); // or http://ccdb-test.cern.ch:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::vector<TF1, std::allocator<TF1>>* mChargeCut = api.retrieveFromTFileAny<std::vector<TF1, std::allocator<TF1>>>("HMP/Calib/ChargeCut", metadata, ts);

  int cnt = 0;
  for (TF1& tf1 : *mChargeCut) {
    TCanvas* cChargeCut = new TCanvas(Form("ChargeCut number %i", cnt), Form("ChargeCut number %i", cnt), 1200, 400);
    tf1.Draw();
    cChargeCut->SaveAs(Form("/root/hmpidTest/img/HV/ChargeCut number %i.png", cnt));
    cnt++;
  }

  std::vector<TF1, std::allocator<TF1>>* mRefIndex = api.retrieveFromTFileAny<std::vector<TF1, std::allocator<TF1>>>("HMP/Calib/RefIndex", metadata, ts);
  // std::cout << "size of mRefIndex = " << mRefIndex->size() << std::endl;

  cnt = 0;
  for (TF1& tf1 : *mRefIndex) {
    TCanvas* cRefIndex = new TCanvas(Form("RefIndex number %i", cnt), Form("RefIndex number %i", cnt), 1200, 400);
    tf1.Draw();
    cRefIndex->SaveAs(Form("/root/hmpidTest/img/HV/RefIndex number %i.png", cnt));
    cnt++;
  }

  return;
}
