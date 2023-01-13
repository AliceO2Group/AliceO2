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
// ROOT header
#include <TFile.h>
// O2 header
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbObjectInfo.h"

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>

#endif

/// Upload an ONNX model to the ccdb.
/// This reads the file as a binary file and stores it as such.
void ccdbModelUpload(std::string inFileName, std::string ccdbPath = "TRD_test/PID/xgb")
{

  o2::ccdb::CcdbApi ccdb;
  // ccdb.init("http://alice-ccdb.cern.ch");
  // ccdb.init("http://localhost:8080");
  ccdb.init("http://ccdb-test.cern.ch:8080");
  // ccdb.init("http://o2-ccdb.internal");
  std::map<std::string, std::string> metadata;
  metadata["UploadedBy"] = "Felix Schlepper";
  metadata["EMail"] = "felix.schlepper@cern.ch";
  metadata["Description"] = "ONNX model for TRD PID";
  metadata["default"] = "false"; // tag default objects
  metadata["Created"] = "1";     // tag default objects

  std::vector<unsigned char> input;
  std::ifstream inFile(inFileName, std::ios::binary);
  if (!inFile.is_open()) {
    std::cout << "Could not load file!" << std::endl;
    return;
  }
  inFile.seekg(0, std::ios_base::end);
  auto length = inFile.tellg();
  inFile.seekg(0, std::ios_base::beg);
  input.resize(static_cast<size_t>(length));
  inFile.read(reinterpret_cast<char*>(input.data()), length);
  auto success = !inFile.fail() && length == inFile.gcount();
  if (!success) {
    std::cout << "File loading went wrong!" << std::endl;
    return;
  }
  inFile.close();

  // for default objects:
  // long timeStampStart = 0;
  uint64_t timeStampStart = 1577833200000UL; // 1.1.2020
  uint64_t timeStampEnd = 2208985200000UL;   // 1.1.2040

  auto res = ccdb.storeAsBinaryFile(reinterpret_cast<char*>(input.data()), length, inFileName, "ONNX Model", ccdbPath, metadata, timeStampStart, timeStampEnd);

  if (res == 0) {
    std::cout << "OK" << std::endl;
  } else if (res == -1) {
    std::cout << "ERROR: object bigger than maxSize" << std::endl;
  } else if (res == -2) {
    std::cout << "ERROR: curl initialization error" << std::endl;
  } else {
    std::cout << "ERROR: see curl error codes: " << res << std::endl;
  }
  return;
}
