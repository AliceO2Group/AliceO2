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
// STL headers
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

// O2 header
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbObjectInfo.h"
#include "Framework/Logger.h"
#include "TRDPID/LQND.h"

// ROOT header
#include <TFile.h>
#include <TGraph.h>

#endif

constexpr int fileError{-42};

o2::ccdb::CcdbApi ccdb;
std::map<std::string, std::string> metadata{{"UploadedBy", "Felix Schlepper"}, {"EMail", "felix.schlepper@cern.ch"}, {"default", "false"}, {"Created", "1"}};

/// Upload an ONNX model to the ccdb.
/// This reads the file as a binary file and stores it as such.
int ccdbONNXUpload(std::string inFileName, std::string ccdbPath, uint64_t timeStampStart, uint64_t timeStampEnd)
{
  metadata["Description"] = "ONNX model for TRD PID";

  std::vector<unsigned char> input;
  std::ifstream inFile(inFileName, std::ios::binary);
  if (!inFile.is_open()) {
    LOG(error) << "Could not open file (" << inFileName << "!)";
    return fileError;
  }
  inFile.seekg(0, std::ios_base::end);
  auto length = inFile.tellg();
  inFile.seekg(0, std::ios_base::beg);
  input.resize(static_cast<size_t>(length));
  inFile.read(reinterpret_cast<char*>(input.data()), length);
  auto success = !inFile.fail() && length == inFile.gcount();
  if (!success) {
    LOG(error) << "Could not read file (" << inFileName << "!)";
    return fileError;
  }
  inFile.close();

  return ccdb.storeAsBinaryFile(reinterpret_cast<char*>(input.data()), length, inFileName, "ONNX Model | file read as binary string", ccdbPath, metadata, timeStampStart, timeStampEnd);
}

/// Upload LQND LUTs as std::vector<TGraph>
template <int dim>
int ccdbLQNDUpload(std::string inFileName, std::string ccdbPath, uint64_t timeStampStart, uint64_t timeStampEnd)
{
  metadata["Description"] = Form("LQ%dD model for TRD PID", dim);

  std::unique_ptr<TFile> inFile(TFile::Open(inFileName.c_str()));
  if (!inFile || inFile->IsZombie()) {
    LOG(error) << "Could not open file (" << inFileName << "!)";
    return fileError;
  }
  // copy vector from file
  auto luts = *(inFile->Get<o2::trd::detail::LUT<dim>>("luts"));

  return ccdb.storeAsTFileAny(&luts, ccdbPath, metadata, timeStampStart, timeStampEnd);
}

void ccdbPIDUpload(std::string inFileName, std::string ccdbPath, bool testCCDB = true, bool ml = false, int dim = 1, uint64_t timeStampStart = 1577833200000UL /* 1.1.2020 */, uint64_t timeStampEnd = 2208985200000UL /* 1.1.2040 */)
{
  if (testCCDB) {
    ccdb.init("http://ccdb-test.cern.ch:8080");
  } else {
    ccdb.init("http://alice-ccdb.cern.ch");
  }

  int res{0};
  if (ml) {
    res = ccdbONNXUpload(inFileName, ccdbPath, timeStampStart, timeStampEnd);
  } else {
    if (dim == 1) {
      res = ccdbLQNDUpload<1>(inFileName, ccdbPath, timeStampStart, timeStampEnd);
    } else {
      res = ccdbLQNDUpload<3>(inFileName, ccdbPath, timeStampStart, timeStampEnd);
    }
  }

  if (res == 0) {
    LOG(info) << "Upload: OKAY";
  } else if (res == -1) {
    LOG(error) << "object bigger than maxSize";
  } else if (res == -2) {
    LOG(error) << "curl initialization error";
  } else if (res == fileError) {
    LOG(error) << "File reading error";
  } else {
    LOG(error) << "see curl error codes: " << res;
  }
}
