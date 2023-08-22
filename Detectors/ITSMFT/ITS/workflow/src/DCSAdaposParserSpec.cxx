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

/// @file   DCSAdaposParserSpec.cxx

#include "ITSWorkflow/DCSAdaposParserSpec.h"

namespace o2
{
namespace its
{

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSDCSAdaposParser::ITSDCSAdaposParser()
{
  this->mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSDCSAdaposParser");
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSAdaposParser::init(InitContext& ic)
{
  LOGF(info, "ITSDCSAdaposParser init...", mSelfName);

  this->mCcdbUrl = ic.options().get<std::string>("ccdb-url");

  this->mVerboseOutput = ic.options().get<bool>("verbose");

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get Data from ADAPOS and prepare CCDB objects
void ITSDCSAdaposParser::run(ProcessingContext& pc)
{

  // Retrieve adapos data and process them
  auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
  process(dps);

  if (doStrobeUpload && dps.size() > 0) {
    // upload to ccdb
    pushToCCDB(pc);
  }

  if (mTF % 50 == 0) { // clean memory after some TFs
    mDPstrobe.clear();
  }

  mTF++;

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Function to process DPs
void ITSDCSAdaposParser::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerboseOutput) {
    LOG(info) << "\n\n\nProcessing new TF\n-----------------";
  }

  // Process all DPs, one by one
  for (const auto& it : dps) {
    processDP(it);
  }

  /**************************************
     decide whether to upload or not the object to ccdb. The logic for strobe length is:
     - if the strobe length it's the first value filled in mDPstrobe, then upload it to CCDB (only at the first call of the wf)
     - if the strobe length it's the second value filled, compare it with the previous one: if different, upload the most recent
     - the logic continue... memory is cleaned every 50 TFs (random number of TFs).
  ***************************************/
  auto mapel = mDPstrobe.begin();
  if (!mDPstrobe.size()) {
    doStrobeUpload = false;
    return;
  }

  if (mapel->second.size() == 1 && isFirstCheck) {
    mStrobeToUpload = mapel->second[0].payload_pt1;
    doStrobeUpload = true;
    isFirstCheck = false;
  } else if (mapel->second.size() == 1 && mapel->second[0].payload_pt1 != mStrobeToUpload && !isFirstCheck) { // in case the new value to be uploaded is the first on the list (we erase memory every 50 TFs)
    mStrobeToUpload = mapel->second[0].payload_pt1;
    doStrobeUpload = true;
  } else if (mapel->second.size() > 1) {
    if (mapel->second.rbegin()[0].payload_pt1 == mapel->second.rbegin()[1].payload_pt1) {
      doStrobeUpload = false;
    } else {
      mStrobeToUpload = mapel->second.rbegin()[0].payload_pt1; // take the last
      doStrobeUpload = true;
    }
  } else {
    doStrobeUpload = false;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Process single DPs
void ITSDCSAdaposParser::processDP(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  auto& val = dpcom.data;
  auto flags = val.get_flags();

  if (mVerboseOutput) {
    LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<int>(dpcom);
  }

  if (o2::dcs::getValue<int>(dpcom) < 190) {
    if (mVerboseOutput) {
      LOG(info) << "Value is < 190 BCs, skipping it";
    }
    return;
  }
  // Checking if the DP is updated with time stamp information
  auto etime = val.get_epoch_time();

  if (o2::dcs::getValue<int>(dpcom) > 189) { // Discard strobe length lower than this: thr scan
    mDPstrobe[dpid].push_back(val);
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSDCSAdaposParser::pushToCCDB(ProcessingContext& pc)
{
  // Timestamps for CCDB entry
  long tstart = 0, tend = 0;
  // retireve run start/stop times from CCDB
  o2::ccdb::CcdbApi api;
  api.init("http://alice-ccdb.cern.ch");
  // Initialize empty metadata object for search
  std::map<std::string, std::string> metadata;

  tstart = o2::ccdb::getCurrentTimestamp();
  tend = tstart + 365L * 2 * 24 * 3600 * 1000; // valid two years by default

  // Create metadata for database object
  metadata = {{"comment", "uploaded by flp199 (ADAPOS data)"}};

  std::string path("ITS/Config/AlpideParam");

  std::string filename = "o2-itsmft-DPLAlpideParam<0>_" + std::to_string(tstart) + ".root";
  o2::ccdb::CcdbObjectInfo info(path, "dplalpideparam", filename, metadata, tstart, tend);
  // Define the dpl alpide param and set the strobe length to ship
  auto& dplAlpideParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  dplAlpideParams.updateFromString(fmt::format("ITSAlpideParam.roFrameLengthInBC = {}", (int)mStrobeToUpload + 8)); // +8 is because the strobe length of ALPIDE (sent via ADAPOS) is 200ns shorter than the external trigger strobe length.
  auto class_name = o2::utils::MemFileHelper::getClassName(dplAlpideParams);

  auto image = o2::ccdb::CcdbApi::createObjectImage(&dplAlpideParams, &info);
  info.setFileName(filename);

  // Send to ccdb-populator wf
  LOG(info) << "Class Name: " << class_name << " | File Name: " << filename
            << "\nSending to ccdb-populator the object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << " bytes, valid for "
            << info.getStartValidityTimestamp() << " : "
            << info.getEndValidityTimestamp();

  if (mCcdbUrl.empty()) {

    pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITSALPIDEPARAM", 0}, *image);
    pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITSALPIDEPARAM", 0}, info);

  } else { // if url is specified, send object to ccdb from THIS wf

    LOG(info) << "Sending object " << info.getFileName() << " to " << mCcdbUrl << "/browse/"
              << info.getPath() << " from the ITS ADAPOS parser workflow";
    o2::ccdb::CcdbApi mApi;
    mApi.init(mCcdbUrl);
    mApi.storeAsBinaryFile(
      &image->at(0), image->size(), info.getFileName(), info.getObjectType(), info.getPath(),
      info.getMetaData(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSDCSAdaposParserSpec()
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "ITS", "ITSDATAPOINTS");

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITSALPIDEPARAM"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITSALPIDEPARAM"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "its-adapos-parser",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::its::ITSDCSAdaposParser>()},
    Options{
      {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
      {"ccdb-url", VariantType::String, "", {"CCDB url, default is empty (i.e. send output to CCDB populator workflow)"}}}};
}
} // namespace its
} // namespace o2
