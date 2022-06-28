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

#ifndef O2_TOF_DCSCONFIGPROCESSOR_H
#define O2_TOF_DCSCONFIGPROCESSOR_H

/// @file   TOFDCSConfigProcessorSpec.h
/// @brief  TOF Processor for DCS Configurations

#include "TOFCalibration/TOFFEElightReader.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/CcdbApi.h"

#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using TFType = uint64_t;

namespace o2
{
namespace tof
{

class TOFDCSConfigProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    mVerbose = ic.options().get<bool>("use-verbose-mode");
    LOG(info) << " ************************* Verbose?" << mVerbose;
  }

  //---------------------------------------------------------

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto configBuff = pc.inputs().get<gsl::span<char>>("inputConfig");
    auto configFileName = pc.inputs().get<std::string>("inputConfigFileName");
    LOG(info) << "got input file " << configFileName << " of size " << configBuff.size();
    mTOFFEElightReader.loadFEElightConfig(configBuff);
    mTOFFEElightReader.parseFEElightConfig(mVerbose);
    //auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().getFirstValid(true).header)->startTime;
    sendOutput(pc.outputs(), configFileName);
  }

  //---------------------------------------------------------

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
  }

 private:
  void sendOutput(DataAllocator& output, std::string fileName)
  {
    // sending output to CCDB

    using clbUtils = o2::calibration::Utils;
    const auto& payload = mTOFFEElightReader.getTOFFEElightInfo();
    auto clName = o2::utils::MemFileHelper::getClassName(payload);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    std::map<std::string, std::string> md;
    md.emplace("created_by", "dpl");

    // finding start of validity and run number in filename
    int n = fileName.length();
    char fileName_char[n + 1];
    strcpy(fileName_char, fileName.c_str());
    std::vector<std::string> tokens;
    const char* delimiters = ".";
    const char* token = strtok(fileName_char, delimiters);
    std::string startOfValidityStr = "";
    std::string runNumberStr = "";
    bool foundSOV = false;
    bool foundRUN = false;
    std::string timeStampLabel = "sov";
    std::string runNumberLabel = "run";
    while (token) {
      LOG(debug) << "token = " << token;
      std::string stoken(token);
      auto idxInStringSOV = stoken.find(timeStampLabel);
      auto idxInStringRUN = stoken.find(runNumberLabel);
      if (idxInStringSOV != std::string::npos) {
        foundSOV = true;
        stoken.erase(idxInStringSOV, timeStampLabel.length());
        startOfValidityStr = stoken;
      }
      if (idxInStringRUN != std::string::npos) {
        foundRUN = true;
        stoken.erase(idxInStringRUN, runNumberLabel.length());
        runNumberStr = stoken;
      }
      token = std::strtok(nullptr, delimiters);
    }
    if (!foundSOV) {
      LOG(fatal) << "SOV not found but needed!";
    } else {
    }
    if (!foundRUN) {
      LOG(warning) << "RUN not found, will be left empty!";
    }
    LOG(debug) << "startOfValidityStr = " << startOfValidityStr << ", run = " << runNumberStr;
    long tf = std::stol(startOfValidityStr) * 1E3; // in ms
    md.emplace("runNumberFromTOF", runNumberStr);

    // creating CCDB object to be shipped
    o2::ccdb::CcdbObjectInfo info("TOF/Calib/FEELIGHT", clName, flName, md, tf, tf + o2::ccdb::CcdbObjectInfo::MONTH);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_FEELIGHT", 0}, *image.get()); // vector<char>
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_FEELIGHT", 0}, info);         // root-serialized
  }
  //________________________________________________________________

  TOFFEElightReader mTOFFEElightReader; // reader for configuration
  bool mVerbose = false;                // to enable verbose mode

}; // end class
} // namespace tof

namespace framework
{

DataProcessorSpec getTOFDCSConfigProcessorSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_FEELIGHT"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_FEELIGHT"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tof-dcs-config-processor",
    Inputs{{"inputConfig", o2::header::gDataOriginTOF, "DCS_CONFIG_FILE", Lifetime::Timeframe},
           {"inputConfigFileName", o2::header::gDataOriginTOF, "DCS_CONFIG_NAME", Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::tof::TOFDCSConfigProcessor>()},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2

#endif
