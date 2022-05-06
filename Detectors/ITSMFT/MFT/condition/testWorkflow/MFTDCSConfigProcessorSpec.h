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

#ifndef O2_MFT_DCSCONFIGPROCESSOR_H
#define O2_MFT_DCSCONFIGPROCESSOR_H

/// @file   MFTDCSConfigProcessorSpec.h
/// @brief  MFT Processor for DCS Configurations

#include "MFTCondition/DCSConfigReader.h"

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

#include <chrono>

using namespace o2::framework;

using TFType = uint64_t;
using HighResClock = std::chrono::high_resolution_clock;

namespace o2
{
namespace mft
{

class MFTDCSConfigProcessor : public o2::framework::Task
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
    auto configBuff = pc.inputs().get<gsl::span<char>>("confFile");
    auto configFileName = pc.inputs().get<std::string>("confFileName");

    LOG(info) << "got input file " << configFileName << " of size " << configBuff.size();

    mReader.init(mVerbose);
    mReader.loadConfig(configBuff);

    sendOutput(pc.outputs());

    mReader.clear();
  }

 private:
  void sendOutput(DataAllocator& output)
  {

    auto tf = std::chrono::duration_cast<std::chrono::milliseconds>(HighResClock::now().time_since_epoch()).count();

    using clbUtils = o2::calibration::Utils;

    const auto& payload = mReader.getConfigInfo();

    auto clName = o2::utils::MemFileHelper::getClassName(payload);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

    std::map<std::string, std::string> md;
    md.emplace("created_by", "dpl");

    o2::ccdb::CcdbObjectInfo info("MFT/Config/Params", clName, flName, md, tf, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);

    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "DCS_CONFIG_FILE", 0}, *image.get()); // vector<char>
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "DCS_CONFIG_FILE", 0}, info);         // root-serialized
  }
  //________________________________________________________________

  DCSConfigReader mReader;
  bool mVerbose = false; // to enable verbose mode

}; // end class
} // namespace mft

namespace framework
{

DataProcessorSpec getMFTDCSConfigProcessorSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "DCS_CONFIG_FILE"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "DCS_CONFIG_FILE"}, Lifetime::Sporadic);

  std::string procName = "mft-dcs-config";

  return DataProcessorSpec{
    procName,
    Inputs{{"confFile", ConcreteDataTypeMatcher{"MFT", "DCS_CONFIG_FILE"}, Lifetime::Sporadic},
           {"confFileName", ConcreteDataTypeMatcher{"MFT", "DCS_CONFIG_NAME"}, Lifetime::Sporadic}},
    Outputs{outputs},
    AlgorithmSpec{adaptFromTask<o2::mft::MFTDCSConfigProcessor>()},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2

#endif
