// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <chrono>

using namespace o2::framework;
using TFType = uint64_t;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

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
    LOG(INFO) << " ************************* Verbose?" << mVerbose;
  }

  //---------------------------------------------------------

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto configBuff = pc.inputs().get<gsl::span<char>>("inputConfig");
    auto configFileName = pc.inputs().get<std::string>("inputConfigFileName");
    auto timer = std::chrono::duration_cast<std::chrono::milliseconds>(HighResClock::now().time_since_epoch()).count();
    LOG(INFO) << "got input file " << configFileName << " of size " << configBuff.size();
    mTOFFEElightReader.loadFEElightConfig(configBuff);
    mTOFFEElightReader.parseFEElightConfig(mVerbose);
    //auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().getByPos(0).header)->startTime;
    sendOutput(pc.outputs(), timer);
  }

  //---------------------------------------------------------

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
  }

 private:
  void sendOutput(DataAllocator& output, long tf)
  {
    // sending output to CCDB

    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;

    using clbUtils = o2::calibration::Utils;
    const auto& payload = mTOFFEElightReader.getTOFFEElightInfo();
    auto clName = o2::utils::MemFileHelper::getClassName(payload);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    std::map<std::string, std::string> md;
    md.emplace("created by", "dpl");
    o2::ccdb::CcdbObjectInfo info("TOF/FEELIGHT", clName, flName, md, tf, INFINITE_TF);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
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
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_FEELIGHT"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_FEELIGHT"});

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
