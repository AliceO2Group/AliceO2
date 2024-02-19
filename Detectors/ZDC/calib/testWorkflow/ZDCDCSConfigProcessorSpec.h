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

#ifndef O2_ZDC_DCSCONFIGPROCESSOR_H
#define O2_ZDC_DCSCONFIGPROCESSOR_H

/// @file   ZDCDCSConfigProcessorSpec.h
/// @brief  ZDC Processor for DCS Configurations

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
namespace zdc
{

class ZDCDCSConfigProcessor : public o2::framework::Task
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
    auto timer = std::chrono::duration_cast<std::chrono::milliseconds>(HighResClock::now().time_since_epoch()).count();
    LOG(info) << "got input file " << configFileName << " of size " << configBuff.size();
    // auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().getFirstValid(true).header)->startTime;
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

    /*using clbUtils = o2::calibration::Utils;
    const auto& payload = mCalibrator->getNoiseMap();
    std::map<std::string, std::string> md;
    md.emplace("created by", "dpl");
    o2::ccdb::CcdbObjectInfo info("ZDC/Calib/Mapping", "ZDCMapping", "mapping.root", md, tf, INFINITE_TF);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();*/
  }
  //________________________________________________________________

  bool mVerbose = false; // to enable verbose mode
};                       // end class
} // namespace zdc

namespace framework
{

DataProcessorSpec getZDCDCSConfigProcessorSpec()
{

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "zdc-dcs-config-processor",
    Inputs{{"inputConfig", o2::header::gDataOriginZDC, "DCS_CONFIG_FILE", Lifetime::Sporadic},
           {"inputConfigFileName", o2::header::gDataOriginZDC, "DCS_CONFIG_NAME", Lifetime::Sporadic}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::zdc::ZDCDCSConfigProcessor>()},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2

#endif
