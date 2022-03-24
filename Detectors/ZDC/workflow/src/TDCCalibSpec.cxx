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

/// @file   TDCCalibSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCCalib/InterCalib.h"
#include "ZDCWorkflow/TDCCalibSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

TDCCalibSpec::TDCCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

TDCCalibSpec::TDCCalibSpec(const int verbosity)
  : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void TDCCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
}

void TDCCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ZDCEnergyParam*>("energycalib");
  pc.inputs().get<o2::zdc::ZDCTowerParam*>("towercalib");
  pc.inputs().get<o2::zdc::InterCalibConfig*>("intercalibconfig");
}

void TDCCalibSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  if (!mInitialized) {
    mInitialized = true;
    std::string loadedConfFiles = "Loaded ZDC configuration files:";

    // TDC centering
    auto tdcParam = pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccalib");
    if (!tdcParam) {
      LOG(fatal) << "Missing ZDCTDCParam calibration object";
      return;
    } else {
      loadedConfFiles += " ZDCTDCParam";
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded TDC centering ZDCTDCParam";
      tdcParam->print();
    }
  }

  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto bcrec = pc.inputs().get<gsl::span<o2::zdc::BCRecData>>("bcrec");
  auto energy = pc.inputs().get<gsl::span<o2::zdc::ZDCEnergy>>("energy");
  auto tdc = pc.inputs().get<gsl::span<o2::zdc::ZDCTDCData>>("tdc");
  auto info = pc.inputs().get<gsl::span<uint16_t>>("info");
  InterCalib work;
  work.init();
  work.process(bcrec, energy, tdc, info);

  mTimer.Stop();
}

void TDCCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "ZDC TDC calibration total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getTDCCalibSpec(const int verbosity = 0)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("bcrec", "ZDC", "BCREC", 0, Lifetime::Timeframe);
  inputs.emplace_back("energy", "ZDC", "ENERGY", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdc", "ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("info", "ZDC", "INFO", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdccalib", "ZDC", "TDCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCalib.data())));

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "zdc-tdc-calib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TDCCalibSpec>(verbosity)},
    o2::framework::Options{{"verbosity-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
