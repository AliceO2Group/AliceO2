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

/// @file   InterCalibEPNSpec.cxx
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
#include "Framework/DataRefUtils.h"
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
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalibEPNSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

InterCalibEPNSpec::InterCalibEPNSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

InterCalibEPNSpec::InterCalibEPNSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void InterCalibEPNSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
}

void InterCalibEPNSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::InterCalibConfig*>("intercalibconfig");
}

void InterCalibEPNSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  if (!mInitialized) {
    mInitialized = true;
    std::string loadedConfFiles = "Loaded ZDC configuration files:";
    // InterCalib configuration
    auto config = pc.inputs().get<o2::zdc::InterCalibConfig*>("intercalibconfig");
    loadedConfFiles += " InterCalibConfig";
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded InterCalib configuration object";
      config->print();
    }
    mWorker.setInterCalibConfig(config.get());
    LOG(info) << loadedConfFiles;
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }

  const auto ref = pc.inputs().getFirstValid(true);
  auto creationTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->creation; // approximate time in ms
  mWorker.getData().setCreationTime(creationTime);

  auto bcrec = pc.inputs().get<gsl::span<o2::zdc::BCRecData>>("bcrec");
  auto energy = pc.inputs().get<gsl::span<o2::zdc::ZDCEnergy>>("energy");
  auto tdc = pc.inputs().get<gsl::span<o2::zdc::ZDCTDCData>>("tdc");
  auto info = pc.inputs().get<gsl::span<uint16_t>>("info");

  // Process reconstructed data
  mWorker.process(bcrec, energy, tdc, info);

  // Send intermediate calibration data and debug histograms
  o2::framework::Output output("ZDC", "INTERCALIBDATA", 0, Lifetime::Timeframe);
  pc.outputs().snapshot(output, mWorker.mData);
  char outputd[o2::header::gSizeDataDescriptionString];
  for (int ih = 0; ih < (2 * InterCalibData::NH); ih++) {
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "INTER_1DH%d", ih);
    o2::framework::Output output("ZDC", outputd, 0, Lifetime::Timeframe);
    pc.outputs().snapshot(output, mWorker.mH[ih]->getBase());
  }
  for (int ih = 0; ih < InterCalibData::NH; ih++) {
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "INTER_2DH%d", ih);
    o2::framework::Output output("ZDC", outputd, 0, Lifetime::Timeframe);
    pc.outputs().snapshot(output, mWorker.mC[ih]->getBase());
  }
}

void InterCalibEPNSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  LOGF(info, "ZDC EPN Intercalibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getInterCalibEPNSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("bcrec", "ZDC", "BCREC", 0, Lifetime::Timeframe);
  inputs.emplace_back("energy", "ZDC", "ENERGY", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdc", "ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("info", "ZDC", "INFO", 0, Lifetime::Timeframe);
  inputs.emplace_back("intercalibconfig", "ZDC", "INTERCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathInterCalibConfig.data())));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "INTERCALIBDATA", 0, Lifetime::Timeframe);
  char outputd[o2::header::gSizeDataDescriptionString];
  for (int ih = 0; ih < (2 * InterCalibData::NH); ih++) {
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "INTER_1DH%d", ih);
    outputs.emplace_back("ZDC", outputd, 0, Lifetime::Timeframe);
  }
  for (int ih = 0; ih < InterCalibData::NH; ih++) {
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "INTER_2DH%d", ih);
    outputs.emplace_back("ZDC", outputd, 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "zdc-intercalib-epn",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<InterCalibEPNSpec>()},
    o2::framework::Options{{"verbosity-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
