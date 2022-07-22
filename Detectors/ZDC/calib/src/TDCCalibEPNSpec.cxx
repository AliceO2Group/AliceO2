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

/// @file   TDCCalibEPNSpec.cxx
/// @brief  EPN Spec file for TDC calibration
/// @author luca.quaglia@cern.ch

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
#include "ZDCCalib/TDCCalibEPNSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

TDCCalibEPNSpec::TDCCalibEPNSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

TDCCalibEPNSpec::TDCCalibEPNSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void TDCCalibEPNSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
}

void TDCCalibEPNSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::TDCCalibConfig*>("tdccalibconfig"); //added by me
}

void TDCCalibEPNSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCALIBCONFIG", 0)) {
    auto* config = (const o2::zdc::TDCCalibConfig*)obj;
    if (mVerbosity > DbgZero) {
      LOG(info) << "Loaded TDCCalib configuration object";
      config->print();
    }
    mWorker.setTDCCalibConfig(config);
  }
}

void TDCCalibEPNSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }
  //auto config = pc.inputs().get<o2::zdc::TDCCalibConfig*>("tdccalibconfig");
  const auto ref = pc.inputs().getFirstValid(true);
  auto creationTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->creation; // approximate time in ms
  mWorker.getData().setCreationTime(creationTime);

  auto bcrec = pc.inputs().get<gsl::span<o2::zdc::BCRecData>>("bcrec");
  auto energy = pc.inputs().get<gsl::span<o2::zdc::ZDCEnergy>>("energy"); //maybe not needed for TDC configuration
  auto tdc = pc.inputs().get<gsl::span<o2::zdc::ZDCTDCData>>("tdc");
  auto info = pc.inputs().get<gsl::span<uint16_t>>("info");

  // Process reconstructed data
  mWorker.process(bcrec, energy, tdc, info);

  // Send debug histograms and intermediate calibration data
  o2::framework::Output output("ZDC", "TDCCALIBDATA", 0, Lifetime::Timeframe);
  pc.outputs().snapshot(output, mWorker.mData);
  char outputd[o2::header::gSizeDataDescriptionString];
  for (int ih = 0; ih < TDCCalibData::NTDC; ih++) {
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "TDC_1DH%d", ih);
    o2::framework::Output output("ZDC", outputd, 0, Lifetime::Timeframe);
    pc.outputs().snapshot(output, mWorker.mTDC[ih]->getBase());
  }
}

void TDCCalibEPNSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  LOGF(info, "ZDC EPN TDC calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1); //added by me
}

framework::DataProcessorSpec getTDCCalibEPNSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("bcrec", "ZDC", "BCREC", 0, Lifetime::Timeframe);
  inputs.emplace_back("energy", "ZDC", "ENERGY", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdc", "ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("info", "ZDC", "INFO", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdccalibconfig", "ZDC", "TDCCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathTDCCalibConfig.data()));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "TDCCALIBDATA", 0, Lifetime::Timeframe); //added by me
  char outputd[o2::header::gSizeDataDescriptionString];

  for (int ih = 0; ih < TDCCalibData::NTDC; ih++) {
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "TDC_1DH%d", ih);
    outputs.emplace_back("ZDC", outputd, 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "zdc-tdccalib-epn",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TDCCalibEPNSpec>()},
    o2::framework::Options{{"verbosity-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2