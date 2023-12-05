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

/// @file   WaveformCalibEPNSpec.cxx
/// @brief  ZDC waveform calibration
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
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
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCCalib/WaveformCalibData.h"
#include "ZDCCalib/WaveformCalibEPNSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

WaveformCalibEPNSpec::WaveformCalibEPNSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

WaveformCalibEPNSpec::WaveformCalibEPNSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void WaveformCalibEPNSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
}

void WaveformCalibEPNSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::WaveformCalibConfig*>("wavecalibconfig");
}

void WaveformCalibEPNSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "WAVECALIBCONFIG", 0)) {
    // InterCalib configuration
    auto* config = (const o2::zdc::WaveformCalibConfig*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setConfig(config);
  }
}

void WaveformCalibEPNSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  } else {
    mWorker.clear();
  }

  auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
  WaveformCalibData& data = mWorker.getData();

  data.setCreationTime(creationTime);

  auto bcrec = pc.inputs().get<gsl::span<o2::zdc::BCRecData>>("bcrec");
  auto energy = pc.inputs().get<gsl::span<o2::zdc::ZDCEnergy>>("energy");
  auto tdc = pc.inputs().get<gsl::span<o2::zdc::ZDCTDCData>>("tdc");
  auto info = pc.inputs().get<gsl::span<uint16_t>>("info");
  auto wave = pc.inputs().get<gsl::span<o2::zdc::ZDCWaveform>>("wave");

  // Process reconstructed data
  mWorker.process(bcrec, energy, tdc, info, wave);

  // Send intermediate calibration data
  o2::framework::Output output("ZDC", "WAVECALIBDATA", 0);
  pc.outputs().snapshot(output, mWorker.mData);
}

void WaveformCalibEPNSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  LOGF(info, "ZDC EPN Waveform calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getWaveformCalibEPNSpec()
{
  using device = o2::zdc::WaveformCalibEPNSpec;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("bcrec", "ZDC", "BCREC", 0, Lifetime::Timeframe);
  inputs.emplace_back("energy", "ZDC", "ENERGY", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdc", "ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("info", "ZDC", "INFO", 0, Lifetime::Timeframe);
  inputs.emplace_back("wave", "ZDC", "WAVE", 0, Lifetime::Timeframe);
  inputs.emplace_back("wavecalibconfig", "ZDC", "WAVECALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathWaveformCalibConfig.data()));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "WAVECALIBDATA", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "zdc-waveformcalib-epn",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
