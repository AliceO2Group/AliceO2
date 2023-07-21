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

/// @file   NoiseCalibEPNSpec.cxx
/// @brief  ZDC baseline calibration
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
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCCalib/NoiseCalibData.h"
#include "ZDCCalib/NoiseCalibEPNSpec.h"
#include "ZDCCalib/CalibParamZDC.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

NoiseCalibEPNSpec::NoiseCalibEPNSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

NoiseCalibEPNSpec::NoiseCalibEPNSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void NoiseCalibEPNSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
  const auto& opt = CalibParamZDC::Instance();
  mModTF = opt.modTF;
  if (mVerbosity >= DbgZero) {
    LOG(info) << "Sending calibration data to aggregator every mModTF = " << mModTF << " TF";
  }
}

void NoiseCalibEPNSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
}

void NoiseCalibEPNSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "MODULECONFIG", 0)) {
    auto* config = (const o2::zdc::ModuleConfig*)obj;
    if (mVerbosity > DbgMinimal) {
      LOG(info) << "Loaded ModuleConfig configuration object";
      config->print();
    }
    mWorker.setModuleConfig(config);
  }
}

void NoiseCalibEPNSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }

  auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
  mWorker.getData().mergeCreationTime(creationTime);
  mWorker.getDataSum().mergeCreationTime(creationTime);

  auto trig = pc.inputs().get<gsl::span<o2::zdc::BCData>>("trig");
  auto chan = pc.inputs().get<gsl::span<o2::zdc::ChannelData>>("chan");
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");

  // Process digits
  mWorker.process(peds, trig, chan);
  mProcessed++;

  if (mProcessed >= mModTF) {
    auto& summary = mWorker.mData.getSummary();
    // Send intermediate calibration data and histograms
    if (mVerbosity > DbgMedium && mModTF > 0) {
      LOG(info) << "Send intermediate calibration data mProcessed=" << mProcessed << " >= mModTF=" << mModTF;
    }
    o2::framework::Output outputData("ZDC", "NOISECALIBDATA", 0, Lifetime::Sporadic);
    pc.outputs().snapshot(outputData, summary);
    for (int ih = 0; ih < NChannels; ih++) {
      {
        o2::framework::Output output("ZDC", "NOISE_1DH", ih, Lifetime::Sporadic);
        pc.outputs().snapshot(output, mWorker.mH[0][ih]->getBase());
      }
      {
        o2::framework::Output output("ZDC", "NOISE_1DH_S", ih, Lifetime::Sporadic);
        pc.outputs().snapshot(output, mWorker.mH[1][ih]->getBase());
      }
      {
        o2::framework::Output output("ZDC", "NOISE_1DH_D", ih, Lifetime::Sporadic);
        pc.outputs().snapshot(output, mWorker.mH[2][ih]->getBase());
      }
    }
    mWorker.clear();
    mProcessed = 0;
  }
}

void NoiseCalibEPNSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  LOGF(info, "ZDC EPN Noise calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getNoiseCalibEPNSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  inputs.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathConfigModule.data()));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "NOISECALIBDATA", 0, Lifetime::Sporadic);
  for (int ih = 0; ih < NChannels; ih++) {
    outputs.emplace_back("ZDC", "NOISE_1DH", ih, Lifetime::Sporadic);
    outputs.emplace_back("ZDC", "NOISE_1DH_S", ih, Lifetime::Sporadic);
    outputs.emplace_back("ZDC", "NOISE_1DH_D", ih, Lifetime::Sporadic);
  }
  return DataProcessorSpec{
    "zdc-calib-noise-epn",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibEPNSpec>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
