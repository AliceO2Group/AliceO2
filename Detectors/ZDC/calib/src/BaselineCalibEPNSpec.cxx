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

/// @file   BaselineCalibEPNSpec.cxx
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
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/RecoParamZDC.h"
#include "ZDCCalib/BaselineCalibData.h"
#include "ZDCCalib/BaselineCalibEPNSpec.h"
#include "ZDCCalib/CalibParamZDC.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

BaselineCalibEPNSpec::BaselineCalibEPNSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

BaselineCalibEPNSpec::BaselineCalibEPNSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void BaselineCalibEPNSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
  const auto& opt = CalibParamZDC::Instance();
  mModTF = opt.modTF;
  if (mVerbosity >= DbgZero) {
    LOG(info) << "Sending calibration data to aggregator every mModTF = " << mModTF << " TF";
  }
}

void BaselineCalibEPNSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
}

void BaselineCalibEPNSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "MODULECONFIG", 0)) {
    auto* config = (const o2::zdc::ModuleConfig*)obj;
    if (mVerbosity >= DbgFull) {
      config->print();
    }
    mWorker.setModuleConfig(config);
  }
}

void BaselineCalibEPNSpec::run(ProcessingContext& pc)
{
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();

  if (tinfo.globalRunNumberChanged) { // new run is starting
    LOG(info) << "Run number changed to " << tinfo.runNumber;
    mRunStopRequested = false;
    mInitialized = false;
    mWorker.resetInitFlag();
  }

  if (mRunStopRequested) {
    return;
  }

  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }

  // Process reconstructed data
  auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");
  mWorker.process(peds);
  mWorker.getData().mergeCreationTime(creationTime);
  mProcessed++;

  if (mProcessed >= mModTF || pc.transitionState() == TransitionHandlingState::Requested) {
    if (mVerbosity >= DbgMedium) {
      if (mModTF > 0 && mProcessed >= mModTF) {
        LOG(info) << "Send intermediate calibration data mProcessed=" << mProcessed << " >= mModTF=" << mModTF;
      }
      if (pc.transitionState() == TransitionHandlingState::Requested) {
        LOG(info) << "Send intermediate calibration data pc.transitionState()==TransitionHandlingState::Requested";
      }
    }
    // Send intermediate calibration data
    auto& summary = mWorker.mData.getSummary();
    o2::framework::Output outputData("ZDC", "BASECALIBDATA", 0);
    pc.outputs().snapshot(outputData, summary);
    if (pc.transitionState() == TransitionHandlingState::Requested) {
      // End of processing for this run
      mWorker.endOfRun();
      LOGF(info, "ZDC EPN Baseline pausing at total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
      mRunStopRequested = true;
    } else {
      // Prepare to process other time frames
      mWorker.resetInitFlag();
    }
    // Clear data already transmitted
    mWorker.mData.clear();
    mProcessed = 0;
  }
}

void BaselineCalibEPNSpec::endOfStream(EndOfStreamContext& ec)
{
#ifdef O2_ZDC_DEBUG
  LOG(info) << "BaselineCalibEPNSpec::endOfStream() mRunStopRequested=" << mRunStopRequested << " mProcessed=" << mProcessed;
#endif
  if (mRunStopRequested) {
    return;
  }
  // This (unfortunately) is not received by aggregator
  //   if(mProcessed>0){
  //     if(mVerbosity>=DbgMedium){
  //       LOG(info) << "Send calibration data at endOfStream() mProcessed=" << mProcessed;
  //     }
  //     auto& summary = mWorker.mData.getSummary();
  //     o2::framework::Output outputData("ZDC", "BASECALIBDATA", 0, Lifetime::Sporadic);
  //     printf("Sending last processed data mProcessed = %u\n", mProcessed);
  //     ec.outputs().snapshot(outputData, summary);
  //   }
  mWorker.endOfRun();
  mTimer.Stop();
  LOGF(info, "ZDC EPN Baseline calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  mRunStopRequested = true;
}

framework::DataProcessorSpec getBaselineCalibEPNSpec()
{
  using device = o2::zdc::BaselineCalibEPNSpec;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  inputs.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathConfigModule.data()));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "BASECALIBDATA", 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-calib-baseline-epn",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
