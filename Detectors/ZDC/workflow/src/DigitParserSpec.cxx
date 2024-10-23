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

/// @file   DigitParserSpec.cxx
/// @brief  ZDC digits parser
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "ZDCWorkflow/DigitParserSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

DigitParserSpec::DigitParserSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

DigitParserSpec::DigitParserSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void DigitParserSpec::init(o2::framework::InitContext& ic)
{
  mWorker.setOutput(ic.options().get<std::string>("parser-output"));
  mWorker.setRejectPileUp((ic.options().get<int>("reject-pileup")) != 0);
}

void DigitParserSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
}

void DigitParserSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "MODULECONFIG", 0)) {
    auto* config = (const o2::zdc::ModuleConfig*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setModuleConfig(config);
  }
}

void DigitParserSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    LOG(info) << "DigitParserSpec::run initialization";
    mInitialized = true;
    updateTimeDependentParams(pc);
    mWorker.setVerbosity(mVerbosity);
    mWorker.init();
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  auto bcdata = pc.inputs().get<gsl::span<o2::zdc::BCData>>("trig");
  auto chans = pc.inputs().get<gsl::span<o2::zdc::ChannelData>>("chan");
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");

  int rval = mWorker.process(peds, bcdata, chans);
  if (rval != 0) {
    LOG(warning) << bcdata.size() << " BC " << chans.size() << " CH " << peds.size() << " OD -> processing ended in ERROR @ line " << rval;
  }
  mTimer.Stop();
}

void DigitParserSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.eor();
  LOGF(info, "ZDC digits parsing total time: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getDigitParserSpec(const int verbosity = 0)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  inputs.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathConfigModule.data()));

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "zdc-digi-parser",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DigitParserSpec>(verbosity)},
    o2::framework::Options{{"parser-output", o2::framework::VariantType::String, "ZDCDigiParser.root", {"Output file name"}},
                           {"reject-pileup", o2::framework::VariantType::Int, 1, {"Reject pile-up for signal shapes 0/1"}}}};
}

} // namespace zdc
} // namespace o2
