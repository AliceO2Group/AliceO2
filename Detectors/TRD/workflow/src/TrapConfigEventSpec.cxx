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

/// \file   TrapConfigEventSpec.cxx
/// \brief  DPL device for handling the trap configuration events
/// \author Sean Murray

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include <fairlogger/Logger.h>

#include "TRDReconstruction/TrapConfigEventParser.h"
#include "TRDWorkflow/TrapConfigEventSpec.h"
#include "DataFormatsTRD/TrapConfigEvent.h"

#include "TStopwatch.h"
#include "TFile.h"

using namespace o2::framework;

namespace o2::trd
{

class TrapConfigEventDevice : public Task
{
 public:
  TrapConfigEventDevice() = default;
  ~TrapConfigEventDevice() = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  o2::trd::TrapConfigEventParser mTrapConfigEventParser;
  std::bitset<constants::MAXHALFCHAMBER> mOldHalfChamberPresent; // keep a store of which half chambers have been seen so far.
  uint32_t mConfigCounter = 0;
};

void TrapConfigEventDevice::init(InitContext& ic)
{
  mTrapConfigEventParser.init();
  mOldHalfChamberPresent.set(0);
  mConfigCounter = 0;
}

void TrapConfigEventDevice::run(ProcessingContext& pc)
{
  TStopwatch timer;

  std::vector<uint32_t> configdata = pc.inputs().get<std::vector<uint32_t>>("trapconfigs");

  LOGP(info, " Config event coming in with size : {}", configdata.size());
  timer.Start();
  mTrapConfigEventParser.parse(configdata);
  timer.Stop();

  LOGP(info, "TRD config event took:  Cpu: {:.3} s Real: {:.3} s halfchambers with data : {}, and a datasize of {}MB", timer.CpuTime(), timer.RealTime(), mTrapConfigEventParser.countHCIDPresent(), ((float)configdata.size()) / 1025. / 1024.);
  timer.Start();
  mTrapConfigEventParser.isNewConfig();
  timer.Stop();
  LOGF(info, "Comparison to previous one config in ccdb took : Cpu: %.3e Real: %.3e s", timer.CpuTime(), timer.RealTime());
  mOldHalfChamberPresent = mTrapConfigEventParser.countHCIDPresent();
  //  if (mConfigCounter % 5 == 0) {
  // send the config message
  LOGP(info, "We have a new config and sending ....");
  // LOGP(info,"first value of tpl00 : {}",mTrapConfigEventParser.getNewConfig().getRegisterValue(0,31361));
  // TrapConfigEvent a;
  // auto a=mTrapConfigEventParser.getNewConfig();
  // LOGP(info,"first value of copied trapconfigevent A is tpl00 : {}",a.getRegisterValue(0,31361));
  LOGP(info, "Trap config is size : {}", mTrapConfigEventParser.getConfigSize());
  mTrapConfigEventParser.sendTrapConfigEvent(pc);
  LOGP(info, "Trap config has been sent : {}", mTrapConfigEventParser.getConfigSize());
  //  }
  mConfigCounter++;
  if (mConfigCounter == 20)
    exit(1);
}

void TrapConfigEventDevice::endOfStream(EndOfStreamContext& ec)
{
  LOG(info) << "Finished with TRD Trap Config event (EoS received)";
}

DataProcessorSpec getTrapConfigEventSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trapconfigs", ConcreteDataTypeMatcher{o2::header::gDataOriginTRD, "CONFEVT"}, Lifetime::Sporadic);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRDCFG", 0, Lifetime::Condition);

  return DataProcessorSpec{
    "trd-configevents",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrapConfigEventDevice>()},
    Options{}};
}

} // namespace o2::trd
