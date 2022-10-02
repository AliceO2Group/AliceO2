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

#include "../macro/o2sim.C"
#include <SimConfig/SimConfig.h>
#include <TStopwatch.h>
#include <fairlogger/Logger.h>

int main(int argc, char* argv[])
{
  TStopwatch timer;
  timer.Start();
  auto& conf = o2::conf::SimConfig::Instance();
#ifdef SIM_RUN5
  conf.setRun5();
#endif
  if (!conf.resetFromArguments(argc, argv)) {
    return 1;
  }

  // customize the level of output
  FairLogger::GetLogger()->SetLogScreenLevel(conf.getLogSeverity().c_str());
  FairLogger::GetLogger()->SetLogVerbosityLevel(conf.getLogVerbosity().c_str());

  // call o2sim "macro"
  o2sim(false);

  o2::utils::ShmManager::Instance().release();

  // print total time
  LOG(info) << "Simulation process took " << timer.RealTime() << " s";

  // We do this instead of return 0
  // for the reason that we see lots of problems
  // with TROOTs atexit mechanism often triggering double-free or delete symptoms.
  // While this is not optimal ... I think it is for the moment
  // better to have a stable simulation runtime in contrast to
  // having to debug complicated "atexit" memory problems.
  _exit(0);
}
