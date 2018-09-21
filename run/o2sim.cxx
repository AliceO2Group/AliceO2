// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "../macro/o2sim.C"
#include <SimConfig/SimConfig.h>
#include <TStopwatch.h>
#include <FairLogger.h>

int main(int argc, char* argv[])
{
  TStopwatch timer;
  timer.Start();
  auto& conf = o2::conf::SimConfig::Instance();
  if (!conf.resetFromArguments(argc, argv)) {
    return 1;
  }

  // customize the level of output
  FairLogger::GetLogger()->SetLogScreenLevel(conf.getLogSeverity().c_str());
  FairLogger::GetLogger()->SetLogVerbosityLevel(conf.getLogVerbosity().c_str());

  // call o2sim "macro"
  o2sim(false);

  // print total time
  LOG(INFO) << "Simulation process took " << timer.RealTime() << " s";

  // We do this instead of return 0
  // for the reason that we see lots of problems
  // with TROOTs atexit mechanism often triggering double-free or delete symptoms.
  // While this is not optimal ... I think it is for the moment
  // better to have a stable simulation runtime in contrast to
  // having to debug complicated "atexit" memory problems.
  _exit(0);
}
