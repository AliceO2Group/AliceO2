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

#include "FairRunSim.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/StackParam.h"
#include <iostream>
#include <fairlogger/Logger.h>
#include "FairModule.h"
#include "Generators/DecayerPythia8.h"
#include "SimSetup/FlukaParam.h"
#include "../commonConfig.C"
#include "CommonUtils/ConfigurationMacroHelper.h"

// these are used in commonConfig.C
using o2::eventgen::DecayerPythia8;

namespace o2
{
namespace flukaconfig
{

void linkFlukaFiles()
{
  // Link here some special Fluka files needed
  gSystem->Exec("ln -s $FLUKADATA/neuxsc.bin  .");
  gSystem->Exec("ln -s $FLUKADATA/elasct.bin  .");
  gSystem->Exec("ln -s $FLUKADATA/gxsect.bin  .");
  gSystem->Exec("ln -s $FLUKADATA/nuclear.bin .");
  gSystem->Exec("ln -s $FLUKADATA/sigmapi.bin .");
  gSystem->Exec("ln -s $FLUKADATA/brems_fin.bin .");
  gSystem->Exec("ln -s $FLUKADATA/cohff.bin .");
  gSystem->Exec("ln -s $FLUKADATA/fluodt.dat  .");
  gSystem->Exec("ln -s $FLUKADATA/random.dat  .");
  gSystem->Exec("ln -s $FLUKADATA/dnr.dat  .");
  gSystem->Exec("ln -s $FLUKADATA/nunstab.data .");
  // Give some meaningfull name to the output
  gSystem->Exec("ln -s fluka.out fort.11");
  gSystem->Exec("ln -s fluka.err fort.15");
  gSystem->Exec("ln -fs $O2_ROOT/share/Detectors/gconfig/data/coreFlukaVmc.inp .");
}

void Config()
{
  linkFlukaFiles();
  FairRunSim* run = FairRunSim::Instance();
  // try to see if Fluka is available in the runtime
  auto status = gSystem->Load("libflukavmc");
  if (status == 0 || status == 1) {
    // we load Fluka as a real plugin via a ROOT Macro
    auto fluka = o2::conf::GetFromMacro<TVirtualMC*>("$O2_ROOT/share/Detectors/gconfig/FlukaRuntimeConfig.macro", "FlukaRuntimeConfig()", "TVirtualMC*", "foo");
    stackSetup(fluka, run);
    decayerSetup(fluka);
  } else {
    LOG(error) << "FLUKA is not available in the runtime environment";
    LOG(error) << "Please compile and load by including FLUKA_VMC/latest in the alienv package list";
    LOG(fatal) << "Quitting here due to FLUKA_VMC not being available";
  }
  return;
}

void FlukaConfig()
{
  LOG(info) << "Setting up FLUKA sim from library code";
  Config();
}
} // namespace flukaconfig
} // namespace o2
