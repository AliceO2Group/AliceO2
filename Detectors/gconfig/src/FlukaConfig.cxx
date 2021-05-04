// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FairRunSim.h"
#include "TFluka.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/StackParam.h"
#include <iostream>
#include "FairLogger.h"
#include "FairModule.h"
#include "Generators/DecayerPythia8.h"
#include "SimSetup/FlukaParam.h"
#include "../commonConfig.C"

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
  //
  linkFlukaFiles();
  FairRunSim* run = FairRunSim::Instance();
  TString* gModel = run->GetGeoModel();
  TFluka* fluka = new TFluka("C++ Interface to Fluka", 0);

  // additional configuration paramters if requested from command line
  auto& params = FlukaParam::Instance();
  auto isAct = params.activationSimulation;
  if (isAct) {
    LOG(INFO) << "Set special FLUKA parameters for activation simulation";
    auto hadronCut = params.activationHadronCut;
    auto inpFile = params.scoringFile;
    fluka->SetActivationSimulation(true, hadronCut);
    fluka->SetUserScoringFileName(inpFile.c_str());
  }
  stackSetup(fluka, run);

  // setup decayer
  decayerSetup(fluka);

  // ******* FLUKA  specific configuration for simulated Runs  *******
}

void FlukaConfig()
{
  LOG(INFO) << "Setting up FLUKA sim from library code";
  Config();
}
} // namespace flukaconfig
} // namespace o2
