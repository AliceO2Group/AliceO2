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
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/StackParam.h"
#include <iostream>
#include "FairLogger.h"
#include "TGeant4.h"
#include "TG4RunConfiguration.h"
#include "TPythia6Decayer.h"
#include "FairModule.h"
#include "SimConfig/G4Params.h"
#include "Generators/DecayerPythia8.h"

//using declarations here since SetCuts.C and g4Config.C are included within namespace
// these are used in g4Config.C
using std::cout;
using std::endl;
// these are used in commonConfig.C
using o2::eventgen::DecayerPythia8;

namespace o2
{
namespace g4config
{
#include "../g4Config.C"

void G4Config()
{
  LOG(INFO) << "Setting up G4 sim from library code";
  Config();
}
} // namespace g4config
} // namespace o2
