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
#include <DetectorsPassive/Cave.h>
#include "DetectorsBase/MaterialManager.h"
#include "SimSetup/GlobalProcessCutSimParam.h"
#include "SimConfig/G4Params.h"

//using declarations here since SetCuts.C and g4Config.C are included within namespace
// these are needed for SetCuts.C inclusion
using o2::GlobalProcessCutSimParam;
using o2::base::ECut;
using o2::base::EProc;
using o2::base::MaterialManager;
// these are used in g4Config.C
using std::cout;
using std::endl;

namespace o2
{
namespace g4config
{
#include "../g4Config.C"
#include "../SetCuts.h"

void G4Config()
{
  LOG(INFO) << "Setting up G4 sim from library code";
  Config();
  SetCuts();
}
} // namespace g4config
} // namespace o2
