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
#include "TGeant3.h"
#include "TGeant3TGeo.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/StackParam.h"
#include <fairlogger/Logger.h>
#include "FairModule.h"
#include "Generators/DecayerPythia8.h"

// these are used in commonConfig.C
using o2::eventgen::DecayerPythia8;

namespace o2
{
namespace g3config
{
#include "../g3Config.C"

void G3Config()
{
  LOG(info) << "Setting up G3 sim from library code";
  Config();
}
} // namespace g3config
} // namespace o2
