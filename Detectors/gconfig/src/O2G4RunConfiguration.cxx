// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file O2G4RunConfiguration.cxx
/// \brief Overriding what is necessary for custom O2 version of TG4RunConfiguration

#include <TG4VUserRegionConstruction.h>
#include <TG4VUserFastSimulation.h>

#include <G4VUserPhysicsList.hh>
#include <G4FastSimulationPhysics.hh>
#include <G4VModularPhysicsList.hh>

#include <FairLogger.h>

#include "SimSetup/O2G4RegionsConstruction.h"
#include "SimSetup/O2G4FastSimConstruction.h"
#include "SimSetup/O2G4RunConfiguration.h"

using namespace o2;

O2G4RunConfiguration::O2G4RunConfiguration(const std::string& userGeometry,
                     const std::string& physicsList, const std::string& specialProcess,
                     bool specialStacking, bool mtApplication)
  : TG4RunConfiguration(userGeometry.c_str(), physicsList.c_str(), specialProcess.c_str(),
                        specialStacking, mtApplication), mFastSimConstruction(nullptr)
{
  LOG(INFO) << "O2 specific run configuration used";
}

TG4VUserRegionConstruction* O2G4RunConfiguration::CreateUserRegionConstruction()
{
  LOG(INFO) << "Construct regions for TGeant4";
  return new O2G4RegionsConstruction();
}

TG4VUserFastSimulation* O2G4RunConfiguration::CreateUserFastSimulation()
{
  /// Create the fast sim construction
  LOG(INFO) << "Create fast simulation construction for TGeant4";
  // TODO This is for now hard-coded here but will be changed.
  // TODO Make region extraction base on volume and not on material as it is
  //      done in GEANT4_VMC
  return new O2G4FastSimConstruction({"TPC_DriftGas2"});
}

void O2G4RunConfiguration::SetFastSimConstruction(TG4VUserFastSimulation* fastSimConstruction)
{
  LOG(INFO) << "Set fast simulation for TGeant4";
  mFastSimConstruction = fastSimConstruction;
}
