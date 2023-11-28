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

#include "DetectorsBase/VMCSeederService.h"
#include "TVirtualMC.h"
#include <fairlogger/Logger.h>                    // for FairLogger
#include <CommonUtils/ConfigurationMacroHelper.h> // for ROOT JIT helpers

using namespace o2::base;

void VMCSeederService::initSeederFunction(TVirtualMC const* vmc)
{
  if (strcmp(vmc->GetName(), "TGeant3TGeo") == 0) {
    // Geant3 doesn't need anything special in our context
    mSeederFcn = []() {};
  } else if (strcmp(vmc->GetName(), "TGeant4") == 0) {
    // dynamically get access to the Geant4_VMC seeding function (without this function linking against Geant4)
    std::string G4func("std::function<void()> G4func() { gSystem->Load(\"libgeant4vmc\"); return [](){ ((TGeant4*)TVirtualMC::GetMC())->SetRandomSeed(); };}");
    mSeederFcn = o2::conf::JITAndEvalFunction<SeederFcn>(G4func, "G4func()", "std::function<void()>", "VMCSEEDERFUNC123");
  } else {
    LOG(warn) << "Unknown VMC engine or unimplemented VMC seeding function";
    mSeederFcn = []() {};
  }
}

// constructor
VMCSeederService::VMCSeederService()
{
  auto vmc = TVirtualMC::GetMC();
  if (vmc) {
    LOG(info) << "Seeder initializing for " << vmc->GetName();
    initSeederFunction(vmc);
  } else {
    LOG(fatal) << " Seeder could not be initialized (no VMC instance found)";
  }
}

void VMCSeederService::setSeed() const
{
  mSeederFcn();
}
