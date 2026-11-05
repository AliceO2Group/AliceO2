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
#include "DetectorsBase/SimRandom.h"
#include "TVirtualMC.h"
#include <fairlogger/Logger.h>                    // for FairLogger
#include <CommonUtils/ConfigurationMacroHelper.h> // for ROOT JIT helpers
#include <TInterpreter.h>
#include <TRandom.h>

using namespace o2::base;

void VMCSeederService::initSeederFunction(TVirtualMC const* vmc)
{
  if (strcmp(vmc->GetName(), "TGeant3TGeo") == 0) {
    // Geant3 doesn't need anything special in our context
    mSeederFcn = [](TRandom*) {};
  } else if (strcmp(vmc->GetName(), "TGeant4") == 0) {
    // Seed the Geant4 engine of the calling thread from the given generator, with the same recipe as
    // TG4RunManager::SetRandomSeed (which always reads gRandom). JIT-compiled to avoid linking against Geant4.
    if (!gInterpreter->Declare("#include <CLHEP/Random/Random.h>")) {
      LOG(fatal) << "CLHEP/Random/Random.h not found on ROOT_INCLUDE_PATH; cannot seed Geant4";
    }
    std::string G4func("std::function<void(TRandom*)> G4func() { gSystem->Load(\"libgeant4vmc\"); return [](TRandom* r){ long seeds[10]; seeds[0] = r->GetSeed(); seeds[1] = r->GetSeed(); seeds[2] = 0; CLHEP::HepRandom::setTheSeeds(seeds); };}");
    mSeederFcn = o2::conf::JITAndEvalFunction<SeederFcn>(G4func, "G4func()", "std::function<void(TRandom*)>", "VMCSEEDERFUNC123");
  } else {
    LOG(warn) << "Unknown VMC engine or unimplemented VMC seeding function";
    mSeederFcn = [](TRandom*) {};
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
  // Make sure the generator is sufficiently well initialized
  // by calling Rndm() once.
  // This is ok since in any case r->SetSeed(seed); r->GetSeed() != seed;
  auto r = simRandom();
  r->Rndm();
  mSeederFcn(r);
  ++mSeedCount;
}
