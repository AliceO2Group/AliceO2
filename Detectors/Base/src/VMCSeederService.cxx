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
#include <fairlogger/Logger.h> // for FairLogger
#include <TRandom.h>

using namespace o2::base;

void VMCSeederService::initSeederFunction(TVirtualMC const* vmc)
{
  if (strcmp(vmc->GetName(), "TGeant3TGeo") == 0) {
    // Geant3 doesn't need anything special in our context
    mSeederFcn = [](TRandom*) {};
  } else if (strcmp(vmc->GetName(), "TGeant4") == 0) {
    // installed by the Geant4 configuration, so that this library does not link against Geant4
    mSeederFcn = geant4Seeder();
    if (!mSeederFcn) {
      LOG(fatal) << "No Geant4 seeding function installed (VMCSeederService::setGeant4Seeder)";
    }
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
