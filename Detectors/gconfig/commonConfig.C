// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// common piece of code to setup stack and register
// with VMC instances

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsBase/Stack.h"
#include "SimulationDataFormat/StackParam.h"
#include "SimConfig/SimParams.h"
#include "TVirtualMC.h"
#include "Generators/DecayerPythia8.h"
#endif

template <typename T, typename R>
void stackSetup(T* vmc, R* run)
{
  // create the O2 vmc stack instance
  auto st = new o2::data::Stack();
  st->setMinHits(1);
  auto& stackparam = o2::sim::StackParam::Instance();
  st->StoreSecondaries(stackparam.storeSecondaries);
  st->pruneKinematics(stackparam.pruneKine);
  st->setTrackSeedingMode(o2::conf::SimCutParams::Instance().trackSeed);
  vmc->SetStack(st);

  /*
  // register the stack as an observer on FinishPrimary events (managed by Cave)
  bool foundCave = false;
  auto modules = run->GetListOfModules();

  if( strcmp(vmc->GetName(), "TGeant4")==0 ) {
    // there is no way to get the module by name, so we have
    // to iterate through the complete list
    for (auto m : *modules) {
      if(strcmp("CAVE", ((FairModule*)m)->GetName()) == 0) {
        // this thing is the cave
        if(auto c=dynamic_cast<o2::passive::Cave*>(m)) {
          foundCave = true;
          c->addFinishPrimaryHook([st](){ st->notifyFinishPrimary();});
        }
      }
    }
    if (!foundCave) {
      LOG(fatal) << "Cave volume not found; Could not attach observers";
    }
  }
*/
}

TVirtualMCDecayer* makeDecayer()
{
  auto decayer = new o2::eventgen::DecayerPythia8;
  decayer->Init();
  return decayer;
}

void decayerSetup(TVirtualMC* vmc)
{
  vmc->SetExternalDecayer(makeDecayer());
}
