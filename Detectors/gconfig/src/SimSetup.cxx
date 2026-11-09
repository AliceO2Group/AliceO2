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

#include <cstring>
#include "SimSetup/SimSetup.h"
#include "TVirtualMC.h"
#include <fairlogger/Logger.h>
#include "SetCuts.h"
#include "CommonUtils/DLLoaderBase.h"
#include <dlfcn.h>
#include <string>

namespace o2
{

typedef void (*setup_fnc)();

void execFromPlugin(const char* libname, const char* funcname)
{
  const std::string filename = std::string(libname) + DLLOADER_MAC_LINUX(".dylib", ".so");
  auto libHandle = dlopen(filename.c_str(), RTLD_NOW);
  if (!libHandle) {
    const char* error = dlerror();
    LOG(fatal) << "Cannot load " << filename << ": " << (error ? error : "unknown error");
    // LOG(fatal) ends the process only when a fatal handler is installed
    throw fair::FatalException("Cannot load " + filename);
  }
  dlerror(); // clear any pending error before dlsym
  auto func = (setup_fnc)dlsym(libHandle, funcname);
  if (!func) {
    const char* error = dlerror();
    LOG(fatal) << "Cannot find " << funcname << " in " << filename << ": " << (error ? error : "the symbol is null");
    throw fair::FatalException("Cannot find " + std::string(funcname) + " in " + filename);
  }
  func();
}

void execSetupFromPlugin(const char* libname, const char* funcname)
{
  LOG(info) << "Setting up transport engine from library " << libname;
  execFromPlugin(libname, funcname);
}

void SimSetup::setup(const char* engine)
{
  if (strcmp(engine, "TGeant3") == 0) {
    execSetupFromPlugin("libO2G3Setup", "_ZN2o28g3config8G3ConfigEv");
  } else if (strcmp(engine, "TGeant4") == 0) {
    execSetupFromPlugin("libO2G4Setup", "_ZN2o28g4config8G4ConfigEv");
  } else if (strcmp(engine, "TFluka") == 0) {
    execSetupFromPlugin("libO2FLUKASetup", "_ZN2o211flukaconfig11FlukaConfigEv");
  } else if (strcmp(engine, "MCReplay") == 0) {
    execSetupFromPlugin("libO2MCReplaySetup", "_ZN2o214mcreplayconfig14MCReplayConfigEv");
  } else if (strcmp(engine, "O2TrivialMCEngine") == 0) {
    execSetupFromPlugin("libO2O2TrivialMCEngineSetup", "_ZN2o223o2trivialmcengineconfig23O2TrivialMCEngineConfigEv");
  } else {
    LOG(fatal) << "Unsupported engine " << engine;
  }
  o2::SetCuts();
}

// function to shutdown the engines and do some necessary
// finalisation work
void SimSetup::shutdown()
{
  auto vmc = TVirtualMC::GetMC();
  if (strcmp(vmc->GetName(), "TGeant4") == 0) {
    execFromPlugin("libO2G4Setup", "_ZN2o28g4config11G4TerminateEv");
  }
}

} // namespace o2
