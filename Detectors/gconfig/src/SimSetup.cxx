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
#include <fairlogger/Logger.h>
#include "SetCuts.h"
#include <dlfcn.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <sstream>

namespace o2
{

typedef void (*setup_fnc)();

void setupFromPlugin(const char* libname, const char* setupfuncname)
{
  LOG(info) << "Loading simulation plugin " << libname;
  auto libHandle = dlopen(libname, RTLD_NOW);
  // try to make the library loading a bit more portable:
  if (!libHandle) {
    // try appending *.so
    std::stringstream stream;
    stream << libname << ".so";
    libHandle = dlopen(stream.str().c_str(), RTLD_NOW);
  }
  if (!libHandle) {
    // try appending *.dylib
    std::stringstream stream;
    stream << libname << ".dylib";
    libHandle = dlopen(stream.str().c_str(), RTLD_NOW);
  }
  assert(libHandle);
  auto setup = (setup_fnc)dlsym(libHandle, setupfuncname);
  assert(setup);
  setup();
}

void SimSetup::setup(const char* engine)
{
  if (strcmp(engine, "TGeant3") == 0) {
    setupFromPlugin("libO2G3Setup", "_ZN2o28g3config8G3ConfigEv");
  } else if (strcmp(engine, "TGeant4") == 0) {
    setupFromPlugin("libO2G4Setup", "_ZN2o28g4config8G4ConfigEv");
  } else if (strcmp(engine, "TFluka") == 0) {
    setupFromPlugin("libO2FLUKASetup", "_ZN2o211flukaconfig11FlukaConfigEv");
  } else if (strcmp(engine, "MCReplay") == 0) {
    setupFromPlugin("libO2MCReplaySetup", "_ZN2o214mcreplayconfig14MCReplayConfigEv");
  } else if (strcmp(engine, "O2TrivialMCEngine") == 0) {
    setupFromPlugin("libO2O2TrivialMCEngineSetup", "_ZN2o223o2trivialmcengineconfig23O2TrivialMCEngineConfigEv");
  } else {
    LOG(fatal) << "Unsupported engine " << engine;
  }
  o2::SetCuts();
}
} // namespace o2
