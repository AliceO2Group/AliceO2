// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cstring>
#include "SimSetup/SimSetup.h"
#include "FairLogger.h"
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
  LOG(INFO) << "Loading simulation plugin " << libname;
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
  } else {
    LOG(FATAL) << "Unsupported engine " << engine;
  }
  o2::SetCuts();
}
} // namespace o2
