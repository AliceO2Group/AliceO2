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

namespace o2
{
// forward declarations of functions
namespace g3config
{
void G3Config();
}
namespace g4config
{
void G4Config();
}

void SimSetup::setup(const char* engine)
{
  if (strcmp(engine, "TGeant3") == 0) {
    g3config::G3Config();
  } else if (strcmp(engine, "TGeant4") == 0) {
    g4config::G4Config();
  } else {
    LOG(FATAL) << "Unsupported engine " << engine;
  }
}
} // namespace o2
