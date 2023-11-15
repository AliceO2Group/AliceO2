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

#ifndef O2_FRAMEWORK_O2DATABASEPDG_H_
#define O2_FRAMEWORK_O2DATABASEPDG_H_

#include "Framework/Plugins.h"
#include "TDatabasePDG.h"
#include "SimulationDataFormat/O2DatabasePDG.h"

namespace o2::framework
{
struct O2DatabasePDGImpl : public TDatabasePDG {
  Double_t Mass(int pdg)
  {
    // wrap our own Mass function to expose it in the service
    bool success = false;
    auto mass = o2::O2DatabasePDG::Mass(pdg, success, this);
    if (!success) {
      LOGF(error, "Unknown particle with PDG code %d", pdg);
    }
    return mass;
  }
};

struct O2DatabasePDG : LoadableServicePlugin<O2DatabasePDGImpl> {
  O2DatabasePDG() : LoadableServicePlugin{"O2FrameworkPhysicsSupport:PDGSupport"}
  {
  }
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_O2DATABASEPDG_H_
