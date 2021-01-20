// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsBase/MaterialManager.h"
#include "Alice3DetectorsPassive/PassiveBase.h"

using namespace o2::passive;

void Alice3PassiveBase::SetSpecialPhysicsCuts()
{
  // default implementation for physics cuts setting (might still be overriden by detectors)
  // we try to read an external text file supposed to be installed
  // in a standard directory
  // ${O2_ROOT}/share/Detectors/DETECTORNAME/simulation/data/simcuts.dat
  LOG(INFO) << "Setting special cuts for passive module " << GetName();
  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputFile;
  if (aliceO2env) {
    inputFile = std::string(aliceO2env);
  }
  inputFile += "/share/Detectors/Upgrades/Passive/simulation/data/simcuts_" + std::string(GetName()) + ".dat";
  auto& matmgr = o2::base::MaterialManager::Instance();
  matmgr.loadCutsAndProcessesFromFile(GetName(), inputFile.c_str());
}
