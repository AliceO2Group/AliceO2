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


/// \file FT3Materials.cxx
/// \brief Access to the media created from the FT3 material map

#include "FT3Simulation/FT3Materials.h"

#include "DetectorsBase/MaterialManager.h"

#include <fairlogger/Logger.h>

TGeoMedium* o2::ft3::getMedium(Materials::MaterialID id)
{
  auto* medium = o2::base::MaterialManager::Instance().getTGeoMedium(Materials::moduleName, static_cast<int>(id));
  if (!medium) {
    LOG(fatal) << "FT3: no medium registered for " << Materials::materials.at(id).name
               << "; createMaterials() has to run before the geometry is built";
  }
  return medium;
}
