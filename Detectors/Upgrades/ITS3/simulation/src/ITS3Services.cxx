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

/// \file ITS3Services.h
/// \brief Definition of the ITS3Services class
/// \author Fabrizio Grosa <fgrosa@cern.ch>

#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <TGeoTube.h>

#include "ITS3Simulation/ITS3Services.h"
#include "ITS3Base/SpecsV2.h"

namespace o2::its3
{

void ITS3Services::createCYSSAssembly(TGeoVolume* motherVolume)
{
  auto cyssVol = new TGeoVolumeAssembly("IBCYSSAssembly");
  cyssVol->SetVisibility(kTRUE);
  motherVolume->AddNode(cyssVol, 1., nullptr);

  // Cylinder
  auto cyssInnerCylSh = new TGeoTubeSeg(constants::services::radiusInner, constants::services::radiusOuter, constants::services::length / 2, 180, 360);
  auto medRohacell = gGeoManager->GetMedium("IT3_RIST110$");
  auto cyssInnerCylShVol = new TGeoVolume("IBCYSSCylinder", cyssInnerCylSh, medRohacell);
  cyssInnerCylShVol->SetLineColor(constants::services::color);
  cyssVol->AddNode(cyssInnerCylShVol, 1, new TGeoTranslation(0, 0, 0));
  cyssVol->AddNode(cyssInnerCylShVol, 2, new TGeoCombiTrans(0, 0, 0, new TGeoRotation("", 180, 0, 0)));

  // TODO Cone
  // For now the wrapping volume just extends beyond the cylinder if something is added beyond that this volume has to
  // be exteneded.
}

} // namespace o2::its3
