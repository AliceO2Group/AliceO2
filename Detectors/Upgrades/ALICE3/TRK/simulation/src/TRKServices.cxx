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

#include <TRKSimulation/TRKServices.h>
#include <DetectorsBase/MaterialManager.h>
#include <TGeoVolume.h>
#include <TGeoTube.h>
#include <Rtypes.h>

#include <Framework/Logger.h>

namespace o2
{
namespace trk
{
TRKServices::TRKServices(float rMin, float zLength, float thickness)
{
  mRMin = rMin;
  mZLength = zLength;
  mThickness = thickness;
}

void TRKServices::createMaterials()
{
  int ifield = 2;      // ?
  float fieldm = 10.0; // ?

  // Defines tracking media parameters.
  float epsil = .1;     // Tracking precision,
  float stemax = -0.01; // Maximum displacement for multiple scat
  float tmaxfd = -20.;  // Maximum angle due to field deflection
  float deemax = -.3;   // Maximum fractional energy loss, DLS
  float stmin = -.8;

  auto& matmgr = o2::base::MaterialManager::Instance();

  // Ceramic (Aluminium Oxide)
  float aCer[2] = {26.981538, 15.9994};
  float zCer[2] = {13., 8.};
  float wCer[2] = {0.5294, 0.4706}; // Mass %, which makes sense. TODO: check if Mixture needs mass% or comp%
  float dCer = 3.97;

  matmgr.Mixture("COLDPLATE", 66, "CERAMIC$", aCer, zCer, dCer, 2, wCer); // Ceramic for cold plate
  matmgr.Medium("COLDPLATE", 66, "CER", 66, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);
}

void TRKServices::createServices(TGeoVolume* motherVolume)
{
  createMaterials();

  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCeramic = matmgr.getTGeoMedium("COLDPLATE_CER");

  TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", mRMin, mRMin + mThickness, mZLength / 2.);
  TGeoVolume* coldPlateVolume = new TGeoVolume("TRK_COLDPLATE", coldPlate, medCeramic);
  coldPlateVolume->SetVisibility(1);
  coldPlateVolume->SetLineColor(kYellow);

  LOGP(info, "Creating cold plate service");

  LOGP(info, "Inserting {} in {} ", coldPlateVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(coldPlateVolume, 1, nullptr);
}
// ClassImp(o2::trk::TRKServices);
} // namespace trk
} // namespace o2