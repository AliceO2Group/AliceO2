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
#include <TGeoCompositeShape.h>
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
  mZLengthIRISVacV = 70.;
  mThicknessIRISVacV = 150.e-3;
  mRInIRISVacV = 0.48;
  mROutIRISVacV = mRMin + mThickness;
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

  // IRIS Tracker vacuum vessel materials under consideration: Beryllium, Al-Be-Met, Aluminium 5083
  matmgr.Material("IRISVACUUMVESSEL", 5, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7);
  matmgr.Medium("IRISVACUUMVESSEL", 5, "BE", 5, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // Aluminium 5083 - alloy of Mn, Fe, Cu, Mg, Si, Zn, Cr, Ti, Al
  // https://www.smithmetal.com/5083.htm
  float aAl5083[9] = {54.938, 55.845, 63.546, 24.305, 28.086, 65.38, 51.996, 47.867, 26.982}; 
  float zAl5083[9] = {25., 26., 29., 12., 14., 30., 24., 22., 13.};
  // Exact composition of the Al5083 alloy that will be used for the beam pipe needs to be checked
  float wAl5083[9] = {0.007, 0.004, 0.001, 0.0445, 0.004, 0.0025, 0.0015, 0.0015, 0.934};
  float dAl5083 = 2.650;

  matmgr.Mixture("IRISVACUUMVESSEL", 6, "ALUMINIUM5083$", aAl5083, zAl5083, dAl5083, 9, wAl5083);
  matmgr.Medium("IRISVACUUMVESSEL", 6, "AL5083", 6, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // AlBeMet AM162H is a nanocomposite, not an alloy
  // Considered here as well https://indico.cern.ch/event/1168385/contributions/5355805/attachments/2681743/4652030/Jul%2010%201030-1045%20AM%20(Hawaii)%20M1Or1C-05%20AlBeMet.pdf
  float aAlBeMet[2] = {26.982, 9.012}; 
  float zAlBeMet[2] = {13., 4.};
  float wAlBeMet[2] = {0.38, 0.62};
  float dAlBeMet = 2.071;
  matmgr.Mixture("IRISVACUUMVESSEL", 7, "ALUMINIUM-BERYLLIUM-METAL$", aAlBeMet, zAlBeMet, dAlBeMet, 2, wAlBeMet);
  matmgr.Medium("IRISVACUUMVESSEL", 7, "ALBEMET", 7, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);
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

  // IRIS Tracker Vacuum Vessel
  TGeoTube* irisVacuumVesselInnerTube = new TGeoTube("TRK_IRISVACUUMVESSEL_INNERTUBEsh", mRInIRISVacV, mRInIRISVacV + mThicknessIRISVacV, mZLengthIRISVacV/2.);
  TGeoTube* irisVacuumVesselOuterTube = new TGeoTube("TRK_IRISVACUUMVESSEL_OUTERTUBEsh", mROutIRISVacV, mROutIRISVacV + mThicknessIRISVacV, mZLengthIRISVacV/2.);
  TGeoTube* irisVacuumVesselWallNegZSideTube = new TGeoTube("TRK_IRISVACUUMVESSEL_WALLNEGZSIDEsh", mRInIRISVacV, mROutIRISVacV + mThicknessIRISVacV, mThicknessIRISVacV/2.);
  TGeoTranslation* irisVacVWallNegZ = new TGeoTranslation("IRISVACVWELLNEGZ", 0., 0., -mZLengthIRISVacV/2. - mThicknessIRISVacV/2.);
  irisVacVWallNegZ->RegisterYourself();
  TGeoTube* irisVacuumVesselWallPosZSideTube = new TGeoTube("TRK_IRISVACUUMVESSEL_WALLPOSZSIDEsh", mRInIRISVacV, mROutIRISVacV + mThicknessIRISVacV, mThicknessIRISVacV/2.);
  TGeoTranslation* irisVacVWallPosZ = new TGeoTranslation("IRISVACVWELLPOSZ", 0., 0., mZLengthIRISVacV/2. + mThicknessIRISVacV/2.);
  irisVacVWallPosZ->RegisterYourself();
  TString irisCompositeFormula = "TRK_IRISVACUUMVESSEL_INNERTUBEsh"
                                  "+TRK_IRISVACUUMVESSEL_OUTERTUBEsh"
                                  "+TRK_IRISVACUUMVESSEL_WALLNEGZSIDEsh:IRISVACVWELLNEGZ"
                                  "+TRK_IRISVACUUMVESSEL_WALLPOSZSIDEsh:IRISVACVWELLPOSZ";
  TGeoCompositeShape* irisVacuumVesselComposite = new TGeoCompositeShape("TRK_IRISVACUUMVESSELsh", irisCompositeFormula);
  const TGeoMedium* medAl5083 = matmgr.getTGeoMedium("IRISVACUUMVESSEL_AL5083");
  TGeoVolume* irisVacuumVesselVolume = new TGeoVolume("TRK_IRISVACUUMVESSEL", irisVacuumVesselComposite, medAl5083);

  irisVacuumVesselVolume->SetVisibility(1);
  irisVacuumVesselVolume->SetLineColor(kYellow);

  LOGP(info, "Creating IRIS Tracker vacuum vessel");
  LOGP(info, "Inserting {} in {} ", irisVacuumVesselVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(irisVacuumVesselVolume, 1, nullptr);

}
// ClassImp(o2::trk::TRKServices);
} // namespace trk
} // namespace o2