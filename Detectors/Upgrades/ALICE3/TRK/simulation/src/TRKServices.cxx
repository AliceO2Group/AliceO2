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
#include <TRKBase/GeometryTGeo.h>
#include <FT3Base/GeometryTGeo.h>
#include <TGeoVolume.h>
#include <TGeoNode.h>
#include <TGeoTube.h>
#include <TGeoCompositeShape.h>
#include <TColor.h>
#include <Rtypes.h>
#include <numeric>

#include <Framework/Logger.h>

namespace o2
{
namespace trk
{
TRKServices::TRKServices(float rMin, float zLength, float thickness)
{
  mColdPlateRMin = rMin;
  mColdPlateZLength = zLength;
  mColdPlateThickness = thickness;
  mZLengthIRISVacV = 70.;
  mThicknessIRISVacV = 150.e-4;
  mRInIRISVacV = 0.48;
  mROutIRISVacV = mColdPlateRMin + mColdPlateThickness;
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

  // Air
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  // Water
  float aWater[2] = {1.00794, 15.9994};
  float zWater[2] = {1., 8.};
  float wWater[2] = {0.111894, 0.888106};
  float dWater = 1.0;

  // Fused silica SiO2 https://pdg.lbl.gov/2023/AtomicNuclearProperties/HTML/silicon_dioxide_fused_quartz.html
  float aSiO2[2] = {28.0855, 15.9990};
  float zSiO2[2] = {14., 8.};
  float wSiO2[2] = {0.467, 0.533};
  float dSiO2 = 2.2;

  // Polyethylene from alice 2 absorber
  float aPolyethylene[2] = {12.01, 1.};
  float zPolyethylene[2] = {6., 1.};
  float wPolyethylene[2] = {.33, .67};

  // Polyurethane [HN-CO-O] from alice 2 mft
  int nPolyurethane = 4;
  float aPolyurethane[4] = {1.00794, 14.010, 12.0107, 15.9994};
  float zPolyurethane[4] = {1.0, 7.0, 6.0, 8.0};
  float wPolyurethane[4] = {0.017077588, 0.237314387, 0.203327619, 0.542280405};
  float dPolyurethane = 1.25;

  // Aluminium 5083 - alloy of Mn, Fe, Cu, Mg, Si, Zn, Cr, Ti, Al
  // Al5083 is considered as material for the iris vacuum vessel
  // https://www.smithmetal.com/5083.htm
  float aAl5083[9] = {54.938, 55.845, 63.546, 24.305, 28.086, 65.38, 51.996, 47.867, 26.982};
  float zAl5083[9] = {25., 26., 29., 12., 14., 30., 24., 22., 13.};
  // The concentration of certain metals in Al5083 have a range. What will be used for the alloy for the iris vacuum vessel will have to be checked
  float wAl5083[9] = {0.007, 0.004, 0.001, 0.0445, 0.004, 0.0025, 0.0015, 0.0015, 0.934};
  float dAl5083 = 2.650;

  // AlBeMet AM162H is a nanocomposite, not an alloy
  // Considered here as well https://indico.cern.ch/event/1168385/contributions/5355805/attachments/2681743/4652030/Jul%2010%201030-1045%20AM%20(Hawaii)%20M1Or1C-05%20AlBeMet.pdf
  float aAlBeMet[2] = {26.982, 9.012};
  float zAlBeMet[2] = {13., 4.};
  float wAlBeMet[2] = {0.38, 0.62};
  float dAlBeMet = 2.071;

  matmgr.Mixture("ALICE3_TRKSERVICES", 66, "CERAMIC", aCer, zCer, dCer, 2, wCer);                                                      // Ceramic for cold plate
  matmgr.Mixture("ALICE3_TRKSERVICES", 68, "AIR", aAir, zAir, dAir, 4, wAir);                                                          // Air for placeholding cables
  matmgr.Mixture("ALICE3_TRKSERVICES", 69, "POLYETHYLENE", aPolyethylene, zPolyethylene, .95, 2, wPolyethylene);                       // Polyethylene for fibers
  matmgr.Mixture("ALICE3_TRKSERVICES", 70, "POLYURETHANE", aPolyurethane, zPolyurethane, dPolyurethane, nPolyurethane, wPolyurethane); // Polyurethane for cooling pipes
  matmgr.Mixture("ALICE3_TRKSERVICES", 71, "SILICONDIOXIDE", aSiO2, zSiO2, dSiO2, 2, wSiO2);                                           // Fused silica SiO2
  matmgr.Mixture("ALICE3_TRKSERVICES", 72, "WATER", aWater, zWater, dWater, 2, wWater);                                                // Water for cooling pipes
  matmgr.Material("ALICE3_TRKSERVICES", 67, "COPPER", 63.546, 29, 8.96, 1.43, 15.1);                                                   // Copper for cables
  matmgr.Material("ALICE3_TRKSERVICES", 73, "BERYLLIUM", 9.01, 4., 1.848, 35.3, 36.7);                                                 // Beryllium - Candidate for IRIS vacuum vessel
  matmgr.Mixture("ALICE3_TRKSERVICES", 74, "ALUMINIUM5083", aAl5083, zAl5083, dAl5083, 9, wAl5083);                                    // AL5083 - Candidate for IRIS vacuum vessel
  matmgr.Mixture("ALICE3_TRKSERVICES", 75, "ALUMINIUMBERYLLIUMMETAL", aAlBeMet, zAlBeMet, dAlBeMet, 2, wAlBeMet);                      // Aluminium-Beryllium metal - Candidate for IRIS vacuum vessel
  matmgr.Material("ALICE3_TRKSERVICES", 76, "CARBONFIBERM55J6K", 12.0107, 6, 1.92, 999, 999);                                          // Carbon Fiber M55J

  matmgr.Medium("ALICE3_TRKSERVICES", 1, "CERAMIC", 66, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);                  // Ceramic for cold plate
  matmgr.Medium("ALICE3_TRKSERVICES", 2, "COPPER", 67, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);                   // Copper for cables
  matmgr.Medium("ALICE3_TRKSERVICES", 3, "AIR", 68, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);                      // Air for placeholding cables
  matmgr.Medium("ALICE3_TRKSERVICES", 4, "POLYETHYLENE", 69, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);             // Polyethylene for fibers
  matmgr.Medium("ALICE3_TRKSERVICES", 5, "POLYURETHANE", 70, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);             // Polyurethane for cooling pipes
  matmgr.Medium("ALICE3_TRKSERVICES", 6, "SILICONDIOXIDE", 71, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);           // Fused silica SiO2
  matmgr.Medium("ALICE3_TRKSERVICES", 7, "WATER", 72, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);                    // Water for cooling pipes
  matmgr.Medium("ALICE3_TRKSERVICES", 8, "BERYLLIUM", 73, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);                // Beryllium for IRIS vacuum vessel
  matmgr.Medium("ALICE3_TRKSERVICES", 9, "ALUMINIUM5083", 74, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);            // Al5083 for IRIS vacuum vessel
  matmgr.Medium("ALICE3_TRKSERVICES", 10, "ALUMINIUMBERYLLIUMMETAL", 75, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin); // AlBeMet for IRIS vacuum vessel
  matmgr.Medium("ALICE3_TRKSERVICES", 11, "CARBONFIBERM55J6K", 76, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);       // Carbon Fiber M55J
}

void TRKServices::createServices(TGeoVolume* motherVolume)
{
  createMaterials();
  createColdplate(motherVolume);
  createMiddleServices(motherVolume);
  createOuterDisksServices(motherVolume);
  createOuterBarrelServices(motherVolume);
}

void TRKServices::createColdplate(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCeramic = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_CERAMIC");

  TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", mColdPlateRMin, mColdPlateRMin + mColdPlateThickness, mColdPlateZLength / 2.);
  TGeoVolume* coldPlateVolume = new TGeoVolume("TRK_COLDPLATE", coldPlate, medCeramic);
  coldPlateVolume->SetVisibility(1);
  coldPlateVolume->SetLineColor(kGray);

  LOGP(info, "Creating cold plate service");

  LOGP(info, "Inserting {} in {} ", coldPlateVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(coldPlateVolume, 1, nullptr);

  // IRIS Tracker Vacuum Vessel
  TGeoTube* irisVacuumVesselInnerTube = new TGeoTube("TRK_IRISVACUUMVESSEL_INNERTUBEsh", mRInIRISVacV, mRInIRISVacV + mThicknessIRISVacV, mZLengthIRISVacV / 2.);
  TGeoTube* irisVacuumVesselOuterTube = new TGeoTube("TRK_IRISVACUUMVESSEL_OUTERTUBEsh", mROutIRISVacV, mROutIRISVacV + mThicknessIRISVacV, mZLengthIRISVacV / 2.);
  TGeoTube* irisVacuumVesselWall = new TGeoTube("TRK_IRISVACUUMVESSEL_WALLsh", mRInIRISVacV, mROutIRISVacV + mThicknessIRISVacV, mThicknessIRISVacV / 2.);
  TGeoTranslation* irisVacVWallNegZ = new TGeoTranslation("IRISVACVWALLNEGZ", 0., 0., -mZLengthIRISVacV / 2. - mThicknessIRISVacV / 2.);
  irisVacVWallNegZ->RegisterYourself();
  TGeoTranslation* irisVacVWallPosZ = new TGeoTranslation("IRISVACVWALLPOSZ", 0., 0., mZLengthIRISVacV / 2. + mThicknessIRISVacV / 2.);
  irisVacVWallPosZ->RegisterYourself();
  TString irisCompositeFormula =
    "TRK_IRISVACUUMVESSEL_INNERTUBEsh"
    "+TRK_IRISVACUUMVESSEL_OUTERTUBEsh"
    "+TRK_IRISVACUUMVESSEL_WALLsh:IRISVACVWALLNEGZ"
    "+TRK_IRISVACUUMVESSEL_WALLsh:IRISVACVWALLPOSZ";
  TGeoCompositeShape* irisVacuumVesselComposite = new TGeoCompositeShape("TRK_IRISVACUUMVESSELsh", irisCompositeFormula);

  const TGeoMedium* medBe = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_BERYLLIUM");
  TGeoVolume* irisVacuumVesselVolume = new TGeoVolume("TRK_IRISVACUUMVESSEL", irisVacuumVesselComposite, medBe);

  irisVacuumVesselVolume->SetVisibility(1);
  irisVacuumVesselVolume->SetLineColor(kGray);

  LOGP(info, "Creating IRIS Tracker vacuum vessel");
  LOGP(info, "Inserting {} in {} ", irisVacuumVesselVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(irisVacuumVesselVolume, 1, nullptr);
}

void TRKServices::createOuterDisksServices(TGeoVolume* motherVolume)
{
  // This method hardcoes the pink shape for the inner services
  auto& matmgr = o2::base::MaterialManager::Instance();

  TGeoMedium* medSiO2 = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_SILICONDIOXIDE");
  TGeoMedium* medPE = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_POLYETHYLENE");
  TGeoMedium* medCu = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_COPPER");
  TGeoMedium* medPU = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_POLYURETHANE");
  TGeoMedium* medH2O = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_WATER");

  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    // Create fibers: 2.12mm
    float siO2FiberThick = 0.5 * 0.212;
    float peFiberThick = 0.5 * 0.212;

    float rMinInnerServices = 68.5f;                                           // 68.5cm
    float zLengthInnerServices = 201.f;                                        // 201cm
    float translation = (int)orientation * (149.f + zLengthInnerServices / 2); // ±149cm

    TGeoTube* outerDisksFiberSIO2 = new TGeoTube("TRK_OUTERDISKS_FIBER_SIO2sh", rMinInnerServices, rMinInnerServices + siO2FiberThick, zLengthInnerServices / 2);
    TGeoTube* outerDisksFiberPE = new TGeoTube("TRK_OUTERDISKS_FIBER_PEsh", rMinInnerServices + siO2FiberThick, rMinInnerServices + siO2FiberThick + peFiberThick, zLengthInnerServices / 2);
    rMinInnerServices += siO2FiberThick + peFiberThick;
    TGeoVolume* outerDisksFiberSIO2Volume = new TGeoVolume("TRK_OUTERDISKS_FIBER_SIO2", outerDisksFiberSIO2, medSiO2);
    TGeoVolume* outerDisksFiberPEVolume = new TGeoVolume("TRK_OUTERDISKS_FIBER_PE", outerDisksFiberPE, medPE);
    outerDisksFiberSIO2Volume->SetLineColor(kGray);
    outerDisksFiberPEVolume->SetLineColor(kGray);
    auto* combiTrans = new TGeoCombiTrans(0, 0, translation, nullptr);
    motherVolume->AddNode(outerDisksFiberSIO2Volume, 1, combiTrans);
    motherVolume->AddNode(outerDisksFiberPEVolume, 1, combiTrans);

    // Create power lines: 11.86mm
    float cuPowerThick = 0.09 * 1.186;
    float pePowerThick = 0.91 * 1.186;

    TGeoTube* outerDisksPowerCu = new TGeoTube("TRK_OUTERDISKS_POWER_CUsh", rMinInnerServices, rMinInnerServices + cuPowerThick, zLengthInnerServices / 2);
    TGeoTube* outerDisksPowerPE = new TGeoTube("TRK_OUTERDISKS_POWER_PEsh", rMinInnerServices + cuPowerThick, rMinInnerServices + cuPowerThick + pePowerThick, zLengthInnerServices / 2);
    rMinInnerServices += cuPowerThick + pePowerThick;
    TGeoVolume* outerDisksPowerCuVolume = new TGeoVolume("TRK_OUTERDISKS_POWER_CU", outerDisksPowerCu, medCu);
    TGeoVolume* outerDisksPowerPEVolume = new TGeoVolume("TRK_OUTERDISKS_POWER_PE", outerDisksPowerPE, medPE);
    outerDisksPowerCuVolume->SetLineColor(kGray);
    outerDisksPowerPEVolume->SetLineColor(kGray);
    motherVolume->AddNode(outerDisksPowerCuVolume, 1, combiTrans);
    motherVolume->AddNode(outerDisksPowerPEVolume, 1, combiTrans);

    // Create cooling: 6.47mm
    float puCoolingThick = 0.56 * 0.647;
    float h2oCoolingThick = 0.44 * 0.647;

    TGeoTube* outerDisksCoolingPU = new TGeoTube("TRK_OUTERDISKS_COOLING_PUsh", rMinInnerServices, rMinInnerServices + puCoolingThick, zLengthInnerServices / 2);
    TGeoTube* outerDisksCoolingH2O = new TGeoTube("TRK_OUTERDISKS_COOLING_H2Osh", rMinInnerServices + puCoolingThick, rMinInnerServices + puCoolingThick + h2oCoolingThick, zLengthInnerServices / 2);
    // rMinInnerServices += puCoolingThick + h2oCoolingThick;
    TGeoVolume* outerDisksCoolingPUVolume = new TGeoVolume("TRK_OUTERDISKS_COOLING_PU", outerDisksCoolingPU, medPU);
    TGeoVolume* outerDisksCoolingH2OVolume = new TGeoVolume("TRK_OUTERDISKS_COOLING_H2O", outerDisksCoolingH2O, medH2O);
    outerDisksCoolingPUVolume->SetLineColor(kGray);
    outerDisksCoolingH2OVolume->SetLineColor(kGray);
    motherVolume->AddNode(outerDisksCoolingPUVolume, 1, combiTrans);
    motherVolume->AddNode(outerDisksCoolingH2OVolume, 1, combiTrans);
  }
}

void TRKServices::createMiddleServices(TGeoVolume* motherVolume)
{
  // This method hardcoes the yellow shape for the middle services
  auto& matmgr = o2::base::MaterialManager::Instance();

  TGeoMedium* medSiO2 = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_SILICONDIOXIDE");
  TGeoMedium* medPE = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_POLYETHYLENE");
  TGeoMedium* medCu = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_COPPER");
  TGeoMedium* medPU = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_POLYURETHANE");
  TGeoMedium* medH2O = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_WATER");
  TGeoMedium* medCFiber = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_CARBONFIBERM55J6K");

  // Create fibers: 3.07mm, 50% SiO2, 50% PE
  float siO2FiberThick = 0.5 * 0.307;
  float peFiberThick = 0.5 * 0.307;
  float puCoolingThick = 0.56 * 0.474;
  float h2oCoolingThick = 0.44 * 0.474;
  float cuPowerThick = 0.09 * 1.09;
  float pePowerThick = 0.91 * 1.09;
  const float totalThickness = siO2FiberThick + peFiberThick + cuPowerThick + pePowerThick + puCoolingThick + h2oCoolingThick;

  // Carbon Fiber Cylinder support for the middle tracker
  float rMinMiddleCarbonSupport = 34.8f; // Arbitrary value
  float rMaxMiddleCarbonSupport = 35.f;  // 2 mm of carbon fiber
  const float zLengthMiddleCarbon = 62.f;
  TGeoTube* middleBarrelCarbonSupport = new TGeoTube("TRK_MID_CARBONSUPPORTsh", rMinMiddleCarbonSupport, rMaxMiddleCarbonSupport, zLengthMiddleCarbon);
  TGeoVolume* middleBarrelCarbonSupportVolume = new TGeoVolume("TRK_MID_CARBONSUPPORT", middleBarrelCarbonSupport, medCFiber);
  middleBarrelCarbonSupportVolume->SetLineColor(kGray);
  LOGP(info, "Creating carbon fiber support for Middle Tracker");
  motherVolume->AddNode(middleBarrelCarbonSupportVolume, 1, nullptr);

  // Get geometry information from TRK which is already present
  float rMinMiddleServices = 35.f;
  float rMinMiddleBarrel = rMinMiddleServices;
  const float zLengthCylinderMiddleServices = 40.5f;
  const float zLengthMiddleServices = 143.f;
  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    rMinMiddleServices = 35.f;
    LOGP(info, "Building services for Middle Tracker rminMiddleServices");
    TGeoTube* middleBarrelFiberSIO2 = new TGeoTube(Form("TRK_MID_FIBER_SIO2sh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleServices, rMinMiddleServices + siO2FiberThick, zLengthCylinderMiddleServices /* + totalThickness*/);
    rMinMiddleServices += siO2FiberThick;
    TGeoTube* middleBarrelFiberPE = new TGeoTube(Form("TRK_MID_FIBER_PEsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleServices, rMinMiddleServices + peFiberThick, zLengthCylinderMiddleServices /* + totalThickness*/);
    rMinMiddleServices += peFiberThick;
    TGeoVolume* middleBarrelFiberSIO2Volume = new TGeoVolume(Form("TRK_MID_FIBER_SIO2_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelFiberSIO2, medSiO2);
    TGeoVolume* middleBarrelFiberPEVolume = new TGeoVolume(Form("TRK_MID_FIBER_PE_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelFiberPE, medPE);
    middleBarrelFiberSIO2Volume->SetLineColor(kGray);
    middleBarrelFiberPEVolume->SetLineColor(kGray);
    auto* combiTrans = new TGeoCombiTrans(0, 0, (int)orientation * (zLengthMiddleServices - zLengthCylinderMiddleServices), nullptr);
    motherVolume->AddNode(middleBarrelFiberSIO2Volume, 1, combiTrans);
    motherVolume->AddNode(middleBarrelFiberPEVolume, 1, combiTrans);

    // Create powerlines: 10.9mm, 9% Cu, 91% PE
    TGeoTube* middleBarrelPowerCu = new TGeoTube(Form("TRK_MID_POWER_CUsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleServices, rMinMiddleServices + cuPowerThick, zLengthCylinderMiddleServices /* + totalThickness*/);
    rMinMiddleServices += cuPowerThick;
    TGeoTube* middleBarrelPowerPE = new TGeoTube(Form("TRK_MID_POWER_PEsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleServices, rMinMiddleServices + pePowerThick, zLengthCylinderMiddleServices /* + totalThickness*/);
    rMinMiddleServices += pePowerThick;
    TGeoVolume* middleBarrelPowerCuVolume = new TGeoVolume(Form("TRK_MID_POWER_CU_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelPowerCu, medCu);
    TGeoVolume* middleBarrelPowerPEVolume = new TGeoVolume(Form("TRK_MID_POWER_PE_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelPowerPE, medPE);
    middleBarrelPowerCuVolume->SetLineColor(kGray);
    middleBarrelPowerPEVolume->SetLineColor(kGray);
    motherVolume->AddNode(middleBarrelPowerCuVolume, 1, combiTrans);
    motherVolume->AddNode(middleBarrelPowerPEVolume, 1, combiTrans);

    // Create cooling pipes: 4.74mm, 56% PU, 44% H2O
    TGeoTube* middleBarrelCoolingPU = new TGeoTube(Form("TRK_MID_COOLING_PUsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleServices, rMinMiddleServices + puCoolingThick, zLengthCylinderMiddleServices /* + totalThickness*/);
    rMinMiddleServices += puCoolingThick;
    TGeoTube* middleBarrelCoolingH2O = new TGeoTube(Form("TRK_MID_COOLING_H2Osh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleServices, rMinMiddleServices + h2oCoolingThick, zLengthCylinderMiddleServices /* + totalThickness*/);
    rMinMiddleServices = rMinMiddleServices += h2oCoolingThick;
    TGeoVolume* middleBarrelCoolingPUVolume = new TGeoVolume(Form("TRK_MID_COOLING_PU_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelCoolingPU, medPU);
    TGeoVolume* middleBarrelCoolingH2OVolume = new TGeoVolume(Form("TRK_MID_COOLING_H2O_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelCoolingH2O, medH2O);
    middleBarrelCoolingPUVolume->SetLineColor(kGray);
    middleBarrelCoolingH2OVolume->SetLineColor(kGray);
    motherVolume->AddNode(middleBarrelCoolingPUVolume, 1, combiTrans);
    motherVolume->AddNode(middleBarrelCoolingH2OVolume, 1, combiTrans);
  }
  // Middle barrel connection disks
  const float rMinMiddleBarrelDisk = 5.68f;
  const float rMaxMiddleBarrelDisk = 35.f;
  const float zLengthMiddleBarrel = 62.f;
  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    TGeoTube* middleBarrelConnDiskSIO2 = new TGeoTube(Form("TRK_MIDBARCONN_DISK_SIO2sh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleBarrelDisk, rMaxMiddleBarrelDisk, siO2FiberThick);
    TGeoTube* middleBarrelConnDiskPE = new TGeoTube(Form("TRK_MIDBARCONN_DISK_PEsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleBarrelDisk, rMaxMiddleBarrelDisk, peFiberThick);
    TGeoVolume* middleBarrelConnDiskSIO2Volume = new TGeoVolume(Form("TRK_MIDBARCONN_DISK_SIO2_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelConnDiskSIO2, medSiO2);
    TGeoVolume* middleBarrelConnDiskPEVolume = new TGeoVolume(Form("TRK_MIDBARCONN_DISK_PE_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelConnDiskPE, medPE);
    middleBarrelConnDiskSIO2Volume->SetLineColor(kGray);
    middleBarrelConnDiskPEVolume->SetLineColor(kGray);
    auto* rot = new TGeoRotation("", 0, 0, 180);
    auto* combiTransSIO2 = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick / 2 + zLengthMiddleBarrel), rot);
    auto* combiTransPE = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick / 2 + zLengthMiddleBarrel), rot);
    motherVolume->AddNode(middleBarrelConnDiskSIO2Volume, 1, combiTransSIO2);
    motherVolume->AddNode(middleBarrelConnDiskPEVolume, 1, combiTransPE);

    TGeoTube* middleBarrelConnDiskCu = new TGeoTube(Form("TRK_MIDBARCONN_DISK_CUsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleBarrelDisk, rMaxMiddleBarrelDisk, cuPowerThick);
    TGeoTube* middleBarrelConnDiskPEPower = new TGeoTube(Form("TRK_MIDBARCONN_DISK_PEsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleBarrelDisk, rMaxMiddleBarrelDisk, pePowerThick);
    TGeoVolume* middleBarrelConnDiskCuVolume = new TGeoVolume(Form("TRK_MIDBARCONN_DISK_CU_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelConnDiskCu, medCu);
    TGeoVolume* middleBarrelConnDiskPEPowerVolume = new TGeoVolume(Form("TRK_MIDBARCONN_DISK_PE_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelConnDiskPEPower, medPE);
    middleBarrelConnDiskCuVolume->SetLineColor(kGray);
    middleBarrelConnDiskPEPowerVolume->SetLineColor(kGray);
    auto* combiTransCu = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick + cuPowerThick / 2 + zLengthMiddleBarrel), rot);
    auto* combiTransPEPower = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick + cuPowerThick + pePowerThick / 2 + zLengthMiddleBarrel), rot);
    motherVolume->AddNode(middleBarrelConnDiskCuVolume, 1, combiTransCu);
    motherVolume->AddNode(middleBarrelConnDiskPEPowerVolume, 1, combiTransPEPower);

    TGeoTube* middleBarrelConnDiskPU = new TGeoTube(Form("TRK_MIDBARCONN_DISK_PUsh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleBarrelDisk, rMaxMiddleBarrelDisk, puCoolingThick);
    TGeoTube* middleBarrelConnDiskH2O = new TGeoTube(Form("TRK_MIDBARCONN_DISK_H2Osh_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), rMinMiddleBarrelDisk, rMaxMiddleBarrelDisk, h2oCoolingThick);
    TGeoVolume* middleBarrelConnDiskPUVolume = new TGeoVolume(Form("TRK_MIDBARCONN_DISK_PU_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelConnDiskPU, medPU);
    TGeoVolume* middleBarrelConnDiskH2OVolume = new TGeoVolume(Form("TRK_MIDBARCONN_DISK_H2O_%s", orientation == Orientation::kASide ? "bwd" : "fwd"), middleBarrelConnDiskH2O, medH2O);
    middleBarrelConnDiskPUVolume->SetLineColor(kGray);
    middleBarrelConnDiskH2OVolume->SetLineColor(kGray);
    motherVolume->AddNode(middleBarrelConnDiskPUVolume, 1, combiTransCu);
    motherVolume->AddNode(middleBarrelConnDiskH2OVolume, 1, combiTransPEPower);
  }

  // Barrel to forward connection disks
  float rMaxMiddleServicesBarFwd = 74.5f + siO2FiberThick + peFiberThick + cuPowerThick + pePowerThick + puCoolingThick + h2oCoolingThick;
  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    // Create fibers: 3.07mm, 50% SiO2, 50% PE
    TGeoTube* middleBarFwdFiberSIO2 = new TGeoTube("TRK_MIDBARFWD_FIBER_SIO2sh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, siO2FiberThick);
    TGeoTube* middleBarFwdFiberPE = new TGeoTube("TRK_MIDBARFWD_FIBER_PEsh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, peFiberThick);
    TGeoVolume* middleBarFwdFiberSIO2Volume = new TGeoVolume("TRK_MIDBARFWD_FIBER_SIO2", middleBarFwdFiberSIO2, medSiO2);
    TGeoVolume* middleBarFwdFiberPEVolume = new TGeoVolume("TRK_MIDBARFWD_FIBER_PE", middleBarFwdFiberPE, medPE);
    middleBarFwdFiberSIO2Volume->SetLineColor(kGray);
    middleBarFwdFiberPEVolume->SetLineColor(kGray);
    auto* rot = new TGeoRotation("", 0, 0, 180);
    auto* combiTransSIO2 = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick / 2 + zLengthMiddleServices), rot);
    auto* combiTransPE = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick / 2 + zLengthMiddleServices), rot);
    motherVolume->AddNode(middleBarFwdFiberSIO2Volume, 1, combiTransSIO2);
    motherVolume->AddNode(middleBarFwdFiberPEVolume, 1, combiTransPE);

    // Create powerlines: 10.9mm, 9% Cu, 91% PE
    TGeoTube* middleBarFwdPowerCu = new TGeoTube("TRK_MIDBARFWD_POWER_CUsh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, cuPowerThick);
    TGeoTube* middleBarFwdPowerPE = new TGeoTube("TRK_MIDBARFWD_POWER_PEsh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, pePowerThick);
    TGeoVolume* middleBarFwdPowerCuVolume = new TGeoVolume("TRK_MIDBARFWD_POWER_CU", middleBarFwdPowerCu, medCu);
    TGeoVolume* middleBarFwdPowerPEVolume = new TGeoVolume("TRK_MIDBARFWD_POWER_PE", middleBarFwdPowerPE, medPE);
    middleBarFwdPowerCuVolume->SetLineColor(kGray);
    middleBarFwdPowerPEVolume->SetLineColor(kGray);
    auto* combiTransCu = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick + cuPowerThick / 2 + zLengthMiddleServices), rot);
    auto* combiTransPEPower = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick + cuPowerThick + pePowerThick / 2 + zLengthMiddleServices), rot);
    motherVolume->AddNode(middleBarFwdPowerCuVolume, 1, combiTransCu);
    motherVolume->AddNode(middleBarFwdPowerPEVolume, 1, combiTransPEPower);

    // Create cooling pipes: 4.74mm, 56% PU, 44% H2O
    TGeoTube* middleBarFwdCoolingPU = new TGeoTube("TRK_MIDBARFWD_COOLING_PUsh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, puCoolingThick);
    TGeoTube* middleBarFwdCoolingH2O = new TGeoTube("TRK_MIDBARFWD_COOLING_H2Osh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, h2oCoolingThick);
    TGeoVolume* middleBarFwdCoolingPUVolume = new TGeoVolume("TRK_MIDBARFWD_COOLING_PU", middleBarFwdCoolingPU, medPU);
    TGeoVolume* middleBarFwdCoolingH2OVolume = new TGeoVolume("TRK_MIDBARFWD_COOLING_H2O", middleBarFwdCoolingH2O, medH2O);
    middleBarFwdCoolingPUVolume->SetLineColor(kGray);
    middleBarFwdCoolingH2OVolume->SetLineColor(kGray);
    auto* combiTransCoolingPU = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick + cuPowerThick + pePowerThick + puCoolingThick / 2 + zLengthMiddleServices), rot);
    auto* combiTransCoolingH2O = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick + cuPowerThick + pePowerThick + puCoolingThick + h2oCoolingThick / 2 + zLengthMiddleServices), rot);
    motherVolume->AddNode(middleBarFwdCoolingPUVolume, 1, combiTransCoolingPU);
    motherVolume->AddNode(middleBarFwdCoolingH2OVolume, 1, combiTransCoolingH2O);
  }

  // Forward part
  const float zLengthMiddleServicesFwd = 350.f - (143.f + totalThickness);

  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    // Create fibers: 3.07mm, 50% SiO2, 50% PE
    float siO2FiberThick = 0.5 * 0.307;
    float peFiberThick = 0.5 * 0.307;
    float rMinMiddleServicesFwd = 74.5f; // 74.5cm

    float translation = (int)orientation * (143.f + totalThickness + zLengthMiddleServicesFwd / 2);

    TGeoTube* middleFwdFiberSIO2 = new TGeoTube("TRK_MIDFWD_FIBER_SIO2sh", rMinMiddleServicesFwd, rMinMiddleServicesFwd + siO2FiberThick, zLengthMiddleServicesFwd / 2);
    TGeoTube* middleFwdFiberPE = new TGeoTube("TRK_MIDFWD_FIBER_PEsh", rMinMiddleServicesFwd + siO2FiberThick, rMinMiddleServicesFwd + siO2FiberThick + peFiberThick, zLengthMiddleServicesFwd / 2);
    rMinMiddleServicesFwd += siO2FiberThick + peFiberThick;
    TGeoVolume* middleFwdFiberSIO2Volume = new TGeoVolume("TRK_MIDFWD_FIBER_SIO2", middleFwdFiberSIO2, medSiO2);
    TGeoVolume* middleFwdFiberPEVolume = new TGeoVolume("TRK_MIDFWD_FIBER_PE", middleFwdFiberPE, medPE);
    middleFwdFiberSIO2Volume->SetLineColor(kGray);
    middleFwdFiberPEVolume->SetLineColor(kGray);
    auto* combiTrans = new TGeoCombiTrans(0, 0, translation, nullptr);
    motherVolume->AddNode(middleFwdFiberSIO2Volume, 1, combiTrans);
    motherVolume->AddNode(middleFwdFiberPEVolume, 1, combiTrans);

    // Create powerlines: 10.9mm, 9% Cu, 91% PE
    float cuPowerThick = 0.09 * 1.09;
    float pePowerThick = 0.91 * 1.09;

    TGeoTube* middleFwdPowerCu = new TGeoTube("TRK_MIDFWD_POWER_CUsh", rMinMiddleServicesFwd, rMinMiddleServicesFwd + cuPowerThick, zLengthMiddleServicesFwd / 2);
    TGeoTube* middleFwdPowerPE = new TGeoTube("TRK_MIDFWD_POWER_PEsh", rMinMiddleServicesFwd + cuPowerThick, rMinMiddleServicesFwd + cuPowerThick + pePowerThick, zLengthMiddleServicesFwd / 2);
    rMinMiddleServicesFwd += cuPowerThick + pePowerThick;
    TGeoVolume* middleFwdPowerCuVolume = new TGeoVolume("TRK_MIDFWD_POWER_CU", middleFwdPowerCu, medCu);
    TGeoVolume* middleFwdPowerPEVolume = new TGeoVolume("TRK_MIDFWD_POWER_PE", middleFwdPowerPE, medPE);
    middleFwdPowerCuVolume->SetLineColor(kGray);
    middleFwdPowerPEVolume->SetLineColor(kGray);
    motherVolume->AddNode(middleFwdPowerCuVolume, 1, combiTrans);
    motherVolume->AddNode(middleFwdPowerPEVolume, 1, combiTrans);

    // Create cooling pipes: 4.74mm, 56% PU, 44% H2O
    float puCoolingThick = 0.56 * 0.474;
    float h2oCoolingThick = 0.44 * 0.474;

    TGeoTube* middleFwdCoolingPU = new TGeoTube("TRK_MIDFWD_COOLING_PUsh", rMinMiddleServicesFwd, rMinMiddleServicesFwd + puCoolingThick, zLengthMiddleServicesFwd / 2);
    TGeoTube* middleFwdCoolingH2O = new TGeoTube("TRK_MIDFWD_COOLING_H2Osh", rMinMiddleServicesFwd + puCoolingThick, rMinMiddleServicesFwd + puCoolingThick + h2oCoolingThick, zLengthMiddleServicesFwd / 2);
    // rMinMiddleServicesFwd += puCoolingThick + h2oCoolingThick;
    TGeoVolume* middleFwdCoolingPUVolume = new TGeoVolume("TRK_MIDFWD_COOLING_PU", middleFwdCoolingPU, medPU);
    TGeoVolume* middleFwdCoolingH2OVolume = new TGeoVolume("TRK_MIDFWD_COOLING_H2O", middleFwdCoolingH2O, medH2O);
    middleFwdCoolingPUVolume->SetLineColor(kGray);
    middleFwdCoolingH2OVolume->SetLineColor(kGray);
    motherVolume->AddNode(middleFwdCoolingPUVolume, 1, combiTrans);
    motherVolume->AddNode(middleFwdCoolingH2OVolume, 1, combiTrans);
  }
}

void TRKServices::createOuterBarrelServices(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();

  TGeoMedium* medSiO2 = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_SILICONDIOXIDE");
  TGeoMedium* medPE = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_POLYETHYLENE");
  TGeoMedium* medCu = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_COPPER");
  TGeoMedium* medPU = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_POLYURETHANE");
  TGeoMedium* medH2O = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_WATER");

  // Fiber 0.269 cm
  const float siO2FiberThick = 0.5 * 0.269;
  const float peFiberThick = 0.5 * 0.269;
  float rMinOuterBarrelServices = ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmax();
  const float zLengthOuterBarrelServices = 350.f; // 175cm

  TGeoTube* outerBarrelFiberSIO2 = new TGeoTube("TRK_OUTERBARREL_FIBER_SIO2sh", rMinOuterBarrelServices, rMinOuterBarrelServices + siO2FiberThick, zLengthOuterBarrelServices);
  TGeoTube* outerBarrelFiberPE = new TGeoTube("TRK_OUTERBARREL_FIBER_PEsh", rMinOuterBarrelServices + siO2FiberThick, rMinOuterBarrelServices + siO2FiberThick + peFiberThick, zLengthOuterBarrelServices);
  rMinOuterBarrelServices += siO2FiberThick + peFiberThick;
  TGeoVolume* outerBarrelFiberSIO2Volume = new TGeoVolume("TRK_OUTERBARREL_FIBER_SIO2", outerBarrelFiberSIO2, medSiO2);
  TGeoVolume* outerBarrelFiberPEVolume = new TGeoVolume("TRK_OUTERBARREL_FIBER_PE", outerBarrelFiberPE, medPE);
  outerBarrelFiberSIO2Volume->SetLineColor(kGray);
  outerBarrelFiberPEVolume->SetLineColor(kGray);
  motherVolume->AddNode(outerBarrelFiberSIO2Volume, 1, nullptr);
  motherVolume->AddNode(outerBarrelFiberPEVolume, 1, nullptr);

  // Power 0.430 cm
  const float cuPowerThick = 0.09 * 0.430;
  const float pePowerThick = 0.91 * 0.430;

  TGeoTube* outerBarrelPowerCu = new TGeoTube("TRK_OUTERBARREL_POWER_CUsh", rMinOuterBarrelServices, rMinOuterBarrelServices + cuPowerThick, zLengthOuterBarrelServices);
  TGeoTube* outerBarrelPowerPE = new TGeoTube("TRK_OUTERBARREL_POWER_PEsh", rMinOuterBarrelServices + cuPowerThick, rMinOuterBarrelServices + cuPowerThick + pePowerThick, zLengthOuterBarrelServices);
  rMinOuterBarrelServices += cuPowerThick + pePowerThick;
  TGeoVolume* outerBarrelPowerCuVolume = new TGeoVolume("TRK_OUTERBARREL_POWER_CU", outerBarrelPowerCu, medCu);
  TGeoVolume* outerBarrelPowerPEVolume = new TGeoVolume("TRK_OUTERBARREL_POWER_PE", outerBarrelPowerPE, medPE);
  outerBarrelPowerCuVolume->SetLineColor(kGray);
  outerBarrelPowerPEVolume->SetLineColor(kGray);
  motherVolume->AddNode(outerBarrelPowerCuVolume, 1, nullptr);
  motherVolume->AddNode(outerBarrelPowerPEVolume, 1, nullptr);

  // Cooling 1.432 cm
  const float puCoolingThick = 0.56 * 1.432;
  const float h2oCoolingThick = 0.44 * 1.432;

  TGeoTube* outerBarrelCoolingPU = new TGeoTube("TRK_OUTERBARREL_COOLING_PUsh", rMinOuterBarrelServices, rMinOuterBarrelServices + puCoolingThick, zLengthOuterBarrelServices);
  TGeoTube* outerBarrelCoolingH2O = new TGeoTube("TRK_OUTERBARREL_COOLING_H2Osh", rMinOuterBarrelServices + puCoolingThick, rMinOuterBarrelServices + puCoolingThick + h2oCoolingThick, zLengthOuterBarrelServices);
  // rMinOuterBarrelServices += puCoolingThick + h2oCoolingThick;
  TGeoVolume* outerBarrelCoolingPUVolume = new TGeoVolume("TRK_OUTERBARREL_COOLING_PU", outerBarrelCoolingPU, medPU);
  TGeoVolume* outerBarrelCoolingH2OVolume = new TGeoVolume("TRK_OUTERBARREL_COOLING_H2O", outerBarrelCoolingH2O, medH2O);
  outerBarrelCoolingPUVolume->SetLineColor(kGray);
  outerBarrelCoolingH2OVolume->SetLineColor(kGray);
  motherVolume->AddNode(outerBarrelCoolingPUVolume, 1, nullptr);
  motherVolume->AddNode(outerBarrelCoolingH2OVolume, 1, nullptr);
}
} // namespace trk
} // namespace o2