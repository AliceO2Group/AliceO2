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

  matmgr.Mixture("ALICE3_TRKSERVICES", 66, "CERAMIC", aCer, zCer, dCer, 2, wCer);                                                      // Ceramic for cold plate
  matmgr.Mixture("ALICE3_TRKSERVICES", 68, "AIR", aAir, zAir, dAir, 4, wAir);                                                          // Air for placeholding cables
  matmgr.Mixture("ALICE3_TRKSERVICES", 69, "POLYETHYLENE", aPolyethylene, zPolyethylene, .95, 2, wPolyethylene);                       // Polyethylene for fibers
  matmgr.Mixture("ALICE3_TRKSERVICES", 70, "POLYURETHANE", aPolyurethane, zPolyurethane, dPolyurethane, nPolyurethane, wPolyurethane); // Polyurethane for cooling pipes
  matmgr.Mixture("ALICE3_TRKSERVICES", 71, "SILICONDIOXIDE", aSiO2, zSiO2, dSiO2, 2, wSiO2);                                           // Fused silica SiO2
  matmgr.Mixture("ALICE3_TRKSERVICES", 72, "WATER", aWater, zWater, dWater, 2, wWater);                                                // Water for cooling pipes
  matmgr.Material("ALICE3_TRKSERVICES", 67, "COPPER", 63.546, 29, 8.96, 1.43, 15.1);                                                   // Copper for cables

  // Danger zone: following mixtures do not use the interface of MaterialManager
  // TGeoMixture* fiber = new TGeoMixture("ALICE3_TRKSERVICES_FIBER", 2 /*nel*/);
  // fiber->AddElement(gGeoManager->GetMaterial("SILICONDIOXIDE"), 0.5);
  // fiber->AddElement(gGeoManager->GetMaterial("POLYETHYLENE"), 0.5);

  // TGeoMixture* powerBundleNoJacket = new TGeoMixture("ALICE3_TRKSERVICES_POWERBUNDLENOJACKET", 2 /*nel*/);
  // powerBundleNoJacket->AddElement(gGeoManager->GetMaterial("COPPER"), 0.09);
  // powerBundleNoJacket->AddElement(gGeoManager->GetMaterial("POLYETHYLENE"), 0.91);

  // TGeoMixture* coolingBundle = new TGeoMixture("ALICE3_TRKSERVICES_COOLINGBUNDLE", 2 /*nel*/);
  // coolingBundle->AddElement(gGeoManager->GetMaterial("POLYURETHANE"), 0.56);
  // coolingBundle->AddElement(gGeoManager->GetMaterial("WATER"), 0.44);

  matmgr.Medium("ALICE3_TRKSERVICES", 1, "CERAMIC", 66, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);        // Ceramic for cold plate
  matmgr.Medium("ALICE3_TRKSERVICES", 2, "COPPER", 67, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);         // Copper for cables
  matmgr.Medium("ALICE3_TRKSERVICES", 3, "AIR", 68, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);            // Air for placeholding cables
  matmgr.Medium("ALICE3_TRKSERVICES", 4, "POLYETHYLENE", 69, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);   // Polyethylene for fibers
  matmgr.Medium("ALICE3_TRKSERVICES", 5, "POLYURETHANE", 70, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);   // Polyurethane for cooling pipes
  matmgr.Medium("ALICE3_TRKSERVICES", 6, "SILICONDIOXIDE", 71, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin); // Fused silica SiO2
  matmgr.Medium("ALICE3_TRKSERVICES", 7, "WATER", 72, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);          // Water for cooling pipes
}

void TRKServices::createServices(TGeoVolume* motherVolume)
{
  createMaterials();
  createColdplate(motherVolume);
  // createCables(motherVolume);
  createMiddleServices(motherVolume);
  createOuterDisksServices(motherVolume);
}

void TRKServices::createColdplate(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCeramic = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_CERAMIC");

  TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", mColdPlateRMin, mColdPlateRMin + mColdPlateThickness, mColdPlateZLength / 2.);
  TGeoVolume* coldPlateVolume = new TGeoVolume("TRK_COLDPLATE", coldPlate, medCeramic);
  coldPlateVolume->SetVisibility(1);
  coldPlateVolume->SetLineColor(kYellow);

  LOGP(info, "Creating cold plate service");

  LOGP(info, "Inserting {} in {} ", coldPlateVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(coldPlateVolume, 1, nullptr);
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
    outerDisksFiberSIO2Volume->SetLineColor(kWhite);
    outerDisksFiberPEVolume->SetLineColor(kGreen);
    auto* combiTrans = new TGeoCombiTrans(0, 0, translation, nullptr);
    motherVolume->AddNode(outerDisksFiberSIO2Volume, 1, combiTrans);
    motherVolume->AddNode(outerDisksFiberPEVolume, 1, combiTrans);

    // Create power lines: 11.86mm
    float cuThick = 0.09 * 1.186;
    float pePowerThick = 0.91 * 1.186;

    TGeoTube* outerDisksPowerCu = new TGeoTube("TRK_OUTERDISKS_POWER_CUsh", rMinInnerServices, rMinInnerServices + cuThick, zLengthInnerServices / 2);
    TGeoTube* outerDisksPowerPE = new TGeoTube("TRK_OUTERDISKS_POWER_PEsh", rMinInnerServices + cuThick, rMinInnerServices + cuThick + pePowerThick, zLengthInnerServices / 2);
    rMinInnerServices += cuThick + pePowerThick;
    TGeoVolume* outerDisksPowerCuVolume = new TGeoVolume("TRK_OUTERDISKS_POWER_CU", outerDisksPowerCu, medCu);
    TGeoVolume* outerDisksPowerPEVolume = new TGeoVolume("TRK_OUTERDISKS_POWER_PE", outerDisksPowerPE, medPE);
    outerDisksPowerCuVolume->SetLineColor(kOrange);
    outerDisksPowerPEVolume->SetLineColor(kGreen);
    motherVolume->AddNode(outerDisksPowerCuVolume, 1, combiTrans);
    motherVolume->AddNode(outerDisksPowerPEVolume, 1, combiTrans);

    // Create cooling: 6.47mm
    float puThick = 0.56 * 0.647;
    float h2oThick = 0.44 * 0.647;

    TGeoTube* outerDisksCoolingPU = new TGeoTube("TRK_OUTERDISKS_COOLING_PUsh", rMinInnerServices, rMinInnerServices + puThick, zLengthInnerServices / 2);
    TGeoTube* outerDisksCoolingH2O = new TGeoTube("TRK_OUTERDISKS_COOLING_H2Osh", rMinInnerServices + puThick, rMinInnerServices + puThick + h2oThick, zLengthInnerServices / 2);
    // rMinInnerServices += puThick + h2oThick;
    TGeoVolume* outerDisksCoolingPUVolume = new TGeoVolume("TRK_OUTERDISKS_COOLING_PU", outerDisksCoolingPU, medPU);
    TGeoVolume* outerDisksCoolingH2OVolume = new TGeoVolume("TRK_OUTERDISKS_COOLING_H2O", outerDisksCoolingH2O, medH2O);
    outerDisksCoolingPUVolume->SetLineColor(kBlue);
    outerDisksCoolingH2OVolume->SetLineColor(kAzure);
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

  // Create fibers: 3.07mm, 50% SiO2, 50% PE
  float siO2FiberThick = 0.5 * 0.307;
  float peFiberThick = 0.5 * 0.307;

  // Get geometry information from TRK which is already present
  float rMinMiddleServices = 35.f; // ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmax();
  const float rMinMiddleBarrel = rMinMiddleServices;
  float zLengthMiddleServices = 141.f; // ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz();

  LOGP(info, "Building service disk for Middle Tracker rminMiddleServices is: {} Dz is {}", rMinMiddleServices, /* rMaxMiddleServices,*/ zLengthMiddleServices);
  TGeoTube* middleBarrelFiberSIO2 = new TGeoTube("TRK_MID_FIBER_SIO2sh", rMinMiddleServices, rMinMiddleServices + siO2FiberThick, zLengthMiddleServices);
  TGeoTube* middleBarrelFiberPE = new TGeoTube("TRK_MID_FIBER_PEsh", rMinMiddleServices + siO2FiberThick, rMinMiddleServices + siO2FiberThick + peFiberThick, zLengthMiddleServices);
  rMinMiddleServices = rMinMiddleServices + siO2FiberThick + peFiberThick;
  TGeoVolume* middleBarrelFiberSIO2Volume = new TGeoVolume("TRK_MID_FIBER_SIO2", middleBarrelFiberSIO2, medSiO2);
  TGeoVolume* middleBarrelFiberPEVolume = new TGeoVolume("TRK_MID_FIBER_PE", middleBarrelFiberPE, medPE);
  middleBarrelFiberSIO2Volume->SetLineColor(kWhite);
  middleBarrelFiberPEVolume->SetLineColor(kGreen);
  motherVolume->AddNode(middleBarrelFiberSIO2Volume, 1, nullptr);
  motherVolume->AddNode(middleBarrelFiberPEVolume, 1, nullptr);

  // Create powerlines: 10.9mm, 9% Cu, 91% PE
  float cuThick = 0.09 * 1.09;
  float pePowerThick = 0.91 * 1.09;

  TGeoTube* middleBarrelPowerCu = new TGeoTube("TRK_MID_POWER_CUsh", rMinMiddleServices, rMinMiddleServices + cuThick, zLengthMiddleServices);
  TGeoTube* middleBarrelPowerPE = new TGeoTube("TRK_MID_POWER_PEsh", rMinMiddleServices + cuThick, rMinMiddleServices + cuThick + pePowerThick, zLengthMiddleServices);
  rMinMiddleServices = rMinMiddleServices + cuThick + pePowerThick;
  TGeoVolume* middleBarrelPowerCuVolume = new TGeoVolume("TRK_MID_POWER_CU", middleBarrelPowerCu, medCu);
  TGeoVolume* middleBarrelPowerPEVolume = new TGeoVolume("TRK_MID_POWER_PE", middleBarrelPowerPE, medPE);
  middleBarrelPowerCuVolume->SetLineColor(kOrange);
  middleBarrelPowerPEVolume->SetLineColor(kGreen);
  motherVolume->AddNode(middleBarrelPowerCuVolume, 1, nullptr);
  motherVolume->AddNode(middleBarrelPowerPEVolume, 1, nullptr);

  // Create cooling pipes: 4.74mm, 56% PU, 44% H2O
  float puThick = 0.56 * 0.474;
  float h2oThick = 0.44 * 0.474;

  TGeoTube* middleBarrelCoolingPU = new TGeoTube("TRK_MID_COOLING_PUsh", rMinMiddleServices, rMinMiddleServices + puThick, zLengthMiddleServices);
  TGeoTube* middleBarrelCoolingH2O = new TGeoTube("TRK_MID_COOLING_H2Osh", rMinMiddleServices + puThick, rMinMiddleServices + puThick + h2oThick, zLengthMiddleServices);
  rMinMiddleServices = rMinMiddleServices + puThick + h2oThick;
  TGeoVolume* middleBarrelCoolingPUVolume = new TGeoVolume("TRK_MID_COOLING_PU", middleBarrelCoolingPU, medPU);
  TGeoVolume* middleBarrelCoolingH2OVolume = new TGeoVolume("TRK_MID_COOLING_H2O", middleBarrelCoolingH2O, medH2O);
  middleBarrelCoolingPUVolume->SetLineColor(kBlue);
  middleBarrelCoolingH2OVolume->SetLineColor(kAzure);
  motherVolume->AddNode(middleBarrelCoolingPUVolume, 1, nullptr);
  motherVolume->AddNode(middleBarrelCoolingH2OVolume, 1, nullptr);

  // Barrel to forward connection disks
  float rMaxMiddleServicesBarFwd = 74.5f + siO2FiberThick + peFiberThick + cuThick + pePowerThick + puThick + h2oThick;
  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    TGeoTube* middleBarFwdFiberSIO2 = new TGeoTube("TRK_MIDBARFWD_FIBER_SIO2sh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, siO2FiberThick);
    TGeoTube* middleBarFwdFiberPE = new TGeoTube("TRK_MIDBARFWD_FIBER_PEsh", rMinMiddleBarrel, rMaxMiddleServicesBarFwd, peFiberThick);
    TGeoVolume* middleBarFwdFiberSIO2Volume = new TGeoVolume("TRK_MIDBARFWD_FIBER_SIO2", middleBarFwdFiberSIO2, medSiO2);
    TGeoVolume* middleBarFwdFiberPEVolume = new TGeoVolume("TRK_MIDBARFWD_FIBER_PE", middleBarFwdFiberPE, medPE);
    middleBarFwdFiberSIO2Volume->SetLineColor(kWhite);
    middleBarFwdFiberPEVolume->SetLineColor(kGreen);
    auto* rot = new TGeoRotation("", 0, 0, 180);
    auto* combiTransSIO2 = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick / 2 + 141.f), rot);
    auto* combiTransPE = new TGeoCombiTrans(0, 0, (int)orientation * (siO2FiberThick + peFiberThick / 2 + 141.f), rot);
    motherVolume->AddNode(middleBarFwdFiberSIO2Volume, 1, combiTransSIO2);
    motherVolume->AddNode(middleBarFwdFiberPEVolume, 1, combiTransPE);
  }

  // Forward part
  float zLengthMiddleServicesFwd = 201.f; // 201cm

  for (auto& orientation : {Orientation::kASide, Orientation::kCSide}) {
    // Create fibers: 3.07mm, 50% SiO2, 50% PE
    float siO2FiberThick = 0.5 * 0.307;
    float peFiberThick = 0.5 * 0.307;
    float rMinMiddleServicesFwd = 74.5f; // 74.5cm

    float translation = (int)orientation * (149.f + zLengthMiddleServicesFwd / 2); // ±149cm

    TGeoTube* middleFwdFiberSIO2 = new TGeoTube("TRK_MIDFWD_FIBER_SIO2sh", rMinMiddleServicesFwd, rMinMiddleServicesFwd + siO2FiberThick, zLengthMiddleServicesFwd / 2);
    TGeoTube* middleFwdFiberPE = new TGeoTube("TRK_MIDFWD_FIBER_PEsh", rMinMiddleServicesFwd + siO2FiberThick, rMinMiddleServicesFwd + siO2FiberThick + peFiberThick, zLengthMiddleServicesFwd / 2);
    rMinMiddleServicesFwd += siO2FiberThick + peFiberThick;
    TGeoVolume* middleFwdFiberSIO2Volume = new TGeoVolume("TRK_MIDFWD_FIBER_SIO2", middleFwdFiberSIO2, medSiO2);
    TGeoVolume* middleFwdFiberPEVolume = new TGeoVolume("TRK_MIDFWD_FIBER_PE", middleFwdFiberPE, medPE);
    middleFwdFiberSIO2Volume->SetLineColor(kWhite);
    middleFwdFiberPEVolume->SetLineColor(kGreen);
    auto* combiTrans = new TGeoCombiTrans(0, 0, translation, nullptr);
    motherVolume->AddNode(middleFwdFiberSIO2Volume, 1, combiTrans);
    motherVolume->AddNode(middleFwdFiberPEVolume, 1, combiTrans);

    // Create powerlines: 10.9mm, 9% Cu, 91% PE
    float cuThick = 0.09 * 1.09;
    float pePowerThick = 0.91 * 1.09;

    TGeoTube* middleFwdPowerCu = new TGeoTube("TRK_MIDFWD_POWER_CUsh", rMinMiddleServicesFwd, rMinMiddleServicesFwd + cuThick, zLengthMiddleServicesFwd / 2);
    TGeoTube* middleFwdPowerPE = new TGeoTube("TRK_MIDFWD_POWER_PEsh", rMinMiddleServicesFwd + cuThick, rMinMiddleServicesFwd + cuThick + pePowerThick, zLengthMiddleServicesFwd / 2);
    rMinMiddleServicesFwd += cuThick + pePowerThick;
    TGeoVolume* middleFwdPowerCuVolume = new TGeoVolume("TRK_MIDFWD_POWER_CU", middleFwdPowerCu, medCu);
    TGeoVolume* middleFwdPowerPEVolume = new TGeoVolume("TRK_MIDFWD_POWER_PE", middleFwdPowerPE, medPE);
    middleFwdPowerCuVolume->SetLineColor(kOrange);
    middleFwdPowerPEVolume->SetLineColor(kGreen);
    motherVolume->AddNode(middleFwdPowerCuVolume, 1, combiTrans);
    motherVolume->AddNode(middleFwdPowerPEVolume, 1, combiTrans);

    // Create cooling pipes: 4.74mm, 56% PU, 44% H2O
    float puThick = 0.56 * 0.474;
    float h2oThick = 0.44 * 0.474;

    TGeoTube* middleFwdCoolingPU = new TGeoTube("TRK_MIDFWD_COOLING_PUsh", rMinMiddleServicesFwd, rMinMiddleServicesFwd + puThick, zLengthMiddleServicesFwd / 2);
    TGeoTube* middleFwdCoolingH2O = new TGeoTube("TRK_MIDFWD_COOLING_H2Osh", rMinMiddleServicesFwd + puThick, rMinMiddleServicesFwd + puThick + h2oThick, zLengthMiddleServicesFwd / 2);
    // rMinMiddleServicesFwd += puThick + h2oThick;
    TGeoVolume* middleFwdCoolingPUVolume = new TGeoVolume("TRK_MIDFWD_COOLING_PU", middleFwdCoolingPU, medPU);
    TGeoVolume* middleFwdCoolingH2OVolume = new TGeoVolume("TRK_MIDFWD_COOLING_H2O", middleFwdCoolingH2O, medH2O);
    middleFwdCoolingPUVolume->SetLineColor(kBlue);
    middleFwdCoolingH2OVolume->SetLineColor(kAzure);
    motherVolume->AddNode(middleFwdCoolingPUVolume, 1, combiTrans);
    motherVolume->AddNode(middleFwdCoolingH2OVolume, 1, combiTrans);
  }
}

void TRKServices::createOuterBarrelServices(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();

  // Fiber 0.269 cm
  // Cooling 1.432 cm
  // Power 0.430 cm
}

void TRKServices::createCables(TGeoVolume* motherVolume)
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCopper = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_COPPER");

  // Inner Tracker Services
  // Get geometry information from TRK which is already present
  float rMinInnerCables = ((TGeoTube*)motherVolume->GetNode(Form("%s3_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmin();
  float rMaxInnerCables = 35.0; // ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmax();
  float zLengthInnerCables = ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz();
  LOGP(info, "Building service disk for Inner Tracker rminInnerCables is: {} rmaxInnerCables is {} Dz is {}", rMinInnerCables, rMaxInnerCables, zLengthInnerCables);

  TGeoMedium* medAir = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_AIR");
  TGeoMedium* medCop = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_COPPER");

  for (size_t iCableFan{0}; iCableFan < mCableFanWeights.size(); ++iCableFan) {
    TGeoTube* cableFan = new TGeoTube("TRK_CABLEFAN_MIDsh", rMinInnerCables, rMaxInnerCables, mMiddleDiskThickness * mCableFanWeights[iCableFan]);
    TGeoVolume* cableFanVolume = new TGeoVolume("TRK_CABLEFAN_MID", cableFan, !(iCableFan % 2) ? medAir : medCop);
    cableFanVolume->SetLineColor(!(iCableFan % 2) ? kGray : kBlack);
    auto* rot = new TGeoRotation(Form("TRK_CABLEFAN_MID_ROT_%d", (int)iCableFan), 0, 0, 180);
    float incrShiftW = std::accumulate(mCableFanWeights.begin(), mCableFanWeights.begin() + iCableFan + 1, -mCableFanWeights[0]);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthInnerCables + 0.5 * mMiddleDiskThickness + incrShiftW * mMiddleDiskThickness, rot);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", cableFanVolume->GetName(), motherVolume->GetName(), zLengthInnerCables + incrShiftW * mMiddleDiskThickness, zLengthInnerCables, incrShiftW * mMiddleDiskThickness);
    motherVolume->AddNode(cableFanVolume, 1, combiTrans);
  }

  // Outer Tracker Services
  // Get geometry information from TRK which is already present
  float rMinOuterCables = 35.0 + mMiddleDiskThickness; // ((TGeoTube*)motherVolume->GetNode(Form("%s8_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmin();
  float rMaxOuterCables = ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetRmax();
  float zLengthOuterCables = ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz();
  LOGP(info, "Building service disk for Outer Tracker rminOuterCables is: {} rmaxOuterCables is {} Dz is {}", rMinOuterCables, rMaxOuterCables, zLengthOuterCables);

  for (size_t iCableFan{0}; iCableFan < mCableFanWeights.size(); ++iCableFan) {
    TGeoTube* cableFan = new TGeoTube("TRK_CABLEFAN_EXTsh", rMinOuterCables, rMaxOuterCables, mMiddleDiskThickness * mCableFanWeights[iCableFan]);
    TGeoVolume* cableFanVolume = new TGeoVolume("TRK_CABLEFAN_EXT", cableFan, !(iCableFan % 2) ? medAir : medCop);
    cableFanVolume->SetLineColor(!(iCableFan % 2) ? kGray : kBlack);
    auto* rot = new TGeoRotation(Form("TRK_CABLEFAN_EXT_ROT_%d", (int)iCableFan), 0, 0, 180);
    float incrShiftW = std::accumulate(mCableFanWeights.begin(), mCableFanWeights.begin() + iCableFan + 1, -mCableFanWeights[0]);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthOuterCables + 0.5 * mMiddleDiskThickness + incrShiftW * mMiddleDiskThickness, rot);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", cableFanVolume->GetName(), motherVolume->GetName(), zLengthOuterCables + incrShiftW * mMiddleDiskThickness, zLengthOuterCables, incrShiftW * mMiddleDiskThickness);
    motherVolume->AddNode(cableFanVolume, 1, combiTrans);
  }

  // FWD disks services, middle triplet
  // I put them here for convenience, ideally one would eventually have TRK and FT3 merged

  float rMinFwdCables = 35.0;
  float rMaxFwdCables = rMinFwdCables + mMiddleDiskThickness;
  float zLengthFwdCables = ((TGeoTube*)motherVolume->GetNode(Form("%s8_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz() -
                           ((TGeoTube*)motherVolume->GetNode(Form("%s7_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz() + mMiddleDiskThickness;

  LOGP(info, "Building service disk for FWD rminFwdCables is: {} rmaxFwdCables is {} Dz is {}", rMinFwdCables, rMaxFwdCables, zLengthFwdCables);
  for (size_t iCableLayer{0}; iCableLayer < mCableFanWeights.size(); ++iCableLayer) {
    TGeoTube* middleCableLayer = new TGeoTube("TRK_CABLELAYER_MIDsh", rMinFwdCables, rMaxFwdCables + mMiddleDiskThickness * mCableFanWeights[iCableLayer], zLengthFwdCables / 2.);
    TGeoVolume* middleCableLayerVolume = new TGeoVolume("TRK_CABLELAYER_MID", middleCableLayer, !(iCableLayer % 2) ? medAir : medCop);
    middleCableLayerVolume->SetLineColor(!(iCableLayer % 2) ? kGray : kBlack);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthInnerCables + (zLengthFwdCables) / 2, nullptr);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", middleCableLayerVolume->GetName(), motherVolume->GetName(), zLengthInnerCables + zLengthFwdCables / 2, zLengthInnerCables, zLengthFwdCables / 2);
    motherVolume->AddNode(middleCableLayerVolume, 1, combiTrans);
  }

  // FWD disks services, outer triplet
  // I put them here for convenience, ideally one would eventually have TRK and FT3 merged

  rMinFwdCables = rMaxOuterCables;
  rMaxFwdCables = rMinFwdCables + mMiddleDiskThickness;
  zLengthFwdCables = 400.0 - ((TGeoTube*)motherVolume->GetNode(Form("%s10_1", GeometryTGeo::getTRKLayerPattern()))->GetVolume()->GetShape())->GetDz() + mMiddleDiskThickness;

  LOGP(info, "Building service disk for FWD rminFwdCables is: {} rmaxFwdCables is {} Dz is {}", rMinFwdCables, rMaxFwdCables, zLengthFwdCables);
  for (size_t iCableLayer{0}; iCableLayer < mCableFanWeights.size(); ++iCableLayer) {
    TGeoTube* outerCableLayer = new TGeoTube("TRK_CABLELAYER_EXTsh", rMinFwdCables, rMaxFwdCables + mMiddleDiskThickness * mCableFanWeights[iCableLayer], zLengthFwdCables / 2.);
    TGeoVolume* outerCableLayerVolume = new TGeoVolume("TRK_CABLELAYER_EXT", outerCableLayer, !(iCableLayer % 2) ? medAir : medCop);
    outerCableLayerVolume->SetLineColor(!(iCableLayer % 2) ? kGray : kBlack);
    auto* combiTrans = new TGeoCombiTrans(0, 0, zLengthOuterCables + (zLengthFwdCables) / 2, nullptr);

    LOGP(info, "Inserting {} in {} at Z={} position which is {} + {} ", outerCableLayerVolume->GetName(), motherVolume->GetName(), zLengthOuterCables + zLengthFwdCables / 2, zLengthOuterCables, zLengthFwdCables / 2);
    motherVolume->AddNode(outerCableLayerVolume, 1, combiTrans);
  }
}
} // namespace trk
} // namespace o2