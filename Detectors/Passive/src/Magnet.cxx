// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// ----- main responsible: Sandro Wenzel (sandro.wenzel@cern.ch)       -----
// -------------------------------------------------------------------------

#include <DetectorsBase/Detector.h>
#include <DetectorsBase/MaterialManager.h>
#include <DetectorsPassive/Magnet.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoPgon.h>
#include <TGeoVolume.h>
#include <TGeoXtru.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

Magnet::~Magnet() = default;

Magnet::Magnet() : FairModule("Magnet", "") {}
Magnet::Magnet(const char* name, const char* Title) : FairModule(name, Title) {}
Magnet::Magnet(const Magnet& rhs) = default;

Magnet& Magnet::operator=(const Magnet& rhs)
{
  // self assignment
  if (this == &rhs)
    return *this;

  // base class assignment
  FairModule::operator=(rhs);

  return *this;
}

void Magnet::createMaterials()
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  //
  // Create materials for L3 magnet
  //
  Int_t isxfld = 2.;
  Float_t sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);
  Float_t epsil, stmin, deemax, tmaxfd, stemax;

  // --- Define the various materials for GEANT ---
  // Steel
  Float_t asteel[4] = {55.847, 51.9961, 58.6934, 28.0855};
  Float_t zsteel[4] = {26., 24., 28., 14.};
  Float_t wsteel[4] = {.715, .18, .1, .005};
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;
  Float_t aWater[2] = {1.00794, 15.9994};
  Float_t zWater[2] = {1., 8.};
  Float_t wWater[2] = {0.111894, 0.888106};

  //     Aluminum
  matmgr.Material("MAG", 9, "Al0$", 26.98, 13., 2.7, 8.9, 37.2);
  matmgr.Material("MAG", 29, "Al1$", 26.98, 13., 2.7, 8.9, 37.2);

  //     Stainless Steel
  matmgr.Mixture("MAG", 19, "STAINLESS STEEL1", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("MAG", 39, "STAINLESS STEEL2", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("MAG", 59, "STAINLESS STEEL3", asteel, zsteel, 7.88, 4, wsteel);

  //     Iron
  matmgr.Material("MAG", 10, "Fe0$", 55.85, 26., 7.87, 1.76, 17.1);
  matmgr.Material("MAG", 30, "Fe1$", 55.85, 26., 7.87, 1.76, 17.1);

  //     Air
  matmgr.Mixture("MAG", 15, "AIR0$", aAir, zAir, dAir, 4, wAir);
  matmgr.Mixture("MAG", 35, "AIR1$", aAir, zAir, dAir, 4, wAir);

  //     Water
  matmgr.Mixture("MAG", 16, "WATER", aWater, zWater, 1., 2, wWater);

  // ****************
  //     Defines tracking media parameters.
  //     Les valeurs sont commentees pour laisser le defaut
  //     a GEANT (version 3-21, page CONS200), f.m.
  epsil = .001;  // Tracking precision,
  stemax = -1.;  // Maximum displacement for multiple scat
  tmaxfd = -20.; // Maximum angle due to field deflection
  deemax = -.3;  // Maximum fractional energy loss, DLS
  stmin = -.8;
  // ***************

  //    IRON
  matmgr.Medium("MAG", 10, "FE_C0", 10, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("MAG", 30, "FE_C1", 30, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //     ALUMINUM
  matmgr.Medium("MAG", 9, "ALU_C0", 9, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("MAG", 29, "ALU_C1", 29, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //     AIR
  matmgr.Medium("MAG", 15, "AIR_C0", 15, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("MAG", 35, "AIR_C1", 35, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Steel
  matmgr.Medium("MAG", 19, "ST_C0", 19, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("MAG", 39, "ST_C1", 39, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("MAG", 59, "ST_C3", 59, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  //    WATER
  matmgr.Medium("MAG", 16, "WATER", 16, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

void Magnet::ConstructGeometry()
{
  createMaterials();

  // Octagon
  const Int_t kNSides = 8;
  const Float_t kStartAngle = 22.5;            // deg
  const Float_t kFullAngle = 360.0;            // deg
                                               //  Mother volume
  const Float_t kRBMotherInner = 600.00;       // cm
  const Float_t kRBMotherOuter = 790.50;       // cm
  const Float_t kLBMother = 706.00;            // cm
                                               // Yoke
  const Float_t kRYokeInner = 703.50;          // cm
  const Float_t kRYokeOuter = 790.50;          // cm
  const Float_t kLYoke = 620.00;               // cm
                                               // Coil
  const Float_t kRCoilInner = 593.00;          // cm
  const Float_t kRCoilOuter = 682.00;          // cm
  const Float_t kLCoil = 588.00;               // cm
                                               // Cooling
  const Float_t kRCoolingOuter = 1.70;         // cm
  const Float_t kRCoolingInner = 1.00;         // cm
                                               // Thermal Shield
  const Float_t kRThermalShieldInner = 566.00; // cm
  const Float_t kRThermalShieldOuter = 571.00; // cm
                                               // Crown
  const Float_t kRCrownInner = 600.00;         // cm
  const Float_t kRCrownOuter = 785.50;         // cm
  const Float_t kLCrown1 = 605.00;             // cm
  const Float_t kLCrown2 = 620.00;             // cm
  const Float_t kLCrown3 = 706.00;             // cm
                                               // Door
  const Float_t kRDoorOuter = 600.00;          // cm
  const Float_t kRPlugInner = 183.50;          // cm
  const Float_t kLDoor1 = 615.50;              // cm
  const Float_t kLDoor2 = 714.60;              // cm
                                               //
  const Float_t kDegRad = TMath::Pi() / 180.;

  //
  // Top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  assert(top);

  // Media
  auto& matmgr = o2::base::MaterialManager::Instance();
  auto medAir = matmgr.getTGeoMedium("MAG_AIR_C1");
  auto medAlu = matmgr.getTGeoMedium("MAG_ALU_C1");
  auto medAluI = matmgr.getTGeoMedium("MAG_ALU_C0");
  auto medSteel = matmgr.getTGeoMedium("MAG_ST_C1");
  auto medWater = matmgr.getTGeoMedium("MAG_WATER");
  //
  // Offset between LHC and LEP axis
  Float_t os = -30.;

  //
  //  Define Barrel Mother
  //
  TGeoPgon* shBMother = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 2);
  shBMother->DefineSection(0, -kLBMother, kRBMotherInner, kRBMotherOuter);
  shBMother->DefineSection(1, kLBMother, kRBMotherInner, kRBMotherOuter);
  //
  TGeoVolumeAssembly* voBMother = new TGeoVolumeAssembly("L3BM");
  //
  // Define Thermal Shield
  //
  // Only one layer
  // This can be improved: replace by (protection - shield - insulation) !
  //
  TGeoPgon* shThermSh = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 2);
  shThermSh->DefineSection(0, -kLCoil, kRThermalShieldInner, kRThermalShieldOuter);
  shThermSh->DefineSection(1, kLCoil, kRThermalShieldInner, kRThermalShieldOuter);
  //
  TGeoVolume* voThermSh = new TGeoVolume("L3TS", shThermSh, medAluI);
  voBMother->AddNode(voThermSh, 1, new TGeoTranslation(0., 0., 0.));
  //
  // Define Coils and cooling circuits
  //
  TGeoPgon* shCoilMother = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 2);
  shCoilMother->DefineSection(0, -kLCoil, kRCoilInner - 2. * kRCoolingOuter, kRCoilOuter + 2. * kRCoolingOuter);
  shCoilMother->DefineSection(1, kLCoil, kRCoilInner - 2. * kRCoolingOuter, kRCoilOuter + 2. * kRCoolingOuter);
  //
  // Coils
  TGeoVolume* voCoilMother = new TGeoVolume("L3CM", shCoilMother, medAir);
  voBMother->AddNode(voCoilMother, 1, new TGeoTranslation(0., 0., 0.));
  // Divide into the 168 turns
  TGeoVolume* voCoilTurn = voCoilMother->Divide("L3CD", 3, 168, 0., 0.);
  TGeoPgon* shCoils = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 2);
  shCoils->DefineSection(0, -3., kRCoilInner, kRCoilOuter);
  shCoils->DefineSection(1, 3., kRCoilInner, kRCoilOuter);
  //
  TGeoVolume* voCoils = new TGeoVolume("L3C0", shCoils, medAlu);
  voCoilTurn->AddNode(voCoils, 1, new TGeoTranslation(0., 0., 0.));
  //
  // Hexagonal Cooling circuits
  //
  const Float_t kRCC = kRCoolingOuter;
  const Float_t kRCW = kRCoolingInner;
  const Float_t kRCL = kRCC * TMath::Tan(30. / 180. * TMath::Pi());
  const Float_t kRWL = kRCW * TMath::Tan(30. / 180. * TMath::Pi());
  // Outer Circuits
  //
  // Pipe
  TGeoPgon* shCoolingPipeO = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 4);
  shCoolingPipeO->DefineSection(0, -kRCC, kRCoilOuter + kRCC, kRCoilOuter + kRCC + 0.01);
  shCoolingPipeO->DefineSection(1, -kRCL, kRCoilOuter, kRCoilOuter + 2. * kRCC);
  shCoolingPipeO->DefineSection(2, kRCL, kRCoilOuter, kRCoilOuter + 2. * kRCC);
  shCoolingPipeO->DefineSection(3, kRCC, kRCoilOuter + kRCC, kRCoilOuter + kRCC + 0.01);
  //
  TGeoVolume* voCoolingPipeO = new TGeoVolume("L3CCO", shCoolingPipeO, medAlu);
  voCoilTurn->AddNode(voCoolingPipeO, 1, new TGeoTranslation(0., 0., 0.));
  //
  TGeoPgon* shCoolingWaterO = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 4);
  shCoolingWaterO->DefineSection(0, -kRCW, kRCoilOuter + kRCC, kRCoilOuter + kRCC + 0.01);
  shCoolingWaterO->DefineSection(1, -kRWL, kRCoilOuter + (kRCC - kRCW), kRCoilOuter + kRCC + kRCW);
  shCoolingWaterO->DefineSection(2, kRWL, kRCoilOuter + (kRCC - kRCW), kRCoilOuter + kRCC + kRCW);
  shCoolingWaterO->DefineSection(3, kRCW, kRCoilOuter + kRCC, kRCoilOuter + kRCC + 0.01);
  //
  TGeoVolume* voCoolingWaterO = new TGeoVolume("L3CWO", shCoolingWaterO, medWater);
  voCoolingPipeO->AddNode(voCoolingWaterO, 1, new TGeoTranslation(0., 0., 0.));

  // Inner Circuits
  //
  // Pipe
  TGeoPgon* shCoolingPipeI = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 4);
  shCoolingPipeI->DefineSection(0, -kRCC, kRCoilInner - kRCC, kRCoilInner - kRCC + 0.01);
  shCoolingPipeI->DefineSection(1, -kRCL, kRCoilInner - 2. * kRCC, kRCoilInner);
  shCoolingPipeI->DefineSection(2, kRCL, kRCoilInner - 2. * kRCC, kRCoilInner);
  shCoolingPipeI->DefineSection(3, kRCC, kRCoilInner - kRCC, kRCoilInner - kRCC + 0.01);
  //
  TGeoVolume* voCoolingPipeI = new TGeoVolume("L3CCI", shCoolingPipeI, medAlu);
  voCoilTurn->AddNode(voCoolingPipeI, 1, new TGeoTranslation(0., 0., 0.));
  //
  TGeoPgon* shCoolingWaterI = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 4);
  shCoolingWaterI->DefineSection(0, -kRCW, kRCoilInner - kRCC, kRCoilInner - kRCC + 0.01);
  shCoolingWaterI->DefineSection(1, -kRWL, kRCoilInner - kRCC - kRCW, kRCoilInner - (kRCC - kRCW));
  shCoolingWaterI->DefineSection(2, kRWL, kRCoilInner - kRCC - kRCW, kRCoilInner - (kRCC - kRCW));
  shCoolingWaterI->DefineSection(3, kRCW, kRCoilInner - kRCC, kRCoilInner - kRCC + 0.01);
  //
  TGeoVolume* voCoolingWaterI = new TGeoVolume("L3CWI", shCoolingWaterI, medWater);
  voCoolingPipeI->AddNode(voCoolingWaterI, 1, new TGeoTranslation(0., 0., 0.));

  //
  // Define Yoke
  //
  TGeoPgon* shYoke = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 2);
  shYoke->DefineSection(0, -kLYoke, kRYokeInner, kRYokeOuter);
  shYoke->DefineSection(1, +kLYoke, kRYokeInner, kRYokeOuter);
  //
  TGeoVolume* voYoke = new TGeoVolume("L3YO", shYoke, medSteel);
  voBMother->AddNode(voYoke, 1, new TGeoTranslation(0., 0., 0.));

  //
  // Define Crown
  //
  TGeoPgon* shCrown = new TGeoPgon(kStartAngle, kFullAngle, kNSides, 4);
  shCrown->DefineSection(0, kLCrown1, kRCrownInner, kRYokeInner);
  shCrown->DefineSection(1, kLCrown2, kRCrownInner, kRYokeInner);
  shCrown->DefineSection(2, kLCrown2, kRCrownInner, kRCrownOuter);
  shCrown->DefineSection(3, kLCrown3, kRCrownInner, kRCrownOuter);
  //
  TGeoVolume* voCrown = new TGeoVolume("L3CR", shCrown, medSteel);

  //
  // Door including "Plug"
  //
  Float_t slo = 2. * kRDoorOuter * TMath::Tan(22.5 * kDegRad);
  Float_t sli = 2. * kRPlugInner * TMath::Tan(22.5 * kDegRad);
  Double_t xpol1[12], xpol2[12], ypol1[12], ypol2[12];

  xpol1[0] = 2.;
  ypol1[0] = kRDoorOuter;
  xpol1[1] = slo / 2.;
  ypol1[1] = kRDoorOuter;
  xpol1[2] = kRDoorOuter;
  ypol1[2] = slo / 2.;
  xpol1[3] = kRDoorOuter;
  ypol1[3] = -slo / 2.;
  xpol1[4] = slo / 2.;
  ypol1[4] = -kRDoorOuter;
  xpol1[5] = 2.;
  ypol1[5] = -kRDoorOuter;
  xpol1[6] = 2.;
  ypol1[6] = -kRPlugInner - os;
  xpol1[7] = sli / 2.;
  ypol1[7] = -kRPlugInner - os;
  xpol1[8] = kRPlugInner;
  ypol1[8] = -sli / 2. - os;
  xpol1[9] = kRPlugInner;
  ypol1[9] = sli / 2. - os;
  xpol1[10] = sli / 2.;
  ypol1[10] = kRPlugInner - os;
  xpol1[11] = 2.;
  ypol1[11] = kRPlugInner - os;

  TGeoXtru* shL3DoorR = new TGeoXtru(2);
  shL3DoorR->DefinePolygon(12, xpol1, ypol1);
  shL3DoorR->DefineSection(0, kLDoor1);
  shL3DoorR->DefineSection(1, kLDoor2);
  TGeoVolume* voL3DoorR = new TGeoVolume("L3DoorR", shL3DoorR, medSteel);

  for (Int_t i = 0; i < 12; i++) {
    xpol2[i] = -xpol1[11 - i];
    ypol2[i] = ypol1[11 - i];
  }

  TGeoXtru* shL3DoorL = new TGeoXtru(2);
  shL3DoorL->DefinePolygon(12, xpol2, ypol2);
  shL3DoorL->DefineSection(0, kLDoor1);
  shL3DoorL->DefineSection(1, kLDoor2);
  TGeoVolume* voL3DoorL = new TGeoVolume("L3DoorL", shL3DoorL, medSteel);
  //
  // Plug support plate
  //
  Float_t ro = kRPlugInner + 50.;
  slo = 2. * ro * TMath::Tan(22.5 * kDegRad);

  xpol1[0] = 2.;
  ypol1[0] = ro - os;
  xpol1[1] = slo / 2.;
  ypol1[1] = ro - os;
  xpol1[2] = ro;
  ypol1[2] = slo / 2. - os;
  xpol1[3] = ro;
  ypol1[3] = -slo / 2. - os;
  xpol1[4] = slo / 2.;
  ypol1[4] = -ro - os;
  xpol1[5] = 2.;
  ypol1[5] = -ro - os;

  for (Int_t i = 0; i < 12; i++) {
    xpol2[i] = -xpol1[11 - i];
    ypol2[i] = ypol1[11 - i];
  }

  TGeoXtru* shL3PlugSPR = new TGeoXtru(2);
  shL3PlugSPR->DefinePolygon(12, xpol1, ypol1);
  shL3PlugSPR->DefineSection(0, kLDoor1 - 10.);
  shL3PlugSPR->DefineSection(1, kLDoor1);
  TGeoVolume* voL3PlugSPR = new TGeoVolume("L3PlugSPR", shL3PlugSPR, medSteel);

  TGeoXtru* shL3PlugSPL = new TGeoXtru(2);
  shL3PlugSPL->DefinePolygon(12, xpol2, ypol2);
  shL3PlugSPL->DefineSection(0, kLDoor1 - 10.);
  shL3PlugSPL->DefineSection(1, kLDoor1);
  TGeoVolume* voL3PlugSPL = new TGeoVolume("L3PlugSPL", shL3PlugSPL, medSteel);

  // Position crown and door
  TGeoRotation* rotxz = new TGeoRotation("rotxz", 90., 0., 90., 90., 180., 0.);

  TGeoVolumeAssembly* l3 = new TGeoVolumeAssembly("L3MO");
  voBMother->AddNode(voCrown, 1, new TGeoTranslation(0., 0., 0.));
  voBMother->AddNode(voCrown, 2, new TGeoCombiTrans(0., 0., 0., rotxz));
  l3->AddNode(voBMother, 1, new TGeoTranslation(0., 0., 0.));
  l3->AddNode(voL3DoorR, 1, new TGeoTranslation(0., 0., 0.));
  l3->AddNode(voL3DoorR, 2, new TGeoCombiTrans(0., 0., 0., rotxz));
  l3->AddNode(voL3DoorL, 1, new TGeoTranslation(0., 0., 0.));
  l3->AddNode(voL3DoorL, 2, new TGeoCombiTrans(0., 0., 0., rotxz));
  l3->AddNode(voL3PlugSPR, 1, new TGeoTranslation(0., 0., 0.));
  l3->AddNode(voL3PlugSPR, 2, new TGeoCombiTrans(0., 0., 0., rotxz));
  l3->AddNode(voL3PlugSPL, 1, new TGeoTranslation(0., 0., 0.));
  l3->AddNode(voL3PlugSPL, 2, new TGeoCombiTrans(0., 0., 0., rotxz));
  top->AddNode(l3, 1, new TGeoTranslation(0., os, 0.));
}

FairModule* Magnet::CloneModule() const { return new Magnet(*this); }
ClassImp(o2::passive::Magnet);
