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

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    Cave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------
#include "DetectorsBase/MaterialManager.h"
#include "DetectorsBase/Detector.h"
#include "DetectorsPassive/Cave.h"
#include "SimConfig/SimConfig.h"
#include "SimConfig/SimParams.h"
#include <TRandom.h>
#include <fairlogger/Logger.h>
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPgon.h"
#include "TGeoPcon.h"
#include "TGeoTube.h"
#include "TGeoCompositeShape.h"
using namespace o2::passive;

void Cave::createMaterials()
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  // Create materials and media
  Int_t isxfld;
  Float_t sxmgmx;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);
  LOG(info) << "Field in CAVE: " << isxfld;
  // AIR
  isxfld = 1;
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3 * 960. / 1014.;
  //
  // Water
  Float_t aWater[2] = {1.00794, 15.9994};
  Float_t zWater[2] = {1., 8.};
  Float_t wWater[2] = {0.111894, 0.888106};
  //
  // Polymer
  //
  Float_t apoly[2] = {12.01, 1.};
  Float_t zpoly[2] = {6., 1.};
  Float_t wpoly[2] = {0.857, .143};

  // Air
  matmgr.Mixture("CAVE", 2, "Air", aAir, zAir, dAir, 4, wAir);
  matmgr.Mixture("CAVE", 3, "Air_NF", aAir, zAir, dAir, 4, wAir);
  matmgr.Medium("CAVE", 2, "Air", 2, 0, isxfld, sxmgmx, 10, -1, -0.1, 0.1, -10);
  matmgr.Medium("CAVE", 3, "Air_NF", 3, 0, 0, sxmgmx, 10, -1, -0.1, 0.1, -10);
  // Copper
  matmgr.Material("CAVE", 10, "COPPER", 63.55, 29, 8.96, 1.43, 85.6 / 8.96);
  matmgr.Medium("CAVE", 10, "COPPER", 10, 0, 0, sxmgmx, 10, -1, -0.1, 0.1, -10);
  // Carbon
  matmgr.Material("CAVE", 13, "Carbon", 12.01, 6., 1.75, 24.4, 49.9);
  matmgr.Medium("CAVE", 13, "Carbon", 13, 0, 0, sxmgmx, 10, -1, -0.1, 0.1, -10);
  // Water
  matmgr.Mixture("CAVE", 11, "Water", aWater, zWater, 1., 2, wWater);
  matmgr.Medium("CAVE", 11, "Water", 11, 0, 0, sxmgmx, 10, -1, -0.1, 0.1, -10);
  // Polymer
  matmgr.Mixture("CAVE", 12, "Polymer", apoly, zpoly, 0.9, 2, wpoly);
  matmgr.Medium("CAVE", 12, "Polymer", 12, 0, 0, sxmgmx, 10, -1, -0.1, 0.1, -10);
}

void Cave::ConstructGeometry()
{
  createMaterials();
  auto& matmgr = o2::base::MaterialManager::Instance();
  auto kMedAir = gGeoManager->GetMedium("CAVE_Air");
  auto kMedCu = gGeoManager->GetMedium("CAVE_COPPER");
  auto kMedWater = gGeoManager->GetMedium("CAVE_Water");
  auto kMedPolymer = gGeoManager->GetMedium("CAVE_Polymer");
  auto kMedCarbon = gGeoManager->GetMedium("CAVE_Carbon");
  Float_t dALIC[3];

  if (mHasZDC) {
    LOG(info) << "Setting up CAVE to host ZDC";
    // dimensions taken from ALIROOT
    dALIC[0] = 2500;
    dALIC[1] = 2500;
    dALIC[2] = 15000;
  } else {
    LOG(info) << "Setting up CAVE without ZDC";
    dALIC[0] = 2000;
    dALIC[1] = 2000;
    dALIC[2] = 3000;
  }
  auto cavevol = gGeoManager->MakeBox("cave", kMedAir, dALIC[0], dALIC[1], dALIC[2]);
  gGeoManager->SetTopVolume(cavevol);

  TGeoPgon* shCaveTR1 = new TGeoPgon("shCaveTR1", 22.5, 360., 8., 2);
  shCaveTR1->DefineSection(0, -706. - 8.6, 0., 790.5);
  shCaveTR1->DefineSection(1, 707. + 7.6, 0., 790.5);
  TGeoTube* shCaveTR2 = new TGeoTube("shCaveTR2", 0., 150., 110.);

  TGeoTranslation* transCaveTR2 = new TGeoTranslation("transTR2", 0, 30., -505. - 110.);
  transCaveTR2->RegisterYourself();
  TGeoCompositeShape* shCaveTR = new TGeoCompositeShape("shCaveTR", "shCaveTR1-shCaveTR2:transTR2");
  TGeoVolume* voBarrel = new TGeoVolume("barrel", shCaveTR, kMedAir);
  cavevol->AddNode(voBarrel, 1, new TGeoTranslation(0., -30., 0.));

  // ITS services
  // Inner Barrel
  // Mother

  TGeoPcon* shITSIB = new TGeoPcon(0., 360., 3);
  auto drCC = 0.2;
  auto drSE = 0.462;
  auto drMo1 = drCC + 1.8 * 0.462;
  auto drMo2 = drCC + 0.462 / 1.8;
  auto r2 = 13.3;
  auto r1 = r2 - drMo1;
  auto r4 = 43.63;
  auto r3 = r4 - drMo2;
  auto z1 = 35.;
  auto z2 = 44.;
  auto z3 = 253.45;
  shITSIB->DefineSection(0, z1, r1, r2);
  shITSIB->DefineSection(1, z2, r1, r2);
  shITSIB->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSIB = new TGeoVolume("ITSIB", shITSIB, kMedCarbon);
  voBarrel->AddNode(voITSIB, 1, new TGeoTranslation(0., 30., 0.));
  // Copper
  r2 -= drCC;
  r4 -= drCC;
  TGeoPcon* shITSIBcu = new TGeoPcon(0., 360., 3);
  shITSIBcu->DefineSection(0, z1, r1, r2);
  shITSIBcu->DefineSection(1, z2, r1, r2);
  shITSIBcu->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSIBcu = new TGeoVolume("ITSIBcu", shITSIBcu, kMedCu);
  voITSIB->AddNode(voITSIBcu, 1, new TGeoTranslation(0., 0., 0.));
  // Polymer
  auto drCu = 0.018;
  r2 -= drCu * 1.8;
  r4 -= drCu / 1.8;
  TGeoPcon* shITSIBpo = new TGeoPcon(0., 360., 3);
  shITSIBpo->DefineSection(0, z1, r1, r2);
  shITSIBpo->DefineSection(1, z2, r1, r2);
  shITSIBpo->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSIBpo = new TGeoVolume("ITSIBpo", shITSIBpo, kMedPolymer);
  voITSIBcu->AddNode(voITSIBpo, 1, new TGeoTranslation(0., 0., 0.));
  // Water
  auto drPo = 0.42;
  r2 -= drPo * 1.8;
  r4 -= drPo / 1.8;
  TGeoPcon* shITSIBwa = new TGeoPcon(0., 360., 3);
  shITSIBwa->DefineSection(0, z1, r1, r2);
  shITSIBwa->DefineSection(1, z2, r1, r2);
  shITSIBwa->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSIBwa = new TGeoVolume("ITSIBwa", shITSIBwa, kMedWater);
  voITSIBpo->AddNode(voITSIBwa, 1, new TGeoTranslation(0., 0., 0.));
  // Outer Barrel
  TGeoPcon* shITSOB = new TGeoPcon(0., 360., 3);
  auto drMo = 4.42;
  r2 = 47.3;
  r1 = r2 - drMo;
  // r4 = 55.25;
  r4 = r2;
  r3 = r4 - drMo;
  z1 = 83.;
  z2 = 117.28;
  z3 = 248.0;
  shITSOB->DefineSection(0, z1, r1, r2);
  shITSOB->DefineSection(1, z2, r1, r2);
  shITSOB->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSOB = new TGeoVolume("ITSOB", shITSOB, kMedCarbon);
  voBarrel->AddNode(voITSOB, 1, new TGeoTranslation(0., 30., 0.));
  drCC = 0.25;
  // Copper
  r2 -= drCC;
  r4 -= drCC;
  TGeoPcon* shITSOBcu = new TGeoPcon(0., 360., 3);
  shITSOBcu->DefineSection(0, z1, r1, r2);
  shITSOBcu->DefineSection(1, z2, r1, r2);
  shITSOBcu->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSOBcu = new TGeoVolume("ITSOBcu", shITSOBcu, kMedCu);
  voITSOB->AddNode(voITSOBcu, 1, new TGeoTranslation(0., 0., 0.));
  // Polymer
  drCu = 0.23;
  r2 -= drCu;
  r4 -= drCu;
  TGeoPcon* shITSOBpo = new TGeoPcon(0., 360., 3);
  shITSOBpo->DefineSection(0, z1, r1, r2);
  shITSOBpo->DefineSection(1, z2, r1, r2);
  shITSOBpo->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSOBpo = new TGeoVolume("ITSOBpo", shITSOBpo, kMedPolymer);
  voITSOBcu->AddNode(voITSOBpo, 1, new TGeoTranslation(0., 0., 0.));
  // Water
  drPo = 3.8;
  r2 -= drPo;
  r4 -= drPo;
  TGeoPcon* shITSOBwa = new TGeoPcon(0., 360., 3);
  shITSOBwa->DefineSection(0, z1, r1, r2);
  shITSOBwa->DefineSection(1, z2, r1, r2);
  shITSOBwa->DefineSection(2, z3, r3, r4);
  TGeoVolume* voITSOBwa = new TGeoVolume("ITSOBwa", shITSOBwa, kMedWater);
  voITSOBpo->AddNode(voITSOBwa, 1, new TGeoTranslation(0., 0., 0.));

  // mother volune for RB24 side (FDD, Compensator)
  const Float_t kRB24CL = 2. * 597.9;
  auto shCaveRB24 = new TGeoPcon(0., 360., 6);
  Float_t z0 = kRB24CL / 2 + 714.6;
  shCaveRB24->DefineSection(0, -kRB24CL / 2., 0., 105.);
  shCaveRB24->DefineSection(1, -z0 + 1705., 0., 105.);
  shCaveRB24->DefineSection(2, -z0 + 1705., 0., 14.5);
  shCaveRB24->DefineSection(3, -z0 + 1880., 0., 14.5);
  shCaveRB24->DefineSection(4, -z0 + 1880., 0., 40.0);
  shCaveRB24->DefineSection(5, kRB24CL / 2, 0., 40.0);

  TGeoVolume* caveRB24 = new TGeoVolume("caveRB24", shCaveRB24, kMedAir);
  caveRB24->SetVisibility(0);
  cavevol->AddNode(caveRB24, 1, new TGeoTranslation(0., 0., z0));
  //
}

Cave::Cave() : FairDetector() {}
Cave::~Cave() = default;
Cave::Cave(const char* name, const char* Title) : FairDetector(name, Title, -1) {}
Cave::Cave(const Cave& rhs) : FairDetector(rhs) {}
Cave& Cave::operator=(const Cave& rhs)
{
  // self assignment
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  FairModule::operator=(rhs);
  return *this;
}

FairModule* Cave::CloneModule() const { return new Cave(*this); }
void Cave::FinishPrimary()
{
  LOG(debug) << "CAVE: Primary finished";
  for (auto& f : mFinishPrimaryHooks) {
    f();
  }
}

// we set the random number generator for each
// primary in order to be reproducible in a multi-processing sub-event
// setting
void Cave::BeginPrimary()
{
  static int primcounter = 0;

  // only do it, if not in pure trackSeeding mode
  if (!o2::conf::SimCutParams::Instance().trackSeed) {
    auto& conf = o2::conf::SimConfig::Instance();
    auto chunks = conf.getInternalChunkSize();
    if (chunks != -1) {
      if (primcounter % chunks == 0) {
        static int counter = 1;
        auto seed = counter + 10;
        gRandom->SetSeed(seed);
        counter++;
      }
    }
  }
  primcounter++;
}

bool Cave::ProcessHits(FairVolume*)
{
  LOG(fatal) << "CAVE ProcessHits called; should never happen";
  return false;
}

ClassImp(o2::passive::Cave);
