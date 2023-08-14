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
  matmgr.Mixture("CAVE", 2, "Air", aAir, zAir, dAir, 4, wAir);
  matmgr.Mixture("CAVE", 3, "Air_NF", aAir, zAir, dAir, 4, wAir);
  //
  matmgr.Medium("CAVE", 2, "Air", 2, 0, isxfld, sxmgmx, 10, -1, -0.1, 0.1, -10);
  matmgr.Medium("CAVE", 3, "Air_NF", 3, 0, 0, sxmgmx, 10, -1, -0.1, 0.1, -10);
}

void Cave::ConstructGeometry()
{
  createMaterials();
  auto& matmgr = o2::base::MaterialManager::Instance();
  auto kMedAir = gGeoManager->GetMedium("CAVE_Air");

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
  if (mHasRB24) { // should be not true only for alice 3
    // mother volume for RB24 side (FDD, Compensator)
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
  } else {
    LOGP(info, "Setting up CAVE without RB24");
  }
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
