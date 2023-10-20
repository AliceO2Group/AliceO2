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

#include <DetectorsBase/Detector.h>
#include <DetectorsBase/MaterialManager.h>
#include <Alice3DetectorsPassive/Absorber.h>
#include <TGeoArb8.h> // for TGeoTrap
#include <TGeoCompositeShape.h>
#include <TGeoCone.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoPcon.h>
#include <TGeoPgon.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

Alice3Absorber::~Alice3Absorber() = default;

Alice3Absorber::Alice3Absorber() : Alice3PassiveBase("A3ABSO", "") {}
Alice3Absorber::Alice3Absorber(const char* name, const char* Title) : Alice3PassiveBase(name, Title) {}
Alice3Absorber::Alice3Absorber(const Alice3Absorber& rhs) = default;

Alice3Absorber& Alice3Absorber::operator=(const Alice3Absorber& rhs)
{
  // self assignment
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  FairModule::operator=(rhs);

  return *this;
}

void Alice3Absorber::createMaterials()
{

  auto& matmgr = o2::base::MaterialManager::Instance();
  // Define materials for muon absorber
  //
  Int_t isxfld = 2.;
  Float_t sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  //
  // Steel
  //
  Float_t asteel[4] = {55.847, 51.9961, 58.6934, 28.0855};
  Float_t zsteel[4] = {26., 24., 28., 14.};
  Float_t wsteel[4] = {.715, .18, .1, .005};
  //
  // Air
  //
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;
  float dAir1 = 1.20479E-11;

  // ****************
  //     Defines tracking media parameters.
  //
  Float_t epsil, stmin, tmaxfd, deemax, stemax;
  epsil = .001;   // Tracking precision,
  stemax = -0.01; // Maximum displacement for multiple scat
  tmaxfd = -20.;  // Maximum angle due to field deflection
  deemax = -.3;   // Maximum fractional energy loss, DLS
  stmin = -.8;
  // ***************
  //

  matmgr.Mixture("ALICE3ABSO", 16, "VACUUM0$", aAir, zAir, dAir1, 4, wAir);
  matmgr.Medium("ALICE3ABSO", 16, "VA_C0", 16, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //
  //    Steel
  matmgr.Mixture("ALICE3ABSO", 19, "STAINLESS STEEL0$", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("ALICE3ABSO", 39, "STAINLESS STEEL1$", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("ALICE3ABSO", 59, "STAINLESS STEEL2$", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Medium("ALICE3ABSO", 19, "ST_C0", 19, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("ALICE3ABSO", 39, "ST_C1", 39, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("ALICE3ABSO", 59, "ST_C3", 59, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

void Alice3Absorber::ConstructGeometry()
{
  createMaterials();

  //
  // Build muon shield geometry
  //
  //

  auto& matmgr = o2::base::MaterialManager::Instance();

  //
  // Media
  //

  auto kMedVac = matmgr.getTGeoMedium("ALICE3ABSO_VA_C0");
  auto kMedSteel = matmgr.getTGeoMedium("ALICE3ABSO_ST_C0");
  auto kMedSteelSh = matmgr.getTGeoMedium("ALICE3ABSO_ST_C3");

  // The top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoVolume* barrel = gGeoManager->GetVolume("barrel");
  if (!barrel) {
    LOG(fatal) << "Could not find the top volume";
  }

  TGeoPcon* absorings = new TGeoPcon(0., 360., 18);

  absorings->DefineSection(0, 500, 236, 274);
  absorings->DefineSection(1, 400, 236, 274);
  absorings->DefineSection(2, 400, 232.5, 277.5);
  absorings->DefineSection(3, 300, 232.5, 277.5);
  absorings->DefineSection(4, 300, 227.5, 282.5);
  absorings->DefineSection(5, 200, 227.5, 282.5);
  absorings->DefineSection(6, 200, 222.5, 287.5);
  absorings->DefineSection(7, 100, 222.5, 287.5);
  absorings->DefineSection(8, 100, 220, 290);
  absorings->DefineSection(9, -100, 220, 290);
  absorings->DefineSection(10, -100, 222.5, 287.5);
  absorings->DefineSection(11, -200, 222.5, 287.5);
  absorings->DefineSection(12, -200, 227.5, 282.5);
  absorings->DefineSection(13, -300, 227.5, 282.5);
  absorings->DefineSection(14, -300, 232.5, 277.5);
  absorings->DefineSection(15, -400, 232.5, 277.5);
  absorings->DefineSection(16, -400, 236, 274);
  absorings->DefineSection(17, -500, 236, 274);

  // Insert
  absorings->SetName("absorings");

  TGeoVolume* abso = new TGeoVolume("Absorber", absorings, kMedSteel);

  abso->SetVisibility(1);
  abso->SetTransparency(50);
  abso->SetLineColor(kGray);

  //
  //    Adding volumes to mother volume
  //

  barrel->AddNode(abso, 1, new TGeoTranslation(0, 30.f, 0));
}

FairModule* Alice3Absorber::CloneModule() const { return new Alice3Absorber(*this); }
ClassImp(o2::passive::Alice3Absorber);
