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
#include <Alice3DetectorsPassive/Magnet.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoVolume.h>
#include <TGeoTube.h>

using namespace o2::passive;

Alice3Magnet::~Alice3Magnet() = default;
Alice3Magnet::Alice3Magnet() : Alice3PassiveBase("A3MAG", "") {}
Alice3Magnet::Alice3Magnet(const char* name, const char* title) : Alice3PassiveBase(name, title) {}
Alice3Magnet::Alice3Magnet(const Alice3Magnet& rhs) = default;

Alice3Magnet& Alice3Magnet::operator=(const Alice3Magnet& rhs)
{
  if (this == &rhs) {
    return *this;
  }

  Alice3PassiveBase::operator=(rhs);

  return *this;
}

void Alice3Magnet::createMaterials()
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  int isxfld = 2.;
  float sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // Current information is scarce, we have some X/X0 and thicknesses but no full material information
  // We use then two main materials: Aluminium for insulation, cryostats, stabiliser, supports and strips.
  // Copper for the coils.
  // Latest updated reference table is, for the moment:
  // +------------------+-------------------------+----------+--------+
  // |  layer           | effective thickness [mm]|  X0 [cm] | X0 [%] |
  // +------------------+-------------------------+----------+--------+
  // | Support cylinder |           20            |  8.896   | 0.225  |
  // | Al-strip         |            1            |  8.896   | 0.011  |
  // | NbTi/Cu          |            3            |  1.598   | 0.188  |
  // | Insulation       |           11            | 17.64    | 0.062  |
  // | Al-stabiliser    |           33            |  8.896   | 0.371  |
  // | Inner cryostat   |           10            |  8.896   | 0.112  |
  // | Outer cryostat   |           30            |  8.896   | 0.337  |
  // +------------------+-------------------------+----------+--------+
  // Geometry will be oversimplified in two wrapping cylindrical Al layers (symmetric for the time being) with a Copper layer in between.

  float epsil, stmin, tmaxfd, deemax, stemax;
  epsil = .001;   // Tracking precision,
  stemax = -0.01; // Maximum displacement for multiple scat
  tmaxfd = -20.;  // Maximum angle due to field deflection
  deemax = -.3;   // Maximum fractional energy loss, DLS
  stmin = -.8;

  matmgr.Material("A3MAG", 9, "Al1$", 26.98, 13., 2.7, 8.9, 37.2);
  matmgr.Material("A3MAG", 19, "Cu1$", 63.55, 29., 8.96, 1.6, 18.8);

  matmgr.Medium("A3MAG", 9, "ALU_C0", 9, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("A3MAG", 19, "CU_C0", 19, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

void Alice3Magnet::ConstructGeometry()
{
  createMaterials();

  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* barrel = geoManager->GetVolume("barrel");
  if (!barrel) {
    LOGP(fatal, "Could not find barrel volume while constructing Alice 3 magnet geometry");
  }

  auto& matmgr = o2::base::MaterialManager::Instance();
  auto kMedAl = matmgr.getTGeoMedium("A3MAG_ALU_C0");
  auto kMedCu = matmgr.getTGeoMedium("A3MAG_CU_C0");

  float wrapThickness = (mTotalThickness - mCoilThickness) / 2;      // Thickness of the Al wraps
  float innerCoilsRadius = mInnerWrapInnerRadius + wrapThickness;    // Inner radius of the Cu coils
  float externalWrapInnerRadius = innerCoilsRadius + mCoilThickness; // Inner radius of the external wrapping Al cylinder

  // inner wrap
  LOGP(debug, "Alice 3 magnet: creating inner wrap with inner radius {} and thickness {}", mInnerWrapInnerRadius, wrapThickness);
  TGeoTube* innerLayer = new TGeoTube(mInnerWrapInnerRadius, mInnerWrapInnerRadius + wrapThickness, mZLength / 2);
  // coils layer
  LOGP(debug, "Alice 3 magnet: creating coils layer with inner radius {} and thickness {}", innerCoilsRadius, mCoilThickness);
  TGeoTube* coilsLayer = new TGeoTube(innerCoilsRadius, innerCoilsRadius + mCoilThickness, mZLength / 2);
  // outer wrap
  LOGP(debug, "Alice 3 magnet: creating outer wrap with inner radius {} and thickness {}", externalWrapInnerRadius, wrapThickness);
  TGeoTube* outerLayer = new TGeoTube(externalWrapInnerRadius, externalWrapInnerRadius + wrapThickness, mZLength / 2);

  TGeoVolume* innerWrap = new TGeoVolume("innerWrap", innerLayer, kMedAl);
  TGeoVolume* coils = new TGeoVolume("coils", coilsLayer, kMedCu);
  TGeoVolume* outerWrap = new TGeoVolume("outerWrap", outerLayer, kMedAl);
  innerWrap->SetLineColor(kRed);
  coils->SetLineColor(kOrange);
  outerWrap->SetLineColor(kRed);

  new TGeoVolumeAssembly("magnet");
  auto* magnet = gGeoManager->GetVolume("magnet");
  magnet->AddNode(innerWrap, 1, nullptr);
  magnet->AddNode(coils, 1, nullptr);
  magnet->AddNode(outerWrap, 1, nullptr);

  magnet->SetVisibility(1);

  barrel->AddNode(magnet, 1, new TGeoTranslation(0, 30.f, 0));
}

FairModule* Alice3Magnet::CloneModule() const
{
  return new Alice3Magnet(*this);
}
ClassImp(o2::passive::Alice3Magnet)