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

#include "Alice3DetectorsPassive/Pipe.h"
#include <DetectorsBase/Detector.h>
#include <DetectorsBase/MaterialManager.h>
#include "TGeoTube.h"
#include "TVirtualMC.h"
#include "TGeoManager.h"        // for TGeoManager, gGeoManager
#include "TGeoMaterial.h"       // for TGeoMaterial
#include "TGeoMedium.h"         // for TGeoMedium
#include "TGeoVolume.h"         // for TGeoVolume
#include "TGeoCompositeShape.h" // for TGeoCompositeShape
#include "TCanvas.h"
// force availability of assert
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

Alice3Pipe::Alice3Pipe() : Alice3PassiveBase{"Alice3PIPE", ""} {}
Alice3Pipe::Alice3Pipe(const char* name,
                       const char* title,
                       float pipeRIn,
                       float pipeThickness,
                       float a3ipLength,
                       float vacuumVesselRIn,
                       float vacuumVesselThickness,
                       float vacuumVesselASideLength)
  : Alice3PassiveBase{name, title},
    mPipeRIn{pipeRIn},
    mPipeThick{pipeThickness},
    mA3IPLength{a3ipLength},
    mVacuumVesselRIn{vacuumVesselRIn},
    mVacuumVesselThick{vacuumVesselThickness},
    mVacuumVesselASideLength{vacuumVesselASideLength}
{
}

Alice3Pipe::~Alice3Pipe() = default;
Alice3Pipe& Alice3Pipe::operator=(const Alice3Pipe& rhs)
{
  // self assignment
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  Alice3PassiveBase::operator=(rhs);

  return *this;
}

void Alice3Pipe::ConstructGeometry()
{
  createMaterials();
  //
  //  Class describing the beam Alice3Pipe geometry
  //
  // Rotation Matrices
  //
  const float kDegRad = TMath::Pi() / 180.;
  // Rotation by 180 deg
  TGeoRotation* rot180 = new TGeoRotation("rot180", 90., 180., 90., 90., 180., 0.);
  TGeoRotation* rotyz = new TGeoRotation("rotyz", 90., 180., 0., 180., 90., 90.);
  TGeoRotation* rotxz = new TGeoRotation("rotxz", 0., 0., 90., 90., 90., 180.);
  //

  //
  // Media
  auto& matmgr = o2::base::MaterialManager::Instance();

  const TGeoMedium* kMedBe = matmgr.getTGeoMedium("ALICE3_PIPE_BE");

  // Top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoVolume* barrel = gGeoManager->GetVolume("barrel");
  if (!barrel) {
    LOG(fatal) << "Could not find the top volume";
  }

  // We split the naming of the parts if the beam pipe for ALICE 3 into parts
  // - pipe A Side
  // - pipe C Side (which hosts the primary vacuum vessel and covers all C Side as well)

  // A3IP update
  Double_t pipeASideLength = mA3IPLength / 2. - mVacuumVesselThick - mVacuumVesselASideLength / 2.;
  Double_t pipeCSideLength = mA3IPLength / 2. + mVacuumVesselASideLength / 2.;

  // Pipe tubes
  TGeoTube* pipeASide = new TGeoTube("PIPE_Ash", mPipeRIn, mPipeRIn + mPipeThick, pipeASideLength / 2.);
  TGeoTube* pipeCSide = new TGeoTube("PIPE_Csh", mVacuumVesselRIn, mVacuumVesselRIn + mVacuumVesselThick, pipeCSideLength / 2.);
  TGeoTube* vacuumVesselWall = new TGeoTube("VACUUM_VESSEL_WALLsh", mPipeRIn, mVacuumVesselRIn + mVacuumVesselThick, mVacuumVesselThick / 2.);

  // Pipe and vacuum vessel positions
  TGeoTranslation* posPipeASide = new TGeoTranslation("PIPE_ASIDE_POSITION", 0, 0, mVacuumVesselASideLength / 2. + mVacuumVesselThick + pipeASideLength / 2.);
  posPipeASide->RegisterYourself();
  TGeoTranslation* posPipeCSide = new TGeoTranslation("PIPE_CSIDE_POSITION", 0, 0, mVacuumVesselASideLength / 2. - pipeCSideLength / 2.);
  posPipeCSide->RegisterYourself();
  TGeoTranslation* posVacuumVesselWall = new TGeoTranslation("WALL_POSITION", 0, 0, mVacuumVesselASideLength / 2. + mVacuumVesselThick / 2.);
  posVacuumVesselWall->RegisterYourself();

  // Pipe composite shape and volume
  TString pipeCompositeFormula =
    "PIPE_Ash:PIPE_ASIDE_POSITION"
    "+PIPE_Csh:PIPE_CSIDE_POSITION"
    "+VACUUM_VESSEL_WALLsh:WALL_POSITION";

  TGeoCompositeShape* pipeComposite = new TGeoCompositeShape("A3IPsh", pipeCompositeFormula);
  TGeoVolume* pipeVolume = new TGeoVolume("A3IP", pipeComposite, kMedBe);

  // Add everything to the barrel
  barrel->AddNode(pipeVolume, 1, new TGeoTranslation(0, 30.f, 0));

  pipeVolume->SetLineColor(37);
  pipeVolume->SetTransparency(0);
}

void Alice3Pipe::createMaterials()
{
  //
  // Define materials for beam Alice3Pipe
  //
  int isxfld = 2;
  float sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // ****************
  //     Defines tracking media parameters.
  //
  float epsil = .1;     // Tracking precision,
  float stemax = -0.01; // Maximum displacement for multiple scat
  float tmaxfd = -20.;  // Maximum angle due to field deflection
  float deemax = -.3;   // Maximum fractional energy loss, DLS
  float stmin = -.8;
  // ***************

  auto& matmgr = o2::base::MaterialManager::Instance();

  // Beryllium
  matmgr.Material("ALICE3_PIPE", 5, "BERYLLIUM$", 9.01, 4., 1.848, 35.3, 36.7);
  matmgr.Medium("ALICE3_PIPE", 5, "BE", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

// ----------------------------------------------------------------------------
FairModule* Alice3Pipe::CloneModule() const { return new Alice3Pipe(*this); }
ClassImp(o2::passive::Alice3Pipe);
