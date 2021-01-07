// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "A3DetectorsPassive/A3Pipe.h"
#include <DetectorsBase/Detector.h>
#include <DetectorsBase/MaterialManager.h>
#include <TGeoTube.h>
#include <TVirtualMC.h>
#include "TGeoManager.h"  // for TGeoManager, gGeoManager
#include "TGeoMaterial.h" // for TGeoMaterial
#include "TGeoMedium.h"   // for TGeoMedium
#include "TGeoVolume.h"   // for TGeoVolume
// force availability of assert
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

A3Pipe::A3Pipe() : A3PassiveBase{"Alice3PIPE", ""} {}
A3Pipe::A3Pipe(const char* name,
               const char* title,
               float innerRho,
               float innerThickness,
               float innerLength,
               float outerRho,
               float outerThickness,
               float outerLength)
  : A3PassiveBase{name, title},
    mBeInnerPipeRmax{innerRho},
    mBeInnerPipeThick{innerThickness},
    mInnerIpHLength{innerLength},
    mBeOuterPipeRmax{outerRho},
    mBeOuterPipeThick{outerThickness},
    mOuterIpHLength{outerLength}
{
}

A3Pipe& A3Pipe::operator=(const A3Pipe& rhs)
{
  // self assignment
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  A3PassiveBase::operator=(rhs);

  return *this;
}

void A3Pipe::ConstructGeometry()
{
  createMaterials();
  //
  //  Class describing the beam A3Pipe geometry
  //
  float z, zsh, z0;
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

  const TGeoMedium* kMedBe = matmgr.getTGeoMedium("A3PIPE_BE");
  const TGeoMedium* kMedVac = matmgr.getTGeoMedium("A3PIPE_VACUUM");
  const TGeoMedium* kMedVacNF = matmgr.getTGeoMedium("A3PIPE_VACUUM_NF");
  const TGeoMedium* kMedVacHC = matmgr.getTGeoMedium("A3PIPE_VACUUM_HC");
  const TGeoMedium* kMedVacNFHC = matmgr.getTGeoMedium("A3PIPE_VACUUM_NFHC");

  // Top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoVolume* barrel = gGeoManager->GetVolume("barrel");
  if (!barrel) {
    LOG(FATAL) << "Could not find the top volume";
  }

  //---------------- Innermost Be pipe around the IP ----------
  TGeoTube* innerBeTube =
    new TGeoTube("INN_PIPEsh", mBeInnerPipeRmax - mBeInnerPipeThick, mBeInnerPipeRmax, mInnerIpHLength);
  TGeoVolume* innerBeTubeVolume = new TGeoVolume("INN_PIPE", innerBeTube, kMedBe);
  innerBeTubeVolume->SetLineColor(kRed);

  TGeoTube* berylliumTubeVacuum =
    new TGeoTube("INN_PIPEVACUUMsh", 0., mBeInnerPipeRmax, mInnerIpHLength);
  TGeoVolume* innerBerylliumTubeVacuumVolume = new TGeoVolume("INN_PIPEMOTHER", berylliumTubeVacuum, kMedVac);
  innerBerylliumTubeVacuumVolume->AddNode(innerBeTubeVolume, 1, gGeoIdentity);
  innerBerylliumTubeVacuumVolume->SetVisibility(0);
  innerBerylliumTubeVacuumVolume->SetLineColor(kGreen);

  barrel->AddNode(innerBerylliumTubeVacuumVolume, 1, gGeoIdentity);

  //---------------- Outermost Be pipe around the IP ----------
  TGeoTube* outerBeTube =
    new TGeoTube("OUT_PIPEsh", mBeOuterPipeRmax - mBeOuterPipeThick, mBeOuterPipeRmax, mOuterIpHLength);
  TGeoVolume* outerBeTubeVolume = new TGeoVolume("OUT_PIPE", outerBeTube, kMedBe);
  outerBeTubeVolume->SetLineColor(kBlue);

  TGeoTube* outerBerylliumTubeVacuum =
    new TGeoTube("OUT_PIPEVACUUMsh", 0., mBeOuterPipeRmax, mOuterIpHLength);
  TGeoVolume* outerBerylliumTubeVacuumVolume = new TGeoVolume("OUT_PIPEMOTHER", outerBerylliumTubeVacuum, kMedVac);
  outerBerylliumTubeVacuumVolume->AddNode(outerBeTubeVolume, 1, gGeoIdentity);
  outerBerylliumTubeVacuumVolume->SetVisibility(0);
  outerBerylliumTubeVacuumVolume->SetLineColor(kGreen);

  barrel->AddNode(outerBerylliumTubeVacuumVolume, 1, gGeoIdentity);
}

void A3Pipe::createMaterials()
{
  //
  // Define materials for beam A3Pipe
  //
  Int_t isxfld = 2.;
  float sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

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
  float epsil = .1;     // Tracking precision,
  float stemax = -0.01; // Maximum displacement for multiple scat
  float tmaxfd = -20.;  // Maximum angle due to field deflection
  float deemax = -.3;   // Maximum fractional energy loss, DLS
  float stmin = -.8;
  // ***************

  auto& matmgr = o2::base::MaterialManager::Instance();

  //    Beryllium
  matmgr.Material("A3PIPE", 5, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7);
  matmgr.Medium("A3PIPE", 5, "BE", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Vacuum
  matmgr.Mixture("A3PIPE", 16, "VACUUM$ ", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("A3PIPE", 36, "VACUUM$_NF", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("A3PIPE", 56, "VACUUM$_HC ", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("A3PIPE", 76, "VACUUM$_NFHC", aAir, zAir, dAir1, 4, wAir);

  matmgr.Medium("A3PIPE", 16, "VACUUM", 16, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("A3PIPE", 36, "VACUUM_NF", 36, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("A3PIPE", 56, "VACUUM_HC", 56, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("A3PIPE", 76, "VACUUM_NFHC", 76, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

// ----------------------------------------------------------------------------
FairModule* A3Pipe::CloneModule() const { return new A3Pipe(*this); }
ClassImp(o2::passive::A3Pipe);