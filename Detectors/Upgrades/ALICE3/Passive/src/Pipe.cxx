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
#include <TGeoTube.h>
#include <TVirtualMC.h>
#include "TGeoManager.h"        // for TGeoManager, gGeoManager
#include "TGeoMaterial.h"       // for TGeoMaterial
#include "TGeoMedium.h"         // for TGeoMedium
#include "TGeoVolume.h"         // for TGeoVolume
#include "TGeoCompositeShape.h" // forTGeoCompositeShape
// force availability of assert
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

Alice3Pipe::Alice3Pipe() : Alice3PassiveBase{"Alice3PIPE", ""} {}
Alice3Pipe::Alice3Pipe(const char* name,
                       const char* title,
                       bool isTRKActivated,
                       float rMinInnerPipe,
                       float innerThickness,
                       float innerLength,
                       float rMinOuterPipe,
                       float outerThickness,
                       float outerLength)
  : Alice3PassiveBase{name, title},
    mIsTRKActivated{isTRKActivated},
    mBeInnerPipeRmin{rMinInnerPipe},
    mBeInnerPipeThick{innerThickness},
    mInnerIpHLength{innerLength},
    mBeOuterPipeRmin{rMinOuterPipe},
    mBeOuterPipeThick{outerThickness},
    mOuterIpHLength{outerLength}
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

  const TGeoMedium* kMedBe = matmgr.getTGeoMedium("ALICE3PIPE_BE");
  const TGeoMedium* kMedVac = matmgr.getTGeoMedium("ALICE3PIPE_VACUUM");
  const TGeoMedium* kMedVacNF = matmgr.getTGeoMedium("ALICE3PIPE_VACUUM_NF");
  const TGeoMedium* kMedVacHC = matmgr.getTGeoMedium("ALICE3PIPE_VACUUM_HC");
  const TGeoMedium* kMedVacNFHC = matmgr.getTGeoMedium("ALICE3PIPE_VACUUM_NFHC");

  // Top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoVolume* barrel = gGeoManager->GetVolume("barrel");
  if (!barrel) {
    LOG(fatal) << "Could not find the top volume";
  }

  //---------------- Outermost Be pipe around the IP ----------
  // Outer pipe has to be filled with vacuum. There we have also TRK layers, which we don't want to depend on the pipe volume.
  // Eventually, we will depend on some information passed from the outside.
  // For A3PIP-only simulations, we don't want TRK's shade.
  // Strategy used here is to use a composite shape where shapes of TRK layers are subtracted to the vacuum volume
  TGeoTube* outerBeTube = new TGeoTube("OUT_PIPEsh", mBeOuterPipeRmin, mBeOuterPipeRmin + mBeOuterPipeThick, mOuterIpHLength);
  TGeoVolume* outerBeTubeVolume = new TGeoVolume("OUT_PIPE", outerBeTube, kMedBe);
  outerBeTubeVolume->SetLineColor(kGreen - 9);

  TGeoTube* outerBerylliumTubeVacuumBase = new TGeoTube("OUT_PIPEVACUUM_BASEsh", mBeInnerPipeRmin + mBeInnerPipeThick, mBeOuterPipeRmin, mOuterIpHLength); // Vacuum filling for outer pipe
  TGeoCompositeShape* outerBerylliumTubeVacuumComposite;                                                                                                   // Composite volume to subctract to vacuum
  TGeoVolume* outerBerylliumTubeVacuumVolume;                                                                                                              // Final volume to be used

  TString compositeFormula{"OUT_PIPEVACUUM_BASEsh"}; // If pipe is alone we won't subctract anything
  TString subtractorsFormula;

  if (!mIsTRKActivated) {
    std::vector<TGeoTube*> trkLayerShapes;

    std::vector<std::array<float, 3>> layersQuotas = {std::array<float, 3>{0.5f, 50.f, 50.e-4}, // TODO: Set layers dynamically. {radius, zLen, thickness}
                                                      std::array<float, 3>{1.2f, 50.f, 50.e-4},
                                                      std::array<float, 3>{2.5f, 50.f, 50.e-4}};

    subtractorsFormula = "TRKLAYER_0sh";                           // First volume to be subctracted (at least one has to be provided)
    for (auto iLayer{0}; iLayer < layersQuotas.size(); ++iLayer) { // Create TRK layers shapes
      auto& layerData = layersQuotas[iLayer];
      trkLayerShapes.emplace_back(new TGeoTube(Form("TRKLAYER_%dsh", iLayer), layerData[0], layerData[0] + layerData[2], layerData[1] / 2));
      if (iLayer > 0) {
        subtractorsFormula += Form("+TRKLAYER_%dsh", iLayer);
      }
    }

    // Escavate vacuum for hosting cold plate
    TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", 2.6f, 2.6f + 150.e-3, 50.f / 2);
    subtractorsFormula += "+TRK_COLDPLATEsh";

    LOG(info) << "Subtractors formula before : " << subtractorsFormula;
    subtractorsFormula = Form("-(%s)", subtractorsFormula.Data());
    LOG(info) << "Subtractors formula after: " << subtractorsFormula;

    outerBerylliumTubeVacuumComposite = new TGeoCompositeShape("OUT_PIPEVACUUMsh", (compositeFormula + subtractorsFormula).Data());
    outerBerylliumTubeVacuumVolume = new TGeoVolume("OUT_PIPEVACUUM", outerBerylliumTubeVacuumComposite, kMedVac);
  } else {
    outerBerylliumTubeVacuumVolume = new TGeoVolume("OUT_PIPEVACUUM", outerBerylliumTubeVacuumBase, kMedVac);
  }

  outerBerylliumTubeVacuumVolume->SetVisibility(1);
  outerBerylliumTubeVacuumVolume->SetTransparency(50);
  outerBerylliumTubeVacuumVolume->SetLineColor(kGreen);

  //  outerBeTubeVolume->AddNode(outerBerylliumTubeVacuumVolume, 1, gGeoIdentity);
  barrel->AddNode(outerBerylliumTubeVacuumVolume, 1, new TGeoTranslation(0, 30.f, 0));

  barrel->AddNode(outerBeTubeVolume, 1, new TGeoTranslation(0, 30.f, 0)); // Add to surrounding geometry

  //---------------- Innermost Be pipe around the IP ----------
  TGeoTube* innerBeTube =
    new TGeoTube("INN_PIPEsh", mBeInnerPipeRmin, mBeInnerPipeRmin + mBeInnerPipeThick, mInnerIpHLength);
  TGeoVolume* innerBeTubeVolume = new TGeoVolume("INN_PIPE", innerBeTube, kMedBe);
  innerBeTubeVolume->SetLineColor(kGreen - 9);

  TGeoTube* berylliumTubeVacuum =
    new TGeoTube("INN_PIPEVACUUMsh", 0., mBeInnerPipeRmin, mInnerIpHLength);
  TGeoVolume* innerBerylliumTubeVacuumVolume = new TGeoVolume("INN_PIPEVACUUM", berylliumTubeVacuum, kMedVac);
  innerBerylliumTubeVacuumVolume->SetVisibility(1);
  innerBerylliumTubeVacuumVolume->SetLineColor(kGreen);

  barrel->AddNode(innerBeTubeVolume, 1, new TGeoTranslation(0, 30.f, 0));
  barrel->AddNode(innerBerylliumTubeVacuumVolume, 1, new TGeoTranslation(0, 30.f, 0));
}

void Alice3Pipe::createMaterials()
{
  //
  // Define materials for beam Alice3Pipe
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
  matmgr.Material("ALICE3PIPE", 5, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7);
  matmgr.Medium("ALICE3PIPE", 5, "BE", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Vacuum
  matmgr.Mixture("ALICE3PIPE", 16, "VACUUM$ ", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("ALICE3PIPE", 36, "VACUUM$_NF", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("ALICE3PIPE", 56, "VACUUM$_HC ", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("ALICE3PIPE", 76, "VACUUM$_NFHC", aAir, zAir, dAir1, 4, wAir);

  matmgr.Medium("ALICE3PIPE", 16, "VACUUM", 16, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("ALICE3PIPE", 36, "VACUUM_NF", 36, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("ALICE3PIPE", 56, "VACUUM_HC", 56, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("ALICE3PIPE", 76, "VACUUM_NFHC", 76, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

// ----------------------------------------------------------------------------
FairModule* Alice3Pipe::CloneModule() const { return new Alice3Pipe(*this); }
ClassImp(o2::passive::Alice3Pipe);
