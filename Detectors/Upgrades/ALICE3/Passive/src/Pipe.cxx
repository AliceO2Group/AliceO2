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
                       bool isTRKActivated,
                       bool isFT3Activated,
                       float pipeRIn,
                       float pipeThickness,
                       float a3ipLength,
                       float vacuumVesselRIn,
                       float vacuumVesselThickness,
                       float vacuumVesselASideLength)
  : Alice3PassiveBase{name, title},
    mIsTRKActivated{isTRKActivated},
    mIsFT3Activated{isFT3Activated},
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
  const TGeoMedium* kMedVac = matmgr.getTGeoMedium("ALICE3_PIPE_VACUUM");

  // Top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoVolume* barrel = gGeoManager->GetVolume("barrel");
  if (!barrel) {
    LOG(fatal) << "Could not find the top volume";
  }

  // We split the naming of the parts if the beam pipe for ALICE 3 into parts
  // - pipe A Side
  // - vacuum vessel (which hosts the primary vacuum and covers all C Side as well)
  // - iris vacuum vessel (which hosts the secondary vacuum)

  // A3IP update
  // Vacuum
  Double_t pipeASideLength = mA3IPLength / 2. - mVacuumVesselThick - mVacuumVesselASideLength;
  Double_t pipeCSideLength = mA3IPLength / 2. + mVacuumVesselASideLength;
  TGeoTube* vacuumBasePipe = new TGeoTube("PIPEVACUUM_BASEsh", 0., mPipeRIn, mA3IPLength / 2.);
  TGeoTube* vacuumBaseVacuumVessel = new TGeoTube("VACUUM_VESSELVACUUM_BASEsh", mPipeRIn, mVacuumVesselRIn, pipeCSideLength / 2.);

  TGeoTranslation* posPipeCSide = new TGeoTranslation("PIPE_CSIDE_POSITION", 0, 0, mVacuumVesselASideLength - pipeCSideLength / 2.);
  posPipeCSide->RegisterYourself();
  // Excavate volumes from the vacuum such that there is place for the TRK barrel layers and FT3 disc layers of the IRIS tracker
  // And the other passive shapes: coldplate, iris tracker vacuum vessel
  TGeoCompositeShape* vacuumComposite;
  TGeoVolume* vacuumVolume;
  TString compositeFormula{"PIPEVACUUM_BASEsh+VACUUM_VESSELVACUUM_BASEsh:PIPE_CSIDE_POSITION"};
  TString subtractorsFormula;

  if (!mIsTRKActivated) {
    std::vector<TGeoTube*> trkLayerShapes;

    std::vector<std::array<float, 3>> layersQuotas = {std::array<float, 3>{0.5f, 50.f, 100.e-4}, // TODO: Set layers dynamically. {radius, zLen, thickness}
                                                      std::array<float, 3>{1.2f, 50.f, 100.e-4},
                                                      std::array<float, 3>{2.5f, 50.f, 100.e-4}};

    for (auto iLayer{0}; iLayer < layersQuotas.size(); ++iLayer) { // Create TRK layers shapes
      auto& layerData = layersQuotas[iLayer];
      trkLayerShapes.emplace_back(new TGeoTube(Form("TRKLAYER_%dsh", iLayer), layerData[0], layerData[0] + layerData[2], layerData[1] / 2.));
      if (iLayer != 0) {
        subtractorsFormula += "+";
      }
      subtractorsFormula += Form("TRKLAYER_%dsh", iLayer);
    }

    // IRIS vacuum vessel and coldplate dimensions
    float coldplateRIn = 2.6f;              // cm
    float coldplateThick = 150.e-3;         // cm
    float coldplateLength = 50.f;           // cm
    float irisVacuumVesselInnerRIn = 0.48f; // cm
    float irisVacuumVesselOuterRIn = coldplateRIn + coldplateThick;
    float irisVacuumVesselLength = 70.f;   // cm
    float irisVacuumVesselThick = 150.e-4; // cm

    // Excavate vacuum for hosting cold plate and IRIS tracker
    TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", coldplateRIn, coldplateRIn + coldplateThick, coldplateLength / 2.);
    subtractorsFormula += "+TRK_COLDPLATEsh";

    TGeoTube* irisVacuumVesselInner = new TGeoTube("TRK_IRISVACUUMVESSELINNERsh", irisVacuumVesselInnerRIn, irisVacuumVesselInnerRIn + irisVacuumVesselThick, irisVacuumVesselLength / 2.);
    subtractorsFormula += "+TRK_IRISVACUUMVESSELINNERsh";

    TGeoTube* irisVacuumVesselOuter = new TGeoTube("TRK_IRISVACUUMVESSELOUTERsh", irisVacuumVesselOuterRIn, irisVacuumVesselOuterRIn + irisVacuumVesselThick, irisVacuumVesselLength / 2.);
    subtractorsFormula += "+TRK_IRISVACUUMVESSELOUTERsh";

    TGeoTube* irisVacuumVesselWall = new TGeoTube("TRK_IRISVACUUMVESSELWALLsh", irisVacuumVesselInnerRIn, irisVacuumVesselOuterRIn + irisVacuumVesselThick, irisVacuumVesselThick / 2.);
    TGeoTranslation* posIrisVacVWallNegZSide = new TGeoTranslation("IRISWALLNEGZ", 0., 0., -irisVacuumVesselLength / 2. - irisVacuumVesselThick / 2.);
    posIrisVacVWallNegZSide->RegisterYourself();
    subtractorsFormula += "+TRK_IRISVACUUMVESSELWALLsh:IRISWALLNEGZ";

    TGeoTranslation* posIrisVacVWallPosZSide = new TGeoTranslation("IRISWALLPOSZ", 0., 0., irisVacuumVesselLength / 2. + irisVacuumVesselThick / 2.);
    posIrisVacVWallPosZSide->RegisterYourself();
    subtractorsFormula += "+TRK_IRISVACUUMVESSELWALLsh:IRISWALLPOSZ";
  }

  if (!mIsFT3Activated) {
    std::vector<TGeoTube*> ft3DiscShapes;
    std::vector<TGeoTranslation*> ft3DiscPositions;

    std::vector<std::array<float, 4>> discsQuotas = {std::array<float, 4>{0.5f, 2.5f, 100.e-4, 26.}, // TODO: Set discs dynamically. {rIn, rOut, thickness, zpos}
                                                     std::array<float, 4>{0.5f, 2.5f, 100.e-4, 30.},
                                                     std::array<float, 4>{0.5f, 2.5f, 100.e-4, 34.},
                                                     std::array<float, 4>{0.5f, 2.5f, 100.e-4, -26.},
                                                     std::array<float, 4>{0.5f, 2.5f, 100.e-4, -30.},
                                                     std::array<float, 4>{0.5f, 2.5f, 100.e-4, -34.}};
    TString tempSubtractorsFormula = "";
    if (!mIsTRKActivated) {
      tempSubtractorsFormula = "+";
    }
    for (auto iDisc{0}; iDisc < discsQuotas.size(); ++iDisc) {
      auto& discData = discsQuotas[iDisc];
      ft3DiscShapes.emplace_back(new TGeoTube(Form("FT3DISC_%dsh", iDisc), discData[0], discData[1], discData[2] / 2.));
      ft3DiscPositions.emplace_back(new TGeoTranslation(Form("t%d", iDisc), 0., 0., discData[3]));
      ft3DiscPositions[iDisc]->RegisterYourself();
      if (iDisc != 0) {
        tempSubtractorsFormula += "+";
      }
      tempSubtractorsFormula += Form("FT3DISC_%dsh:t%d", iDisc, iDisc);
    }
    subtractorsFormula += tempSubtractorsFormula;
  }

  // Pipe tubes
  TGeoTube* pipeASide = new TGeoTube("PIPE_Ash", mPipeRIn, mPipeRIn + mPipeThick, pipeASideLength / 2.);
  TGeoTube* pipeCSide = new TGeoTube("PIPE_Csh", mVacuumVesselRIn, mVacuumVesselRIn + mVacuumVesselThick, pipeCSideLength / 2.);
  TGeoTube* vacuumVesselWall = new TGeoTube("VACUUM_VESSEL_WALLsh", mPipeRIn, mVacuumVesselRIn + mVacuumVesselThick, mVacuumVesselThick / 2.);

  // Pipe and vacuum vessel positions
  TGeoTranslation* posVacuumVesselWall = new TGeoTranslation("WALL_POSITION", 0, 0, mVacuumVesselASideLength + mVacuumVesselThick / 2.);
  posVacuumVesselWall->RegisterYourself();
  TGeoTranslation* posPipeASide = new TGeoTranslation("PIPE_ASIDE_POSITION", 0, 0, mVacuumVesselASideLength + mVacuumVesselThick + pipeASideLength / 2.);
  posPipeASide->RegisterYourself();

  // Pipe composite shape and volume
  TString pipeCompositeFormula =
    "VACUUM_VESSEL_WALLsh:WALL_POSITION"
    "+PIPE_Ash:PIPE_ASIDE_POSITION"
    "+PIPE_Csh:PIPE_CSIDE_POSITION";

  if (subtractorsFormula.Length()) {
    LOG(info) << "Subtractors formula before : " << subtractorsFormula;
    subtractorsFormula = Form("-(%s)", subtractorsFormula.Data());
    LOG(info) << "Subtractors formula after: " << subtractorsFormula;

    vacuumComposite = new TGeoCompositeShape("VACUUM_BASEsh", (compositeFormula + subtractorsFormula).Data());
    vacuumVolume = new TGeoVolume("VACUUM_BASE", vacuumComposite, kMedVac);
  } else {
    vacuumComposite = new TGeoCompositeShape("VACUUM_BASEsh", compositeFormula.Data());
    vacuumVolume = new TGeoVolume("VACUUM_BASE", vacuumComposite, kMedVac);
  }

  TGeoCompositeShape* pipeComposite = new TGeoCompositeShape("A3IPsh", pipeCompositeFormula);
  TGeoVolume* pipeVolume = new TGeoVolume("A3IP", pipeComposite, kMedBe);

  // Add everything to the barrel
  barrel->AddNode(vacuumVolume, 1, new TGeoTranslation(0, 30.f, 0));
  barrel->AddNode(pipeVolume, 1, new TGeoTranslation(0, 30.f, 0));

  vacuumVolume->SetLineColor(kGreen + 3);
  pipeVolume->SetLineColor(kGreen + 3);
}

void Alice3Pipe::createMaterials()
{
  //
  // Define materials for beam Alice3Pipe
  //
  int isxfld = 2;
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

  // Beryllium
  matmgr.Material("ALICE3_PIPE", 5, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7);
  matmgr.Medium("ALICE3_PIPE", 5, "BE", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // Vacuum
  matmgr.Mixture("ALICE3_PIPE", 16, "VACUUM$ ", aAir, zAir, dAir1, 4, wAir);

  matmgr.Medium("ALICE3_PIPE", 16, "VACUUM", 16, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

// ----------------------------------------------------------------------------
FairModule* Alice3Pipe::CloneModule() const { return new Alice3Pipe(*this); }
ClassImp(o2::passive::Alice3Pipe);
