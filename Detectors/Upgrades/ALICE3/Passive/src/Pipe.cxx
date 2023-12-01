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
                       float rMinInnerPipe,
                       float innerThickness,
                       float innerLength,
                       float rMinOuterPipe,
                       float outerThickness,
                       float outerLength)
  : Alice3PassiveBase{name, title},
    mIsTRKActivated{isTRKActivated},
    mIsFT3Activated{isFT3Activated},
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

  // A3IP update
  // Vacuum
  TGeoTube* vacuumBaseInnerPipe = new TGeoTube("INN_PIPEVACUUM_BASEsh", 0., mBeInnerPipeRmin, mInnerIpHLength/2.);
  TGeoTube* vacuumBaseVacuumVessel = new TGeoTube("VACUUM_VESSELVACUUM_BASEsh", mBeInnerPipeRmin, mBeOuterPipeRmin, mOuterIpHLength/2.);
  // Excavate volumes from the vacuum such that there is placew for the TRK barrel layers and FT3 disc layers of the IRIS tracker
  // And the other passive shapes: coldplate, iris tracker vacuum vessel
  TGeoCompositeShape* vacuumComposite;
  TGeoVolume* vacuumVolume;
  TString compositeFormula{"INN_PIPEVACUUM_BASEsh+VACUUM_VESSELVACUUM_BASEsh"};
  TString subtractorsFormula;
  if(!mIsTRKActivated){
    std::vector<TGeoTube*> trkLayerShapes;

    std::vector<std::array<float, 3>> layersQuotas = {std::array<float, 3>{0.5f, 50.f, 50.e-4}, // TODO: Set layers dynamically. {radius, zLen, thickness}
                                                      std::array<float, 3>{1.2f, 50.f, 50.e-4},
                                                      std::array<float, 3>{2.5f, 50.f, 50.e-4}};

    subtractorsFormula = "";                           // First volume to be subctracted (at least one has to be provided)
    for (auto iLayer{0}; iLayer < layersQuotas.size(); ++iLayer) { // Create TRK layers shapes
      
      auto& layerData = layersQuotas[iLayer];
      trkLayerShapes.emplace_back(new TGeoTube(Form("TRKLAYER_%dsh", iLayer), layerData[0], layerData[0] + layerData[2], layerData[1] / 2));
      if(iLayer != 0){subtractorsFormula += Form("+");}
      subtractorsFormula += Form("TRKLAYER_%dsh", iLayer);
    }

    // Escavate vacuum for hosting cold plate
    TGeoTube* coldPlate = new TGeoTube("TRK_COLDPLATEsh", 2.6f, 2.6f + 150.e-3, 50.f / 2);
    subtractorsFormula += "+TRK_COLDPLATEsh";
    // Excavate vacuum for hosting IRIS vacuum vessel
    // IRIS vacuum vessel dimensions:
    // thickness = 150e-3 cm
    // length = 70 cm
    // RIn = 1.8 cm
    // ROut = cold plate ROut + cold plate thickness

    TGeoTube* irisVacuumVesselInner = new TGeoTube("TRK_IRISVACUUMVESSELINNERsh", 0.48f, 0.48f + 150.e-3, 35.f);
    subtractorsFormula += "+TRK_IRISVACUUMVESSELINNERsh";
    TGeoTube* irisVacuumVesselOuter = new TGeoTube("TRK_IRISVACUUMVESSELOUTERsh", 2.6f + 150.e-3, 2.6f + 150.e-3 + 150.e-3, 35.f);
    subtractorsFormula += "+TRK_IRISVACUUMVESSELOUTERsh";
    TGeoTube* irisVacuumVesselNegZSideWall = new TGeoTube("TRK_IRISVACUUMVESSELWALLNEGZSIDEsh", 0.48f, 2.6f + 150.e-3 + 150.e-3, 150.e-3 / 2.);
    TGeoTranslation* posIrisVacVWallNegZSide = new TGeoTranslation("IRISWALLNEGZ", 0., 0., -35.f - 150.e-3 / 2.);
    posIrisVacVWallNegZSide->RegisterYourself();
    subtractorsFormula += "+TRK_IRISVACUUMVESSELWALLNEGZSIDEsh:IRISWALLNEGZ";
    TGeoTube* irisVacuumVesselPosZSideWall = new TGeoTube("TRK_IRISVACUUMVESSELWALLPOSZSIDEsh", 0.48f, 2.6f + 150.e-3 + 150.e-3, 150.e-3 / 2.);
    TGeoTranslation* posIrisVacVWallPosZSide = new TGeoTranslation("IRISWALLPOSZ", 0., 0., 35.f + 150.e-3 / 2.);
    posIrisVacVWallPosZSide->RegisterYourself();
    subtractorsFormula += "+TRK_IRISVACUUMVESSELWALLPOSZSIDEsh:IRISWALLPOSZ";
  }

  if(!mIsFT3Activated){
    std::vector<TGeoTube*> ft3DiscShapes;
    std::vector<TGeoTranslation*> ft3DiscPositions;

    std::vector<std::array<float, 4>> discsQuotas = {std::array<float, 4>{0.5f, 2.5f, 1.e-3, 26.}, // TODO: Set discs dynamically. {rIn, rOut, thickness, zpos}
                                                      std::array<float, 4>{0.5f, 2.5f, 1.e-3, 30.},
                                                      std::array<float, 4>{0.5f, 2.5f, 1.e-3, 34.},
                                                      std::array<float, 4>{0.5f, 2.5f, 1.e-3, -26.},
                                                      std::array<float, 4>{0.5f, 2.5f, 1.e-3, -30.},
                                                      std::array<float, 4>{0.5f, 2.5f, 1.e-3, -34.}};
    TString tempSubtractorsFormula = "";
    if(!mIsTRKActivated){tempSubtractorsFormula = "+";}
    for(auto iDisc{0}; iDisc < discsQuotas.size(); ++iDisc){
      auto& discData = discsQuotas[iDisc];
      ft3DiscShapes.emplace_back(new TGeoTube(Form("FT3DISC_%dsh", iDisc), discData[0], discData[1], discData[2]/2.));
      ft3DiscPositions.emplace_back(new TGeoTranslation(Form("t%d", iDisc), 0., 0., discData[3]));
      ft3DiscPositions[iDisc]->RegisterYourself();
      if(iDisc != 0){tempSubtractorsFormula += "+";}
      tempSubtractorsFormula += Form("FT3DISC_%dsh:t%d", iDisc, iDisc);
    }
    subtractorsFormula += tempSubtractorsFormula;
  }
  if(subtractorsFormula.Length()){
    LOG(info) << "Subtractors formula before : " << subtractorsFormula;
    subtractorsFormula = Form("-(%s)", subtractorsFormula.Data());
    LOG(info) << "Subtractors formula after: " << subtractorsFormula;

    vacuumComposite = new TGeoCompositeShape("VACUUM_BASEsh", (compositeFormula + subtractorsFormula).Data());
    vacuumVolume = new TGeoVolume("VACUUM_BASE", vacuumComposite, kMedVac);
  } else {
    vacuumComposite = new TGeoCompositeShape("VACUUM_BASEsh", compositeFormula.Data());
    vacuumVolume = new TGeoVolume("VACUUM_BASE", vacuumComposite, kMedVac);
  }
  
  // Pipe tubes
  Double_t innerPipeLengthOnePart = mInnerIpHLength/2. - mBeOuterPipeThick - mOuterIpHLength/2.;
  // TGeoTube* innerBePipeNegZSide = new TGeoTube("INN_PIPENEGZ", mBeInnerPipeRmin, mBeInnerPipeRmin + mBeInnerPipeThick, innerPipeLengthOnePart/2.);
  // TGeoTube* innerBePipePosZSide = new TGeoTube("INN_PIPEPOSZ", mBeInnerPipeRmin, mBeInnerPipeRmin + mBeInnerPipeThick, innerPipeLengthOnePart/2.);
  TGeoTube* innerBePipe = new TGeoTube("INN_PIPE", mBeInnerPipeRmin, mBeInnerPipeRmin + mBeInnerPipeThick, innerPipeLengthOnePart/2.);
  TGeoTube* vacuumVesselPipe = new TGeoTube("VACUUM_VESSEL_PIPE", mBeOuterPipeRmin, mBeOuterPipeRmin + mBeOuterPipeThick, mOuterIpHLength/2.);
  // TGeoTube* vacuumVesselWallNegZSide= new TGeoTube("VACUUM_VESSELWALLNEGZ", mBeOuterPipeRmin, mBeOuterPipeRmin + mBeOuterPipeThick, mBeOuterPipeThick/2.);
  // TGeoTube* vacuumVesselWallPosZSide = new TGeoTube("VACUUM_VESSELWALLPOSZ", mBeOuterPipeRmin, mBeOuterPipeRmin + mBeOuterPipeThick, mBeOuterPipeThick/2.);
  TGeoTube* vacuumVesselWall = new TGeoTube("VACUUM_VESSELWALL", mBeOuterPipeRmin, mBeInnerPipeRmin + mBeOuterPipeThick, mBeOuterPipeThick/2.);
  // Pipe positions
  TGeoTranslation* posVacuumVesselWallNegZSide = new TGeoTranslation("WALLNEGZ", 0, 0, -mOuterIpHLength/2. - mBeOuterPipeThick/2.);
  posVacuumVesselWallNegZSide->RegisterYourself();
  TGeoTranslation* posVacuumVesselWallPosZSide = new TGeoTranslation("WALLPOSZ", 0, 0, mOuterIpHLength/2. + mBeOuterPipeThick/2.);
  posVacuumVesselWallPosZSide->RegisterYourself();
  TGeoTranslation* posInnerBePipeNegZSide = new TGeoTranslation("POS_INN_PIPENEGZ", 0, 0, -mOuterIpHLength/2. - mBeOuterPipeThick - innerPipeLengthOnePart/2.);
  posInnerBePipeNegZSide->RegisterYourself();
  TGeoTranslation* posInnerBePipePosZSide = new TGeoTranslation("POS_INN_PIPEPOSZ", 0, 0, mOuterIpHLength/2. + mBeOuterPipeThick + innerPipeLengthOnePart/2.);
  posInnerBePipePosZSide->RegisterYourself();
  // Pipe composite shape and volume
  TString pipeCompositeFormula = "INN_PIPE:POS_INN_PIPENEGZ"
                                  "+INN_PIPE:POS_INN_PIPEPOSZ"
                                  "+VACUUM_VESSELWALL:WALLNEGZ"
                                  "+VACUUM_VESSELWALL:WALLPOSZ"
                                  "+VACUUM_VESSEL_PIPE";
  TGeoCompositeShape* pipeComposite = new TGeoCompositeShape("A3IPsh", pipeCompositeFormula);
  TGeoVolume* pipeVolume = new TGeoVolume("A3IP", pipeComposite, kMedBe);



  // Add everything to the barrel
  barrel->AddNode(vacuumVolume, 1, new TGeoTranslation(0, 30.f, 0));
  barrel->AddNode(pipeVolume, 1, new TGeoTranslation(0, 30.f, 0));

  vacuumVolume->SetLineColor(kGreen + 3);
  pipeVolume->SetLineColor(kGreen + 3);

  TCanvas *c1 = new TCanvas("c1", "c1", 500, 500);
  vacuumVolume->Draw();
  c1->Print("vacuumVolume.pdf");

  TCanvas *c2 = new TCanvas("c2", "c2", 500, 500);
  c2->cd();
  pipeVolume->Draw();
  c2->Print("pipeVolume.pdf");
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
