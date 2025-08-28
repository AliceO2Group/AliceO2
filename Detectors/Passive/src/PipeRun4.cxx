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

#include "DetectorsPassive/PipeRun4.h"
#include <DetectorsBase/Detector.h>
#include <DetectorsBase/MaterialManager.h>
#include <TGeoCompositeShape.h>
#include <TGeoCone.h>
#include <TGeoPcon.h>
#include <TGeoTorus.h>
#include <TGeoTube.h>
#include <TGeoEltu.h>
#include <TVirtualMC.h>
#include "TGeoManager.h"  // for TGeoManager, gGeoManager
#include "TGeoMaterial.h" // for TGeoMaterial
#include "TGeoMedium.h"   // for TGeoMedium
#include "TGeoVolume.h"   // for TGeoVolume
#include <TGeoArb8.h>     // for TGeoTrap
#include <TGeoTrd1.h>     // for TGeoTrap
// force availability of assert
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

//-------------------------------------------------------------------------
//  Beam pipe class for ALICE ITS3 & FOCAL upgrade
//  Imported from Pipe class
//  Original Authors:
//  F. Manso
//  A. Morsch
//  R. Tieulent
//  M. Sitta
//-------------------------------------------------------------------------

using namespace o2::passive;

PipeRun4::~PipeRun4() = default;
PipeRun4::PipeRun4() : PassiveBase("PIPE", "") {}
PipeRun4::PipeRun4(const char* name, const char* title, float rho, float thick)
  : PassiveBase(name, title), mBePipeRmax(rho), mBePipeThick(thick)
{
}
PipeRun4::PipeRun4(const PipeRun4& rhs) = default;

PipeRun4& PipeRun4::operator=(const PipeRun4& rhs)
{
  // self assignment
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  PassiveBase::operator=(rhs);

  return *this;
}

void PipeRun4::ConstructGeometry()
{
  createMaterials();
  //
  //  Class describing the beam pipe geometry
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

  // Media
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* kMedAir = matmgr.getTGeoMedium("PIPE_AIR");
  const TGeoMedium* kMedAirNF = matmgr.getTGeoMedium("PIPE_AIR_NF");
  const TGeoMedium* kMedAirHigh = matmgr.getTGeoMedium("PIPE_AIR_HIGH");

  const TGeoMedium* kMedVac = matmgr.getTGeoMedium("PIPE_VACUUM");
  const TGeoMedium* kMedVacNF = matmgr.getTGeoMedium("PIPE_VACUUM_NF");
  const TGeoMedium* kMedVacHC = matmgr.getTGeoMedium("PIPE_VACUUM_HC");
  const TGeoMedium* kMedVacNFHC = matmgr.getTGeoMedium("PIPE_VACUUM_NFHC");

  const TGeoMedium* kMedInsu = matmgr.getTGeoMedium("PIPE_INS_C0");

  const TGeoMedium* kMedSteel = matmgr.getTGeoMedium("PIPE_INOX");
  const TGeoMedium* kMedSteelNF = matmgr.getTGeoMedium("PIPE_INOX_NF");
  const TGeoMedium* kMedSteelHC = matmgr.getTGeoMedium("PIPE_INOX_HC");
  const TGeoMedium* kMedSteelNFHC = matmgr.getTGeoMedium("PIPE_INOX_NFHC");

  const TGeoMedium* kMedBe = matmgr.getTGeoMedium("PIPE_BE");

  const TGeoMedium* kMedCu = matmgr.getTGeoMedium("PIPE_CU");
  const TGeoMedium* kMedCuNF = matmgr.getTGeoMedium("PIPE_CU_NF");
  const TGeoMedium* kMedCuHC = matmgr.getTGeoMedium("PIPE_CU_HC");
  const TGeoMedium* kMedCuNFHC = matmgr.getTGeoMedium("PIPE_CU_NFHC");

  const TGeoMedium* kMedAlu2219 = matmgr.getTGeoMedium("PIPE_AA2219");
  const TGeoMedium* kMedRohacell = matmgr.getTGeoMedium("PIPE_ROHACELL");
  const TGeoMedium* kMedPolyimide = matmgr.getTGeoMedium("PIPE_POLYIMIDE");
  const TGeoMedium* kMedAlBe = matmgr.getTGeoMedium("PIPE_AlBe");
  const TGeoMedium* kMedCarbonFiber = matmgr.getTGeoMedium("PIPE_M55J6K");
  const TGeoMedium* kMedTitanium = matmgr.getTGeoMedium("PIPE_TITANIUM");
  const TGeoMedium* kMedAlu7075 = matmgr.getTGeoMedium("PIPE_AA7075");

  // Top volume
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoVolume* barrel = gGeoManager->GetVolume("barrel");
  TGeoVolume* caveRB24 = gGeoManager->GetVolume("caveRB24");
  //
  //
  ////////////////////////////////////////////////////////////////////////////////
  //                                                                            //
  //                                  The Central Vacuum system                 //
  //                                                                            //
  ////////////////////////////////////////////////////////////////////////////////
  //
  //
  //  The ALICE central beam-pipe according to drawing         LHCVC2C_0001
  //  Drawings of sub-elements:
  //
  //  Pos 7 - Minimised Flange:                                LHCVFX_P0025
  //  Pos 6 - Standard Flange:                                 STDVFUHV0009
  //  Pos 8 - Bellow:                                          LHCVBX__0001
  //
  //  Absolute z-coordinates -82.0 - 400.0 cm
  //  Total length:                                          482.0 cm
  //  It consists of 3 main parts:
  //  CP/2 The flange on the non-absorber side:               36.5 cm
  //  CP/1 The central Be pipe:                              405.0 cm
  //  CP/3 The double-bellow and flange on the absorber side: 40.5 cm

  //
  /*
  //  Starting position in z
  const float kCPz0      = -400.0;
  //  Length of the CP/1 section
  const float kCP1Length =  405.0;
  //  Length of the CP/2 section
  const float kCP2Length =   36.5;
  //  Length of the CP/3 section
  const float kCP3Length =   40.5;
  //  Position of the CP/2 section
  //    const float kCP2pos    = kCPz0 + kCP2Length / 2.;
  //  Position of the CP/3 section
  const float kCP3pos    = kCPz0 + kCP2Length + kCP1Length + kCP3Length/2.;
  */

  ////////////////////        NEW BEAM PIPE GEOMETRY FOR MuonForwardTracker     ////////////////////////
  // Authors: F. Manso, R. Tieulent
  // Drawings from C. Gargiulo :
  // \\cern.ch\dfs\Workspaces\c\cgargiul\EXPERIMENT\ALICE\ALICE_MECHANICS\ALICE_DATA_PACKAGE\IN\DETECTORS\ITS_UPGRADE\1-DESIGN\3D_cad_model\R14_20140311_ALI\
  //
  //
  // central beam pipe
  //------------------- Pipe version 4.7 March 2014 -----------------------------
  TGeoVolumeAssembly* beamPipeCsideSection = new TGeoVolumeAssembly("BeamPipeCsideSection");
  // If user set Rmax=0/Thick=0 use defaults, else use user input
  const float kBeryliumSectionOuterRadius = (mBePipeRmax > 0.) ? mBePipeRmax : 1.65;
  const float kBeryliumSectionThickness = (mBePipeThick > 0.) ? mBePipeThick : 0.05;
  float kBeryliumSectionZmax = 25.0;
  float kBeryliumSectionZmin = -25.0;

  const float kBellowSectionOuterRadius = 2.15;
  const float kCSideBPSOuterRadius = 2.22;
  const float kCSideBPSWallThickness = 0.15;
  const float kBellowSectionZmax = -55.35;
  const float kBellowOuterRadius = 2.8;
  const float kFirstConeAngle = 15. * TMath::DegToRad();
  const float kChangeThicknessAngle = 45. * TMath::DegToRad();
  const float kCSideBPSLength = 3.53;
  const float kDzFirstCone = (kCSideBPSOuterRadius - kBeryliumSectionOuterRadius) / TMath::Tan(kFirstConeAngle);
  const float kReduceThicknessPartAfterBPSLength = 1.52;
  const float kThinPartBeforeBellowLength = 1.025;

  const float kDistanceBetweenBellows = 2.5;

  const float kAdaptConeZmax = -77.43;
  const float kAdaptConeZmin = -80.6;
  const float kAdaptConeRmax = 3.0;
  const float kFlangeRmax = 4.3;
  const float kFlangeLength = 1.4;

  const float kBellowPlieRadius = 0.17;    // radius of bellow plies
  const float kBellowPlieThickness = 0.03; // Thickness of bellow plies 300 microns
  const int kNBellowConvolutions = 7;

  const float kZ1 = kBeryliumSectionZmin;                                                                                                          // z of Be - Al jonction on the C-side
  const float kZ2 = kBellowSectionZmax + kDzFirstCone;                                                                                             // z of end of small diameter part (beginning of first cone before the bellow
  const float kZ3 = kBellowSectionZmax + (kCSideBPSOuterRadius - kBellowSectionOuterRadius) / TMath::Tan(kFirstConeAngle);                         // z of End of first cone part with 0.8mm thickness
  const float kZ4 = kBellowSectionZmax;                                                                                                            // z of End of first Cone
  const float kZ5 = kBellowSectionZmax - kCSideBPSLength;                                                                                          // z of End of Beam Pipe support section
  const float kZ6 = kBellowSectionZmax - kCSideBPSLength - (kCSideBPSOuterRadius - kBellowSectionOuterRadius) / TMath::Tan(kChangeThicknessAngle); // z of End of Beam Pipe support section after reduction of thickness
  const float kZ7 = kZ6 - kReduceThicknessPartAfterBPSLength;                                                                                      // Z of end of 800 microns section after Beam Pipe Support
  const float kZ8 = kZ7 - (kBeryliumSectionThickness - kBellowPlieThickness) / TMath::Tan(kChangeThicknessAngle);
  const float kZ9 = kZ7 - kThinPartBeforeBellowLength; // Z of the start of first bellow
  const float kFirstBellowZmax = kZ9;

  //---------------- Be pipe around the IP ----------
  TGeoTube* berylliumTube =
    new TGeoTube("IP_PIPEsh", kBeryliumSectionOuterRadius - kBeryliumSectionThickness, kBeryliumSectionOuterRadius,
                 (kBeryliumSectionZmax - kBeryliumSectionZmin) / 2);
  TGeoVolume* voberylliumTube = new TGeoVolume("IP_PIPE", berylliumTube, kMedBe);
  voberylliumTube->SetLineColor(kRed);

  TGeoTube* berylliumTubeVacuum =
    new TGeoTube("IP_PIPEVACUUMsh", 0., kBeryliumSectionOuterRadius, (kBeryliumSectionZmax - kBeryliumSectionZmin) / 2);
  TGeoVolume* voberylliumTubeVacuum = new TGeoVolume("IP_PIPEMOTHER", berylliumTubeVacuum, kMedVac);
  voberylliumTubeVacuum->AddNode(voberylliumTube, 1, gGeoIdentity);
  voberylliumTubeVacuum->SetVisibility(0);
  voberylliumTubeVacuum->SetLineColor(kGreen);

  beamPipeCsideSection->AddNode(voberylliumTubeVacuum, 1,
                                new TGeoTranslation(0., 0., (kBeryliumSectionZmax + kBeryliumSectionZmin) / 2));

  //----------------  Al tube ------------------
  TGeoPcon* aluBeforeBellows = new TGeoPcon(0., 360., 9);
  aluBeforeBellows->DefineSection(0, kZ9, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  aluBeforeBellows->DefineSection(1, kZ8, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  aluBeforeBellows->DefineSection(2, kZ7, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  aluBeforeBellows->DefineSection(3, kZ6, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  aluBeforeBellows->DefineSection(4, kZ5, kCSideBPSOuterRadius - kCSideBPSWallThickness, kCSideBPSOuterRadius);
  aluBeforeBellows->DefineSection(5, kZ4, kCSideBPSOuterRadius - kCSideBPSWallThickness, kCSideBPSOuterRadius);
  aluBeforeBellows->DefineSection(6, kZ3, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  aluBeforeBellows->DefineSection(7, kZ2, kBeryliumSectionOuterRadius - kBeryliumSectionThickness, kBeryliumSectionOuterRadius);
  aluBeforeBellows->DefineSection(8, kZ1, kBeryliumSectionOuterRadius - kBeryliumSectionThickness, kBeryliumSectionOuterRadius);
  TGeoVolume* voaluBeforeBellows = new TGeoVolume("aluBeforeBellows", aluBeforeBellows, kMedAlu2219);
  voaluBeforeBellows->SetLineColor(kBlue);
  beamPipeCsideSection->AddNode(voaluBeforeBellows, 1, gGeoIdentity);

  TGeoPcon* aluBeforeBellowsVacuum = new TGeoPcon(0., 360., 7);
  aluBeforeBellowsVacuum->DefineSection(0, kZ9, 0., kBellowSectionOuterRadius - kBeryliumSectionThickness);
  aluBeforeBellowsVacuum->DefineSection(1, kZ6, 0., kBellowSectionOuterRadius - kBeryliumSectionThickness);
  aluBeforeBellowsVacuum->DefineSection(2, kZ5, 0., kCSideBPSOuterRadius - kCSideBPSWallThickness);
  aluBeforeBellowsVacuum->DefineSection(3, kZ4, 0., kCSideBPSOuterRadius - kCSideBPSWallThickness);
  aluBeforeBellowsVacuum->DefineSection(4, kZ3, 0., kBellowSectionOuterRadius - kBeryliumSectionThickness);
  aluBeforeBellowsVacuum->DefineSection(5, kZ2, 0., kBeryliumSectionOuterRadius - kBeryliumSectionThickness);
  aluBeforeBellowsVacuum->DefineSection(6, kZ1, 0., kBeryliumSectionOuterRadius - kBeryliumSectionThickness);
  TGeoVolume* voaluBeforeBellowsVacuum = new TGeoVolume("aluBeforeBellowsVacuum", aluBeforeBellowsVacuum, kMedVac);
  voaluBeforeBellowsVacuum->SetVisibility(1);
  voaluBeforeBellowsVacuum->SetLineColor(kGreen);
  voaluBeforeBellows->AddNode(voaluBeforeBellowsVacuum, 1, gGeoIdentity);
  //-------------------------------------------------

  float kBellowLength = kNBellowConvolutions * (4. * kBellowPlieRadius - 2. * kBellowPlieThickness);
  // ------------------ First Bellow  --------------------
  TGeoVolume* vobellows1 =
    makeBellowCside("bellows1", kNBellowConvolutions, kBellowSectionOuterRadius - kBeryliumSectionThickness,
                    kBellowOuterRadius, kBellowPlieRadius, kBellowPlieThickness);
  beamPipeCsideSection->AddNode(
    vobellows1, 1, new TGeoTranslation(0., 0., kFirstBellowZmax - kBellowLength / 2. - 2. * kBellowPlieRadius));
  //------------------------------------------------------

  const float kZ10 = kFirstBellowZmax - kBellowLength; // End of First bellow
  const float kZ12 = kZ10 - kThinPartBeforeBellowLength;
  const float kZ11 = kZ12 +
                     (kBeryliumSectionThickness - kBellowPlieThickness) /
                       TMath::Tan(kChangeThicknessAngle); // End of 300 microns thickness part after first bellow
  const float kZ13 = kZ12 - kDistanceBetweenBellows;
  const float kZ14 = kZ13 - (kBeryliumSectionThickness - kBellowPlieThickness) / TMath::Tan(kChangeThicknessAngle);
  const float kZ15 = kZ14 - kThinPartBeforeBellowLength;
  const float kSecondBellowZmax = kZ15;

  //---------- Al tube between the bellows ----------
  TGeoPcon* tube4 = new TGeoPcon(0., 360., 6);
  tube4->DefineSection(0, kZ10, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  tube4->DefineSection(1, kZ11, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  tube4->DefineSection(2, kZ12, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  tube4->DefineSection(3, kZ13, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  tube4->DefineSection(4, kZ14, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  tube4->DefineSection(5, kZ15, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  TGeoVolume* votube4 = new TGeoVolume("votube4", tube4, kMedAlu2219);
  votube4->SetLineColor(kBlue);
  beamPipeCsideSection->AddNode(votube4, 1, gGeoIdentity);

  TGeoTube* tube4Vacuum = new TGeoTube(0., kBellowSectionOuterRadius - kBeryliumSectionThickness, -(kZ15 - kZ10) / 2.);
  TGeoVolume* votube4Vacuum = new TGeoVolume("tube4Vacuum", tube4Vacuum, kMedVac);
  votube4Vacuum->SetVisibility(1);
  votube4->AddNode(votube4Vacuum, 1, new TGeoTranslation(0., 0., (kZ10 + kZ15) / 2.));

  // ------------------ Second Bellow --------------------
  TGeoVolume* vobellows2 =
    makeBellowCside("bellows2", kNBellowConvolutions, kBellowSectionOuterRadius - kBeryliumSectionThickness,
                    kBellowOuterRadius, kBellowPlieRadius, kBellowPlieThickness);
  beamPipeCsideSection->AddNode(
    vobellows2, 1, new TGeoTranslation(0., 0., kSecondBellowZmax - kBellowLength / 2. - 2. * kBellowPlieRadius));
  // -----------------------------------------------------

  const float kZ16 = kSecondBellowZmax - kBellowLength; // End of Second bellow
  const float kZ18 = kZ16 - kThinPartBeforeBellowLength;
  const float kZ17 = kZ18 +
                     (kBeryliumSectionThickness - kBellowPlieThickness) /
                       TMath::Tan(kChangeThicknessAngle); // End of 300 microns thickness part after first bellow
  const float kZ19 = kAdaptConeZmax;                      // Start of the Adpation Cone
  const float kZ20 = kAdaptConeZmin;                      // End of the Adpation Cone
  const float kZ21 = kAdaptConeZmin - kFlangeLength;      // End of the Flange

  //----------- 15 deg Conical adaptator + flange ----------
  TGeoPcon* adaptator = new TGeoPcon(0., 360., 7);
  adaptator->DefineSection(0, kZ16, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  adaptator->DefineSection(1, kZ17, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius - kBeryliumSectionThickness + kBellowPlieThickness);
  adaptator->DefineSection(2, kZ18, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  adaptator->DefineSection(3, kZ19, kBellowSectionOuterRadius - kBeryliumSectionThickness, kBellowSectionOuterRadius);
  adaptator->DefineSection(4, kZ20, kBellowSectionOuterRadius - kBeryliumSectionThickness, kAdaptConeRmax);
  adaptator->DefineSection(5, kZ20, kBellowSectionOuterRadius - kBeryliumSectionThickness, kFlangeRmax);
  adaptator->DefineSection(6, kZ21, kBellowSectionOuterRadius - kBeryliumSectionThickness, kFlangeRmax);
  TGeoVolume* voadaptator = new TGeoVolume("voadaptator", adaptator, kMedAlu2219);
  voadaptator->SetLineColor(kBlue);
  beamPipeCsideSection->AddNode(voadaptator, 1, gGeoIdentity);

  TGeoPcon* adaptatorvide = new TGeoPcon(0., 360., 4);
  adaptatorvide->DefineSection(0, kZ16, 0., kBellowSectionOuterRadius - kBeryliumSectionThickness);
  adaptatorvide->DefineSection(1, kZ19, 0., kBellowSectionOuterRadius - kBeryliumSectionThickness);
  adaptatorvide->DefineSection(2, kZ20, 0., kAdaptConeRmax - kBeryliumSectionThickness);
  adaptatorvide->DefineSection(3, kZ21, 0., kAdaptConeRmax - kBeryliumSectionThickness);
  TGeoVolume* voadaptatorvide = new TGeoVolume("voadaptatorvide", adaptatorvide, kMedVac);
  voadaptatorvide->SetVisibility(1);
  //  voadaptatorvide->SetLineColor(kGreen);
  voadaptator->AddNode(voadaptatorvide, 1, gGeoIdentity);
  //------------------------------------------------------

  barrel->AddNode(beamPipeCsideSection, 1, new TGeoTranslation(0., 30., 0.));

  ///////////////////////////////////////////////////////////////////
  //              Beam Pipe support       F.M.     2021  rev 2023  //
  ///////////////////////////////////////////////////////////////////

  // Beam Pipe Support
  TGeoVolume* beamPipeSupport = new TGeoVolumeAssembly("BeamPipeSupport");
  const float kBeamPipesupportZpos = kZ5;

  // Dimensions :
  const float kSupportXdim = 20.67;
  const float kBeamPipeRingZdim = 3.6;
  const float kVespelRmax = 2.3;
  const float kVespelRmin = 2.22;
  const float kBeampipeCarbonCollarRmin = 2.5;
  const float kBeampipeCarbonCollarRmax = 2.7;
  const float kFixationCarbonCollarRmin = 1.5;
  const float kFixationCarbonCollarRmax = 1.7;
  const float kFixationCarbonCollarDZ = 2.5;
  const float kSkinThickness = 0.3;
  const float kSkinXdim = 14.2;
  const float kSkinYdim = 1.4;
  const float kSkinZdim = kFixationCarbonCollarDZ;
  const float kCarbonEarsXdim = 2.8;
  const float kCarbonEarsYdimIn = 1.1;
  const float kCarbonEarsYdimOut = 0.6;
  const float kCarbonEarsZdim = kFixationCarbonCollarDZ;
  const float kScrewDiameter = 0.4;
  const float kScrewHeadHeight = 0.2;
  const float kScrewHeadDiameter = 0.6;
  const float kScrewPositionIn = 3.25;
  const float kScrewPositionOut = 21.80;
  const float kScrewThreadLength = 1.0;
  const float holeSightDiameterOut = 0.60;
  const float holeSightDiameterIn = 0.25;

  // Support Bar
  TGeoVolumeAssembly* supportBar = new TGeoVolumeAssembly("BPS_SupportBar");
  TGeoBBox* carbonSkinBPS = new TGeoBBox("carbonSkinBPS", kSkinXdim / 2., kSkinYdim / 2., kSkinZdim / 2.);
  TGeoBBox* foambarBPS = new TGeoBBox("foambarBPS", kSkinXdim / 2. - kSkinThickness, kSkinYdim / 2. - kSkinThickness,
                                      kSkinZdim / 2. - kSkinThickness / 2.);
  TGeoBBox* carbonEarsBPSin = new TGeoBBox("carbonEarsBPSin", kCarbonEarsXdim / 2., kCarbonEarsYdimIn / 2., kCarbonEarsZdim / 2.);
  TGeoBBox* carbonEarsBPSout = new TGeoBBox("carbonEarsBPSout", kCarbonEarsXdim / 2., kCarbonEarsYdimOut / 2., kCarbonEarsZdim / 2.);

  //===== building the main support bar in carbon ====
  TGeoTranslation* tBP1 = new TGeoTranslation("tBP1", (kSkinXdim + kCarbonEarsXdim) / 2., -(kSkinYdim - kCarbonEarsYdimIn) / 2., 0.);
  TGeoTranslation* tBP2 = new TGeoTranslation("tBP2", -(kSkinXdim + kCarbonEarsXdim) / 2., 0., 0.);
  tBP1->RegisterYourself();
  tBP2->RegisterYourself();

  TGeoRotation* rotScrew = new TGeoRotation("rotScrew", 0., 90., 0.);
  rotScrew->RegisterYourself();

  TGeoTube* holeScrew = new TGeoTube("holeScrew", 0., kScrewDiameter / 2., kCarbonEarsYdimIn / 2. + 0.001);
  TGeoTube* holeSight = new TGeoTube("holeSight", 0., holeSightDiameterOut / 2., kSkinZdim / 2. + 0.001);
  TGeoTranslation* tHoleSight = new TGeoTranslation("tHoleSight", kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax - 6.55, 0., 0.);
  tHoleSight->RegisterYourself();
  double kXHoleIn = kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax - kScrewPositionIn;
  double kXHoleOut = kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax - kScrewPositionOut;
  TGeoCombiTrans* tHoleScrew1 = new TGeoCombiTrans("tHoleScrew1", kXHoleIn, -(kSkinYdim - kCarbonEarsYdimIn) / 2., -0.7, rotScrew);
  TGeoCombiTrans* tHoleScrew2 = new TGeoCombiTrans("tHoleScrew2", kXHoleIn, -(kSkinYdim - kCarbonEarsYdimIn) / 2., 0.7, rotScrew);
  TGeoCombiTrans* tHoleScrew3 = new TGeoCombiTrans("tHoleScrew3", kXHoleOut, -(kSkinYdim - kCarbonEarsYdimIn) / 2., -0.7, rotScrew);
  TGeoCombiTrans* tHoleScrew4 = new TGeoCombiTrans("tHoleScrew4", kXHoleOut, -(kSkinYdim - kCarbonEarsYdimIn) / 2., 0.7, rotScrew);
  tHoleScrew1->RegisterYourself();
  tHoleScrew2->RegisterYourself();
  tHoleScrew3->RegisterYourself();
  tHoleScrew4->RegisterYourself();

  TGeoCompositeShape* supportBarCarbon = new TGeoCompositeShape("BPS_supportBarCarbon", "(carbonSkinBPS-foambarBPS)+carbonEarsBPSin:tBP1-holeScrew:tHoleScrew1-holeScrew:tHoleScrew2+carbonEarsBPSout:tBP2-holeSight:tHoleSight-holeScrew:tHoleScrew3-holeScrew:tHoleScrew4");
  TGeoVolume* supportBarCarbonVol = new TGeoVolume("BPS_supportBarCarbon", supportBarCarbon, kMedCarbonFiber);
  supportBarCarbonVol->SetLineColor(kGray + 2);
  supportBar->AddNode(supportBarCarbonVol, 1, new TGeoTranslation(-(kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax), 0, 0));
  TGeoRotation* rotBar1 = new TGeoRotation("rotBar1", 0., 180., 180.);
  rotBar1->RegisterYourself();
  TGeoCombiTrans* transBar1 = new TGeoCombiTrans("transBar1", kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax, 0, 0, rotBar1);
  transBar1->RegisterYourself();
  supportBar->AddNode(supportBarCarbonVol, 2, transBar1);
  //==================================================

  //==== Adding the internal foam volumes ============
  TGeoCompositeShape* foamVolume = new TGeoCompositeShape("foamVolume", "foambarBPS-holeSight:tHoleSight");
  TGeoVolume* FoamVolume = new TGeoVolume("supportBarFoam", foamVolume, kMedRohacell);
  FoamVolume->SetLineColor(kGreen);
  TGeoRotation* rotBar2 = new TGeoRotation("rotBar2", 0., 0., 180.);
  rotBar2->RegisterYourself();
  TGeoCombiTrans* transBar2 = new TGeoCombiTrans("transBar2", kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax, 0, 0, rotBar2);
  transBar2->RegisterYourself();
  supportBar->AddNode(FoamVolume, 1, transBar1);
  supportBar->AddNode(FoamVolume, 2, new TGeoTranslation(-(kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax), 0, 0));
  //==================================================

  //================= Screws ====================
  TGeoVolumeAssembly* screw = new TGeoVolumeAssembly("screw");
  TGeoTube* headScrew = new TGeoTube("headScrew", 0., kScrewHeadDiameter / 2., kScrewHeadHeight / 2.);
  TGeoVolume* HeadScrew = new TGeoVolume("HeadScrew", headScrew, kMedTitanium);
  HeadScrew->SetLineColor(kRed);
  TGeoTube* threadScrew = new TGeoTube("threadScrew", 0., kScrewDiameter / 2., kCarbonEarsYdimIn / 2.);
  TGeoVolume* ThreadScrew = new TGeoVolume("ThreadScrew", threadScrew, kMedTitanium);
  ThreadScrew->SetLineColor(kRed);
  screw->AddNode(HeadScrew, 1, new TGeoTranslation(0., 0., -(kCarbonEarsYdimIn + kScrewHeadHeight) / 2.));
  screw->AddNode(ThreadScrew, 1);
  TGeoCombiTrans* tScrew1 = new TGeoCombiTrans("transScrew1", kScrewPositionIn, (kCarbonEarsYdimIn - kSkinYdim) / 2., -0.7, rotScrew);
  TGeoCombiTrans* tScrew2 = new TGeoCombiTrans("transScrew2", kScrewPositionIn, (kCarbonEarsYdimIn - kSkinYdim) / 2., 0.7, rotScrew);
  TGeoCombiTrans* tScrew3 = new TGeoCombiTrans("transScrew3", -kScrewPositionIn, (kCarbonEarsYdimIn - kSkinYdim) / 2., -0.7, rotScrew);
  TGeoCombiTrans* tScrew4 = new TGeoCombiTrans("transScrew4", -kScrewPositionIn, (kCarbonEarsYdimIn - kSkinYdim) / 2., 0.7, rotScrew);
  tScrew1->RegisterYourself();
  tScrew2->RegisterYourself();
  tScrew3->RegisterYourself();
  tScrew4->RegisterYourself();
  supportBar->AddNode(screw, 1, tScrew1);
  supportBar->AddNode(screw, 2, tScrew2);
  supportBar->AddNode(screw, 3, tScrew3);
  supportBar->AddNode(screw, 4, tScrew4);
  //==============================================

  // === Optical sights  (assuming the same than the MFT ones) ===
  TGeoVolumeAssembly* fixationSight = new TGeoVolumeAssembly("fixationSight");
  TGeoTube* screwSight = new TGeoTube("screwSight", holeSightDiameterIn / 2., holeSightDiameterOut / 2., kScrewThreadLength / 2.);
  TGeoVolume* ScrewSight = new TGeoVolume("ScrewSight", screwSight, kMedSteel);
  ScrewSight->SetLineColor(kBlue);
  double supportSightLength = 0.5;
  TGeoTube* supportSight = new TGeoTube("supportSight", holeSightDiameterIn / 2., 1.4 / 2., supportSightLength / 2.);
  TGeoVolume* SupportSight = new TGeoVolume("SupportSight", supportSight, kMedSteel);
  SupportSight->SetLineColor(kBlue);
  fixationSight->AddNode(ScrewSight, 1);
  fixationSight->AddNode(SupportSight, 1, new TGeoTranslation(0., 0., (kScrewThreadLength + supportSightLength) / 2.));
  SupportSight->SetVisibility(kTRUE);
  fixationSight->SetVisibility(kTRUE);
  TGeoTranslation* tSight1 = new TGeoTranslation("tSight1", 6.55, 0., (kSkinZdim - kScrewThreadLength) / 2.);
  TGeoTranslation* tSight2 = new TGeoTranslation("tSight2", -6.55, 0., (kSkinZdim - kScrewThreadLength) / 2.);
  tSight1->RegisterYourself();
  tSight2->RegisterYourself();
  supportBar->AddNode(fixationSight, 1, tSight1);
  supportBar->AddNode(fixationSight, 2, tSight2);
  // =====================

  beamPipeSupport->AddNode(supportBar, 1);

  //=======================  Fixation to pipe ========================
  TGeoTube* pipeSupportTubeCarbon = new TGeoTube(kBeampipeCarbonCollarRmin, kBeampipeCarbonCollarRmax, kFixationCarbonCollarDZ / 2.);
  TGeoVolume* FixationToPipeVol = new TGeoVolume("FixationToPipe", pipeSupportTubeCarbon, kMedCarbonFiber);
  FixationToPipeVol->SetLineColor(kGray + 2);
  beamPipeSupport->AddNode(FixationToPipeVol, 1);
  //==================================================================

  //================ Beam Pipe Ring =================
  TGeoVolumeAssembly* beamPipeRing = new TGeoVolumeAssembly("beamPipeRing");
  TGeoTube* beamPipeRingCarbon = new TGeoTube(kVespelRmax, kBeampipeCarbonCollarRmin, kBeamPipeRingZdim / 2.);
  TGeoVolume* beamPipeRingCarbonVol = new TGeoVolume("beamPipeRingCarbon", beamPipeRingCarbon, kMedCarbonFiber);
  beamPipeRingCarbonVol->SetLineColor(kGray + 2);
  beamPipeRing->AddNode(beamPipeRingCarbonVol, 1,
                        new TGeoTranslation(0., 0, (kBeamPipeRingZdim - kFixationCarbonCollarDZ) / 2.));
  TGeoTube* beamPipeRingVespel = new TGeoTube(kVespelRmin, kVespelRmax, (kBeamPipeRingZdim + 0.4) / 2.);
  TGeoVolume* beamPipeRingVespelVol = new TGeoVolume("beamPipeRingVespel", beamPipeRingVespel, kMedPolyimide);
  beamPipeRingVespelVol->SetLineColor(kGreen + 2);
  beamPipeRing->AddNode(beamPipeRingVespelVol, 1,
                        new TGeoTranslation(0., 0, (kBeamPipeRingZdim - kFixationCarbonCollarDZ) / 2.));
  beamPipeSupport->AddNode(beamPipeRing, 1);
  beamPipeSupport->SetVisibility(1);
  beamPipeSupport->IsVisible();
  //==================================================

  //============  Wings   (connecting the support bars to the cage support) ===============
  TGeoVolumeAssembly* Wing = new TGeoVolumeAssembly("Wing");

  // Tige
  double lengthRod = 28.7 - 1.0 - 1.0 - 1.9; // sligtly decreased to accomodate to the fixation pieces
  double diameterRod = 1.815;                // sligtly increased to account of the two ends of the rod
  double xRod = 22.1;
  TGeoTube* Rod = new TGeoTube(0., diameterRod / 2., lengthRod / 2.);
  TGeoVolume* rod = new TGeoVolume("rod", Rod, kMedAlu7075);
  rod->SetLineColor(kGray);

  // Connecteur Tige / Beam support
  double lengthFixRod = 4.0;
  double diameterFixRod = 3.0;
  //---------------------------------------
  TGeoTube* RodBracket = new TGeoTube("RodBracket", 0., diameterFixRod / 2., lengthFixRod / 2.);
  TGeoBBox* BracketPlane = new TGeoBBox("BracketPlane", 3., 3., 3.);
  TGeoTranslation* tBracketPlane = new TGeoTranslation("tBracketPlane", 0., 3. - kCarbonEarsYdimOut / 2., (lengthFixRod + 6.) / 2. - 2.6);
  tBracketPlane->RegisterYourself();
  TGeoCompositeShape* Bracket = new TGeoCompositeShape("Bracket", "RodBracket-BracketPlane:tBracketPlane");
  TGeoVolume* bracket = new TGeoVolume("bracket", Bracket, kMedAlu7075);
  //---------------------------------------

  // Carbon box surrounding the aluminum rod
  TGeoVolumeAssembly* carbonBox = new TGeoVolumeAssembly("carbonBox");
  double eCarbonBox = 0.1;
  double trdWidth = 8.6;
  double trdLength = 11.05 - 1.0 - 0.6; // on each side to accomodate the bracket and TRDPlate
  TGeoTrd1* trdOut = new TGeoTrd1("trdOut", 1.405 / 2, 6.632 / 2, trdLength / 2, trdWidth / 2);
  TGeoTrd1* trdIn = new TGeoTrd1("trdIn", 1.405 / 2 - eCarbonBox, 6.632 / 2 - eCarbonBox, trdLength / 2 + eCarbonBox, trdWidth / 2 - eCarbonBox);
  TGeoCompositeShape* trd = new TGeoCompositeShape("trd", "trdOut-trdIn");
  TGeoVolume* TRD = new TGeoVolume("TRD", trd, kMedCarbonFiber);
  TRD->SetLineColor(kGray);

  // To close the carbon box
  TGeoTrd1* trdPlate = new TGeoTrd1("trdPlate", 1.405 / 2, 6.632 / 2, 1.0 / 2, trdWidth / 2);
  TGeoVolume* TRDPlate = new TGeoVolume("TDRPlate", trdPlate, kMedAlu7075);

  // To connect on the main cage
  TGeoBBox* plateBox = new TGeoBBox("plateBox", 7.5 / 2., 9.5 / 2., 1.9 / 2.);
  TGeoBBox* removeBox = new TGeoBBox("removeBox", 2.1 / 2 + 0.0001, 2.5 / 2. + 0.0001, 1.9 / 2. + 0.0001);
  TGeoTranslation* tRemove1 = new TGeoTranslation("tRemove1", (7.5 - 2.1) / 2, -(9.5 - 2.5) / 2, 0.);
  TGeoTranslation* tRemove2 = new TGeoTranslation("tRemove2", -(7.5 - 2.1) / 2, -(9.5 - 2.5) / 2, 0.);
  tRemove1->RegisterYourself();
  tRemove2->RegisterYourself();

  // Connectors Rod / Cage
  TGeoCompositeShape* PlateBox = new TGeoCompositeShape("PlateBox", "plateBox-removeBox:tRemove1-removeBox:tRemove2");
  TGeoVolume* PLATEBox = new TGeoVolume("PLATEBox", PlateBox, kMedAlu7075);

  TGeoRotation* PlateRot = new TGeoRotation("PlateRot", 0., 0., 0.);
  TGeoRotation* FrontRot = new TGeoRotation("FrontRot", 180., 90., 0.);
  TGeoCombiTrans* tFrontCarbonBox = new TGeoCombiTrans("tFrontCarbonBox", 0., 0., 0., FrontRot);
  PlateRot->RegisterYourself();
  FrontRot->RegisterYourself();
  tFrontCarbonBox->RegisterYourself();
  TGeoCombiTrans* tTRDPlate = new TGeoCombiTrans("tTRDPlate", 0., 0., -(trdLength + 1.0) / 2, FrontRot);
  tTRDPlate->RegisterYourself();
  TRDPlate->SetLineColor(kGray + 2);
  TGeoCombiTrans* tPlateBox = new TGeoCombiTrans("tPlateBox", 0., 0., -(trdLength + 1.9) / 2 - 1.0, PlateRot);
  tPlateBox->RegisterYourself();
  PLATEBox->SetLineColor(kGray);

  double xyOut[16] = {0};
  xyOut[0] = 3.316;
  xyOut[1] = 4.3;
  xyOut[2] = 0.7025;
  xyOut[3] = -xyOut[1];
  xyOut[4] = -xyOut[2];
  xyOut[5] = -xyOut[1];
  xyOut[6] = -xyOut[0];
  xyOut[7] = xyOut[1];
  //--------------
  xyOut[8] = 1.3;
  xyOut[9] = 1.3 - xyOut[1] + xyOut[8];
  xyOut[10] = xyOut[8];
  xyOut[11] = -xyOut[8] - xyOut[1] + xyOut[8];
  xyOut[12] = -xyOut[8];
  xyOut[13] = -xyOut[8] - xyOut[1] + xyOut[8];
  xyOut[14] = -xyOut[8];
  xyOut[15] = xyOut[8] - xyOut[1] + xyOut[8];
  double ARB8Length = 15.35;
  TGeoArb8* ARB8Out = new TGeoArb8("ARB8Out", ARB8Length / 2, xyOut);

  double xyIn[16] = {0};
  xyIn[0] = xyOut[0] - eCarbonBox;
  xyIn[1] = xyOut[1] - eCarbonBox;
  xyIn[2] = 0.7025 - eCarbonBox;
  xyIn[3] = -xyIn[1];
  xyIn[4] = -xyIn[2];
  xyIn[5] = -xyIn[1];
  xyIn[6] = -xyIn[0];
  xyIn[7] = xyIn[1];
  //--------------
  xyIn[8] = xyOut[8] - eCarbonBox;
  xyIn[9] = xyOut[8] - xyIn[1] + xyIn[8] - eCarbonBox;
  xyIn[10] = xyIn[8];
  xyIn[11] = -xyIn[8] - xyOut[1] + xyOut[8];
  xyIn[12] = -xyIn[8];
  xyIn[13] = -xyIn[8] - xyOut[1] + xyOut[8];
  xyIn[14] = -xyIn[8];
  xyIn[15] = xyIn[8] - xyOut[1] + xyOut[8];
  TGeoArb8* ARB8In = new TGeoArb8("ARB8In", ARB8Length / 2 + 0.0001, xyIn);

  TGeoCompositeShape* arb8 = new TGeoCompositeShape("arb8", "ARB8Out-ARB8In");
  TGeoVolume* ARB8 = new TGeoVolume("ARB8", arb8, kMedCarbonFiber);
  ARB8->SetLineColor(kGray);
  TGeoRotation* RearRot = new TGeoRotation("RearRot", 0., 0., 0.);
  TGeoCombiTrans* tRearCarbonBox = new TGeoCombiTrans("tRearCarbonBox", 0., 0., (ARB8Length + trdLength) / 2, RearRot);
  RearRot->RegisterYourself();
  tRearCarbonBox->RegisterYourself();
  //===============================================================

  carbonBox->AddNode(TRD, 1, tFrontCarbonBox);
  carbonBox->AddNode(ARB8, 1, tRearCarbonBox);
  carbonBox->AddNode(TRDPlate, 1, tTRDPlate);
  carbonBox->AddNode(PLATEBox, 1, tPlateBox);

  TGeoRotation* CarbonBoxRot1 = new TGeoRotation("CarbonBoxRot1", 90., 0., 0.);
  double xCarbonBox = xRod + trdWidth / 2 - xyOut[8];
  double zCarbonBox = -trdLength / 2 - ARB8Length - lengthFixRod + 1.3;
  TGeoCombiTrans* tCarbonBox1 = new TGeoCombiTrans("tCarbonBox1", -xCarbonBox, 0., zCarbonBox, CarbonBoxRot1);
  CarbonBoxRot1->RegisterYourself();
  tCarbonBox1->RegisterYourself();
  TGeoRotation* CarbonBoxRot2 = new TGeoRotation("CarbonBoxRot2", 270., 0., 0.);
  TGeoCombiTrans* tCarbonBox2 = new TGeoCombiTrans("tCarbonBox2", xCarbonBox, 0., zCarbonBox, CarbonBoxRot2);
  CarbonBoxRot2->RegisterYourself();
  tCarbonBox2->RegisterYourself();

  Wing->AddNode(rod, 1, new TGeoTranslation(xRod, 0., -(lengthRod / 2. + lengthFixRod) + 1.3));
  Wing->AddNode(rod, 2, new TGeoTranslation(-xRod, 0., -(lengthRod / 2. + lengthFixRod) + 1.3));
  bracket->SetLineColor(kGray);
  Wing->AddNode(bracket, 1, new TGeoTranslation(xRod, 0., -lengthFixRod / 2. + 1.3));
  Wing->AddNode(bracket, 2, new TGeoTranslation(-xRod, 0., -lengthFixRod / 2. + 1.3));
  Wing->AddNode(carbonBox, 1, tCarbonBox1);
  Wing->AddNode(carbonBox, 2, tCarbonBox2);

  beamPipeSupport->AddNode(Wing, 1);
  double mGlobalShift = 2.45; // to be closest to the first bellow according to Corrado blueprints
  barrel->AddNode(beamPipeSupport, 1, new TGeoTranslation(0., 30, kBeamPipesupportZpos + kFixationCarbonCollarDZ / 2. - mGlobalShift));

  ///////////// END NEW BEAM PIPE GEOMETRY FOR MFT ////////////////////

  /////////////////////////////////////////////////////////////////////
  // Side A section after Beryllium
  // Authors: M.Sitta - 19 Sep 2014
  // Drawings from C. Gargiulo :
  // \\cern.ch\dfs\Workspaces\c\cgargiul\EXPERIMENT\ALICE\ALICE_MECHANICS\ALICE_DATA_PACKAGE\IN\DETECTORS\ITS_UPGRADE\1-DESIGN\0-IF_Control_Drawing\20140207_ICD_ITS_MFT_BP
  /////////////////////////////////////////////////////////////////////

  float kConicalBerilliumMinThickness = 0.08;
  float kConicalBerilliumMaxThickness = 0.1;
  float kFlangeZ = 483.75;
  float kFlangeWidth = 2.74;
  float kFlangeThickness = 4.3;
  float kConicalBerylliumEnd = 473.3;
  float kSupport1 = 178.6;
  float kSupport2 = 471.3;
  float kSupportWidth = 5.25;
  float kPipeRadiusAtSupport1 = 2.2;
  float kConicalBePipeEndOuterRadius = 3.0;

  TGeoPcon* tube0 = new TGeoPcon(0., 360., 5);
  tube0->DefineSection(0, kFlangeZ - kFlangeWidth / 2, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness, kConicalBePipeEndOuterRadius);
  tube0->DefineSection(1, kConicalBerylliumEnd, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness, kConicalBePipeEndOuterRadius);
  tube0->DefineSection(2, kSupport1 + kSupportWidth, kPipeRadiusAtSupport1 - kConicalBerilliumMinThickness, kPipeRadiusAtSupport1);
  tube0->DefineSection(3, kSupport1, kPipeRadiusAtSupport1 - kConicalBerilliumMinThickness, kPipeRadiusAtSupport1);
  tube0->DefineSection(4, kBeryliumSectionZmax, kBeryliumSectionOuterRadius - kConicalBerilliumMinThickness, kBeryliumSectionOuterRadius); // need a transition to kConicalBerilliumMaxThickness

  TGeoPcon* tube0vide = new TGeoPcon(0., 360., 5);
  tube0vide->DefineSection(0, kFlangeZ - kFlangeWidth / 2, 0., kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness - 0.01);
  tube0vide->DefineSection(1, kConicalBerylliumEnd, 0., kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness - 0.01);
  tube0vide->DefineSection(2, kSupport1 + kSupportWidth, 0, kPipeRadiusAtSupport1 - kConicalBerilliumMinThickness - 0.01);
  tube0vide->DefineSection(3, kSupport1, 0, kPipeRadiusAtSupport1 - kConicalBerilliumMinThickness - 0.01);
  tube0vide->DefineSection(4, kBeryliumSectionZmax, 0., kBeryliumSectionOuterRadius - kConicalBerilliumMinThickness - 0.01);

  TGeoVolume* votube0 = new TGeoVolume("votube0", tube0, kMedBe);
  votube0->SetLineColor(kRed);
  TGeoVolume* votube0vide = new TGeoVolume("votube0vide", tube0vide, kMedVac);
  votube0vide->SetLineColor(kGreen);

  barrel->AddNode(votube0, 1, new TGeoTranslation(0., 30., 0.));
  barrel->AddNode(votube0vide, 1, new TGeoTranslation(0., 30., 0.));

  TGeoVolume* beampipeSupportA1 = makeSupportBar("A1", kPipeRadiusAtSupport1 + 0.01, kPipeRadiusAtSupport1 + 0.38, 20.67, 14.25);
  barrel->AddNode(beampipeSupportA1, 1, new TGeoTranslation(0., 30, kSupport1 + kSupportWidth / 2.));

  // Length is approximate
  TGeoVolume* beampipeSupportA2 = makeSupportBar("A2", kConicalBePipeEndOuterRadius, kConicalBePipeEndOuterRadius + 0.38, 44, 37.5);
  barrel->AddNode(beampipeSupportA2, 1, new TGeoTranslation(0., 30, kConicalBerylliumEnd + kSupportWidth / 2.));

  TGeoPcon* Bolt1 = new TGeoPcon(0., 360, 8);
  Bolt1->DefineSection(0, 0, 0, 0.5);
  Bolt1->DefineSection(1, 0.515 - 0.01, 0, 0.5);
  Bolt1->DefineSection(2, 0.515 - 0.01, 0, 0.25);
  Bolt1->DefineSection(3, kFlangeWidth + 0.515 + 0.01, 0, 0.25);
  Bolt1->DefineSection(4, kFlangeWidth + 0.515 + 0.01, 0, 0.5);
  Bolt1->DefineSection(5, kFlangeWidth + 0.515 + 0.55, 0, 0.5);
  Bolt1->DefineSection(6, kFlangeWidth + 0.515 + 0.55, 0, 0.25);
  Bolt1->DefineSection(7, kFlangeWidth + 0.515 + 0.55 + 0.5, 0, 0.25);
  Bolt1->SetName("BOLT");

  TGeoVolume* volBolt1 = new TGeoVolume("volBolt1", Bolt1, kMedTitanium);
  volBolt1->SetLineWidth(2);
  volBolt1->SetLineColor(kRed);

  TGeoTranslation* t1 = new TGeoTranslation((kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), (kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t1->SetName("t1");
  t1->RegisterYourself();
  TGeoTranslation* t2 = new TGeoTranslation((kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), (kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t2->SetName("t2");
  t2->RegisterYourself();
  TGeoTranslation* t3 = new TGeoTranslation(-(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), (kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t3->SetName("t3");
  t3->RegisterYourself();
  TGeoTranslation* t4 = new TGeoTranslation(-(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), (kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t4->SetName("t4");
  t4->RegisterYourself();
  TGeoTranslation* t5 = new TGeoTranslation(-(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), -(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t5->SetName("t5");
  t5->RegisterYourself();
  TGeoTranslation* t6 = new TGeoTranslation(-(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), -(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t6->SetName("t6");
  t6->RegisterYourself();
  TGeoTranslation* t7 = new TGeoTranslation((kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), -(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t7->SetName("t7");
  t7->RegisterYourself();
  TGeoTranslation* t8 = new TGeoTranslation((kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Cos(TMath::Pi() / 8), -(kConicalBePipeEndOuterRadius + (kFlangeThickness - kConicalBePipeEndOuterRadius) / 2) * TMath::Sin(TMath::Pi() / 8), kFlangeZ - kFlangeWidth / 2 - 0.515);
  t8->SetName("t8");
  t8->RegisterYourself();

  TGeoVolumeAssembly* Bolts = new TGeoVolumeAssembly("Bolts");
  Bolts->AddNode(volBolt1, 1, t1);
  Bolts->AddNode(volBolt1, 2, t2);
  Bolts->AddNode(volBolt1, 3, t3);
  Bolts->AddNode(volBolt1, 4, t4);
  Bolts->AddNode(volBolt1, 5, t5);
  Bolts->AddNode(volBolt1, 6, t6);
  Bolts->AddNode(volBolt1, 7, t7);
  Bolts->AddNode(volBolt1, 8, t8);

  barrel->AddNode(Bolts, 1, new TGeoTranslation(0., 30., 0.));

  TGeoTranslation* Tflange = new TGeoTranslation(0, 0, kFlangeZ);
  Tflange->SetName("Tflange");
  Tflange->RegisterYourself();

  // Flange
  TGeoTube* flange = new TGeoTube("voFlangeA1", kConicalBePipeEndOuterRadius + 0.01, kFlangeThickness, kFlangeWidth / 2.);

  TGeoPcon* HoleF = new TGeoPcon("HoleF", 0., 360., 2);
  HoleF->DefineSection(0, 0., 0, 0.25 + 0.01);
  HoleF->DefineSection(1, 4.305, 0, 0.25 + 0.01);

  // create the flange with holes for the titanium bolts
  TGeoCompositeShape* FlangeWithHoles = new TGeoCompositeShape("voFlangeWithHoles", "((voFlangeA1:Tflange)-((voFlangeA1:Tflange)*(HoleF:t1+HoleF:t2+HoleF:t3+HoleF:t4+HoleF:t5+HoleF:t6+HoleF:t7+HoleF:t8)))");

  TGeoVolume* volflange = new TGeoVolume("voFlangeHoles", FlangeWithHoles, kMedAlBe);
  volflange->SetLineWidth(2);
  volflange->SetLineColor(kGray);

  barrel->AddNode(volflange, 1, new TGeoTranslation(0., 30., 0.));

  TGeoPcon* pipeSamell = new TGeoPcon(0., 360., 2);
  pipeSamell->DefineSection(0, kFlangeZ + kFlangeWidth / 2, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness, kConicalBePipeEndOuterRadius);
  pipeSamell->DefineSection(1, kFlangeZ + 5.13 + 0.435 + 0.4 + 0.08, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness, kConicalBePipeEndOuterRadius);
  pipeSamell->SetName("pipeSamell");

  TGeoVolume* VolpipeSmall = new TGeoVolume("voPipeSmallVac", pipeSamell, kMedAlu2219);
  VolpipeSmall->SetLineWidth(2);
  barrel->AddNode(VolpipeSmall, 1, new TGeoTranslation(0., 30., 0.));

  TGeoPcon* pipeSmallVac = new TGeoPcon(0., 360., 2);
  pipeSmallVac->DefineSection(0, kFlangeZ + kFlangeWidth / 2, 0, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness - 0.01);
  pipeSmallVac->DefineSection(1, kFlangeZ + 5.13 + 0.435 + 0.4 + 0.08, 0, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness - 0.01);
  TGeoVolume* vopipeSmallVac = new TGeoVolume("voPipeSmallVac", pipeSmallVac, kMedVac);
  vopipeSmallVac->SetLineColor(kGreen);

  barrel->AddNode(vopipeSmallVac, 1, new TGeoTranslation(0., 30., 0.));

  //  -- Bellows on A side
  // float plieradius = (3.72 + (2. *  7 - 2.) * 0.03) / (4. * 7);  // radius of bellows "plis"
  float plieradiusA = 0.2; // radius of bellow plies

  // ------------------ First Bellow  --------------------
  // Inner: 3.0 cm, outer 3.97 cm length 8.47 cm with 10 wiggles
  // check meaning of dU ; it is probably the total length, see also below
  TGeoVolume* vobellows1A = makeBellow("bellows1A", 10, 3.0, 3.97, 8.47, plieradiusA, 0.03);
  // Z position is rough for now.
  barrel->AddNode(vobellows1A, 1, new TGeoTranslation(0., 30., kFlangeZ + 10));
  // Comments: removing 1/2 plie (see makeBellow):  0.31= 2*0.17-0.03    and   0.08: free space

  // ------------------ Outer pipe after flange  --------------------
  TGeoPcon* pipeOut = new TGeoPcon(0., 360., 2);
  pipeOut->DefineSection(0, kFlangeZ + 13.6 - 0.08, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness, kConicalBePipeEndOuterRadius);
  pipeOut->DefineSection(1, 714.6, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness, kConicalBePipeEndOuterRadius);

  TGeoVolume* OuterPIPE = new TGeoVolume("pipeOut", pipeOut, kMedAlu2219);
  barrel->AddNode(OuterPIPE, 1, new TGeoTranslation(0., 30., 0.));

  // The end of the barrel volume is at 714.6 cm, after that we start with RB24 volume
  TGeoPcon* pipeOutVac = new TGeoPcon(0., 360., 2);
  pipeOutVac->DefineSection(0, kFlangeZ + 13.6 - 0.08, 0, kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness);
  pipeOutVac->DefineSection(1, 714.6, 0., kConicalBePipeEndOuterRadius - kConicalBerilliumMaxThickness);

  TGeoVolume* OuterPIPEVac = new TGeoVolume("pipeOutVac", pipeOutVac, kMedAlu2219);
  barrel->AddNode(OuterPIPEVac, 1, new TGeoTranslation(0., 30., 0.));

  //-------------------------------------------------

  ////////////////////////////////////////////////////////////////////////////////
  //                                                                            //
  //                                  RB24/1                                    //
  //                                                                            //
  ////////////////////////////////////////////////////////////////////////////////
  //
  //
  // Drawing LHCVC2U_0001
  // Copper Tube RB24/1      393.5 cm
  // Warm module VMACA        18.0 cm
  // Annular Ion Pump         35.0 cm
  // Valve                     7.5 cm
  // Warm module VMABC        28.0 cm
  // ================================
  //                         462.0 cm
  //

  // Copper Tube RB24/1
  const float kRB24CuTubeL = 381.5;
  const float kRB24cCuTubeL = 155.775 + (28.375 - 18.135);
  const float kRB24bCuTubeL = kRB24CuTubeL - kRB24cCuTubeL;
  const float kRB24CuTubeRi = 5.8 / 2.;
  const float kRB24CuTubeRo = 6.0 / 2.;
  const float kRB24CuTubeFRo = 7.6;
  const float kRB24CuTubeFL = 1.86;
  const float kRB24CL = 2. * 597.9;

  //
  // introduce cut at end of barrel 714.6m
  //
  // outside barrel
  TGeoVolume* voRB24cCuTubeM = new TGeoVolume("voRB24cCuTubeM", new TGeoTube(0., kRB24CuTubeRi, kRB24cCuTubeL / 2.), kMedVacNFHC);
  TGeoVolume* voRB24cCuTube = new TGeoVolume("voRB24cCuTube", new TGeoTube(kRB24CuTubeRi, kRB24CuTubeRo, kRB24cCuTubeL / 2.), kMedAlu2219);
  voRB24cCuTubeM->AddNode(voRB24cCuTube, 1, gGeoIdentity);

  // Air outside tube with higher transport cuts
  TGeoVolume* voRB24CuTubeA = new TGeoVolume("voRB24CuTubeA", new TGeoTube(80., 81., kRB24bCuTubeL / 2.), kMedAirHigh);
  voRB24CuTubeA->SetVisibility(0);

  // Simplified DN 100 Flange
  TGeoVolume* voRB24CuTubeF = new TGeoVolume("voRB24CuTubeF", new TGeoTube(kRB24CuTubeRo, kRB24CuTubeFRo, kRB24CuTubeFL / 2.), kMedSteelNF);

  // Warm Module Type VMACA
  // LHCVMACA_0002
  //
  // Pos 1 Warm Bellows DN100       LHCVBU__0012
  // Pos 2 RF Contact   D80         LHCVSR__0005
  // Pos 3 Trans. Tube Flange       LHCVSR__0065
  // [Pos 4 Hex. Countersunk Screw   Bossard BN4719]
  // [Pos 5 Tension spring           LHCVSR__0011]
  //
  //
  //
  // Pos1    Warm Bellows DN100
  // Pos1.1  Bellows                  LHCVBU__0006
  //
  //
  // Connection Tubes
  // Connection tube inner r
  const float kRB24B1ConTubeRin = 10.0 / 2.;
  // Connection tube outer r
  const float kRB24B1ConTubeRou = 10.3 / 2.;
  // Connection tube length
  const float kRB24B1ConTubeL = 2.5;
  //
  const float kRB24B1CompL = 16.375;        // Length of the compensator
  const float kRB24B1BellowRi = 10.25 / 2.; // Bellow inner radius
  const float kRB24B1BellowRo = 11.40 / 2.; // Bellow outer radius
  const int kRB24B1NumberOfPlies = 27;      // Number of plies
  const float kRB24B1BellowUndL = 11.00;    // Length of undulated region
  const float kRB24B1PlieThickness = 0.015; // Plie thickness

  const float kRB24B1PlieRadius =
    (kRB24B1BellowUndL + (2. * kRB24B1NumberOfPlies - 2.) * kRB24B1PlieThickness) / (4. * kRB24B1NumberOfPlies);

  const float kRB24B1ProtTubeThickness = 0.02; // Thickness of the protection tube
  const float kRB24B1ProtTubeLength = 4.2;     // Length of the protection tube

  const float kRB24B1RFlangeL = 1.86;         // Length of the flanges
  const float kRB24B1RFlangeLO = 0.26;        // Flange overlap
  const float kRB24B1RFlangeRO = 11.18 / 2;   // Inner radius at Flange overlap
  const float kRB24B1RFlangeRou = 15.20 / 2.; // Outer radius of flange
  const float kRB24B1RFlangeRecess = 0.98;    // Flange recess
  const float kRB24B1L = kRB24B1CompL + 2. * (kRB24B1RFlangeL - kRB24B1RFlangeRecess);

  ///
  //
  // Bellow Section
  TGeoVolume* voRB24B1Bellow = makeBellow("RB24B1", kRB24B1NumberOfPlies, kRB24B1BellowRi, kRB24B1BellowRo,
                                          kRB24B1BellowUndL, kRB24B1PlieRadius, kRB24B1PlieThickness);
  voRB24B1Bellow->SetVisibility(0);
  float newRB24B1BellowUndL = 2 * (static_cast<TGeoTube*>(voRB24B1Bellow->GetShape()))->GetDz();

  //
  // Bellow mother volume
  TGeoPcon* shRB24B1BellowM = new TGeoPcon(0., 360., 12);
  // Connection Tube and Flange
  z = 0.;
  shRB24B1BellowM->DefineSection(0, z, 0., kRB24B1RFlangeRou);
  z += kRB24B1RFlangeLO;
  shRB24B1BellowM->DefineSection(1, z, 0., kRB24B1RFlangeRou);
  z = kRB24B1RFlangeL;
  shRB24B1BellowM->DefineSection(2, z, 0., kRB24B1RFlangeRou);
  shRB24B1BellowM->DefineSection(3, z, 0., kRB24B1ConTubeRou);
  z = kRB24B1ConTubeL + kRB24B1RFlangeL - kRB24B1RFlangeRecess;
  shRB24B1BellowM->DefineSection(4, z, 0., kRB24B1ConTubeRou);
  // Plie
  shRB24B1BellowM->DefineSection(5, z, 0., kRB24B1BellowRo + kRB24B1ProtTubeThickness);
  z += newRB24B1BellowUndL;
  shRB24B1BellowM->DefineSection(6, z, 0., kRB24B1BellowRo + kRB24B1ProtTubeThickness);
  shRB24B1BellowM->DefineSection(7, z, 0., kRB24B1ConTubeRou);
  // Connection Tube and Flange
  z = kRB24B1L - shRB24B1BellowM->GetZ(3);
  shRB24B1BellowM->DefineSection(8, z, 0., kRB24B1ConTubeRou);
  shRB24B1BellowM->DefineSection(9, z, 0., kRB24B1RFlangeRou);
  z = kRB24B1L - shRB24B1BellowM->GetZ(1);
  shRB24B1BellowM->DefineSection(10, z, 0., kRB24B1RFlangeRou);
  z = kRB24B1L - shRB24B1BellowM->GetZ(0);
  shRB24B1BellowM->DefineSection(11, z, 0., kRB24B1RFlangeRou);

  TGeoVolume* voRB24B1BellowM = new TGeoVolume("RB24B1BellowM", shRB24B1BellowM, kMedVacNF);
  voRB24B1BellowM->SetVisibility(0);
  //
  // End Parts (connection tube)
  TGeoVolume* voRB24B1CT = new TGeoVolume("RB24B1CT", new TGeoTube(kRB24B1ConTubeRin, kRB24B1ConTubeRou, kRB24B1ConTubeL / 2.), kMedSteelNF);
  //
  // Protection Tube
  TGeoVolume* voRB24B1PT = new TGeoVolume(
    "RB24B1PT", new TGeoTube(kRB24B1BellowRo, kRB24B1BellowRo + kRB24B1ProtTubeThickness, kRB24B1ProtTubeLength / 2.),
    kMedSteelNF);

  z = kRB24B1ConTubeL / 2. + (kRB24B1RFlangeL - kRB24B1RFlangeRecess);

  voRB24B1BellowM->AddNode(voRB24B1CT, 1, new TGeoTranslation(0., 0., z));
  z += (kRB24B1ConTubeL / 2. + newRB24B1BellowUndL / 2.);
  voRB24B1BellowM->AddNode(voRB24B1Bellow, 1, new TGeoTranslation(0., 0., z));
  z += (newRB24B1BellowUndL / 2. + kRB24B1ConTubeL / 2);
  voRB24B1BellowM->AddNode(voRB24B1CT, 2, new TGeoTranslation(0., 0., z));
  z = kRB24B1ConTubeL + kRB24B1ProtTubeLength / 2. + 1. + kRB24B1RFlangeLO;
  voRB24B1BellowM->AddNode(voRB24B1PT, 1, new TGeoTranslation(0., 0., z));
  z += kRB24B1ProtTubeLength + 0.6;
  voRB24B1BellowM->AddNode(voRB24B1PT, 2, new TGeoTranslation(0., 0., z));

  // Pos 1/2 Rotatable Flange         LHCVBU__0013
  // Pos 1/3 Flange DN100/103         LHCVBU__0018
  // The two flanges can be represented by the same volume
  // Outer Radius (including the outer movable ring).
  // The inner ring has a diameter of 12.04 cm

  TGeoPcon* shRB24B1RFlange = new TGeoPcon(0., 360., 10);
  z = 0.;
  shRB24B1RFlange->DefineSection(0, z, 10.30 / 2., kRB24B1RFlangeRou);
  z += 0.55; // 5.5 mm added for outer ring
  z += 0.43;
  shRB24B1RFlange->DefineSection(1, z, 10.30 / 2., kRB24B1RFlangeRou);
  shRB24B1RFlange->DefineSection(2, z, 10.06 / 2., kRB24B1RFlangeRou);
  z += 0.15;
  shRB24B1RFlange->DefineSection(3, z, 10.06 / 2., kRB24B1RFlangeRou);
  // In reality this part is rounded
  shRB24B1RFlange->DefineSection(4, z, 10.91 / 2., kRB24B1RFlangeRou);
  z += 0.15;
  shRB24B1RFlange->DefineSection(5, z, 10.91 / 2., kRB24B1RFlangeRou);
  shRB24B1RFlange->DefineSection(6, z, 10.06 / 2., kRB24B1RFlangeRou);
  z += 0.32;
  shRB24B1RFlange->DefineSection(7, z, 10.06 / 2., kRB24B1RFlangeRou);
  shRB24B1RFlange->DefineSection(8, z, kRB24B1RFlangeRO, kRB24B1RFlangeRou);
  z += kRB24B1RFlangeLO;
  shRB24B1RFlange->DefineSection(9, z, kRB24B1RFlangeRO, kRB24B1RFlangeRou);

  TGeoVolume* voRB24B1RFlange = new TGeoVolume("RB24B1RFlange", shRB24B1RFlange, kMedSteelNF);

  z = kRB24B1L - kRB24B1RFlangeL;
  voRB24B1BellowM->AddNode(voRB24B1RFlange, 1, new TGeoTranslation(0., 0., z));
  z = kRB24B1RFlangeL;
  voRB24B1BellowM->AddNode(voRB24B1RFlange, 2, new TGeoCombiTrans(0., 0., z, rot180));
  //
  // Pos 2 RF Contact   D80         LHCVSR__0005
  //
  // Pos 2.1 RF Contact Flange      LHCVSR__0003
  //
  TGeoPcon* shRB24B1RCTFlange = new TGeoPcon(0., 360., 6);
  const float kRB24B1RCTFlangeRin = 8.06 / 2. + 0.05; // Inner radius
  const float kRB24B1RCTFlangeL = 1.45;               // Length

  z = 0.;
  shRB24B1RCTFlange->DefineSection(0, z, kRB24B1RCTFlangeRin, 8.20 / 2.);
  z += 0.15;
  shRB24B1RCTFlange->DefineSection(1, z, kRB24B1RCTFlangeRin, 8.20 / 2.);
  shRB24B1RCTFlange->DefineSection(2, z, kRB24B1RCTFlangeRin, 8.60 / 2.);
  z += 1.05;
  shRB24B1RCTFlange->DefineSection(3, z, kRB24B1RCTFlangeRin, 8.60 / 2.);
  shRB24B1RCTFlange->DefineSection(4, z, kRB24B1RCTFlangeRin, 11.16 / 2.);
  z += 0.25;
  shRB24B1RCTFlange->DefineSection(5, z, kRB24B1RCTFlangeRin, 11.16 / 2.);
  TGeoVolume* voRB24B1RCTFlange = new TGeoVolume("RB24B1RCTFlange", shRB24B1RCTFlange, kMedCuNF);
  z = kRB24B1L - kRB24B1RCTFlangeL;

  voRB24B1BellowM->AddNode(voRB24B1RCTFlange, 1, new TGeoTranslation(0., 0., z));
  //
  // Pos 2.2 RF-Contact        LHCVSR__0004
  //
  TGeoPcon* shRB24B1RCT = new TGeoPcon(0., 360., 3);
  const float kRB24B1RCTRin = 8.00 / 2.;  // Inner radius
  const float kRB24B1RCTCRin = 8.99 / 2.; // Max. inner radius conical section
  const float kRB24B1RCTL = 11.78;        // Length
  const float kRB24B1RCTSL = 10.48;       // Length of straight section
  const float kRB24B1RCTd = 0.03;         // Thickness

  z = 0;
  shRB24B1RCT->DefineSection(0, z, kRB24B1RCTCRin, kRB24B1RCTCRin + kRB24B1RCTd);
  z = kRB24B1RCTL - kRB24B1RCTSL;
  // In the (VSR0004) this section is straight in (LHCVC2U_0001) it is conical ????
  shRB24B1RCT->DefineSection(1, z, kRB24B1RCTRin + 0.35, kRB24B1RCTRin + 0.35 + kRB24B1RCTd);
  z = kRB24B1RCTL - 0.03;
  shRB24B1RCT->DefineSection(2, z, kRB24B1RCTRin, kRB24B1RCTRin + kRB24B1RCTd);

  TGeoVolume* voRB24B1RCT = new TGeoVolume("RB24B1RCT", shRB24B1RCT, kMedCuNF);
  z = kRB24B1L - kRB24B1RCTL - 0.45;
  voRB24B1BellowM->AddNode(voRB24B1RCT, 1, new TGeoTranslation(0., 0., z));

  //
  // Pos 3 Trans. Tube Flange       LHCVSR__0065
  //
  // Pos 3.1 Transition Tube D53    LHCVSR__0064
  // Pos 3.2 Transition Flange      LHCVSR__0060
  // Pos 3.3 Transition Tube        LHCVSR__0058
  TGeoPcon* shRB24B1TTF = new TGeoPcon(0., 360., 7);
  // Flange
  z = 0.;
  shRB24B1TTF->DefineSection(0, z, 6.30 / 2., 11.16 / 2.);
  z += 0.25;
  shRB24B1TTF->DefineSection(1, z, 6.30 / 2., 11.16 / 2.);
  shRB24B1TTF->DefineSection(2, z, 6.30 / 2., 9.3 / 2.);
  z += 0.55;
  shRB24B1TTF->DefineSection(3, z, 6.30 / 2., 9.3 / 2.);
  // Tube
  shRB24B1TTF->DefineSection(4, z, 6.30 / 2., 6.7 / 2.);
  z += 5.80;
  shRB24B1TTF->DefineSection(5, z, 6.30 / 2., 6.7 / 2.);
  // Transition Tube
  z += 3.75;
  shRB24B1TTF->DefineSection(6, z, 8.05 / 2., 8.45 / 2.);
  TGeoVolume* voRB24B1TTF = new TGeoVolume("RB24B1TTF", shRB24B1TTF, kMedSteelNF);
  z = 0.;
  voRB24B1BellowM->AddNode(voRB24B1TTF, 1, new TGeoTranslation(0., 0., z));

  // Annular Ion Pump
  // LHCVC2U_0003
  //
  // Pos  1 Rotable Flange         LHCVFX__0031
  // Pos  2 RF Screen Tube         LHCVC2U_0005
  // Pos  3 Shell                  LHCVC2U_0007
  // Pos  4 Extruded Shell         LHCVC2U_0006
  // Pos  5 Feedthrough Tube       LHCVC2U_0004
  // Pos  6 Tubulated Flange       STDVFUHV0021
  // Pos  7 Fixed Flange           LHCVFX__0032
  // Pos  8 Pumping Elements

  //
  // Pos 1 Rotable Flange          LHCVFX__0031
  // pos 7 Fixed Flange            LHCVFX__0032
  //
  //  Mother volume

  //
  // Length 35 cm
  // Flange 2 x 1.98 =   3.96
  // Tube            =  32.84
  //==========================
  //                    36.80
  // Overlap 2 * 0.90 =  1.80

  const float kRB24IpRFD1 = 0.68; // Length of section 1
  const float kRB24IpRFD2 = 0.30; // Length of section 2
  const float kRB24IpRFD3 = 0.10; // Length of section 3
  const float kRB24IpRFD4 = 0.35; // Length of section 4
  const float kRB24IpRFD5 = 0.55; // Length of section 5

  const float kRB24IpRFRo = 15.20 / 2.;  // Flange outer radius
  const float kRB24IpRFRi1 = 6.30 / 2.;  // Flange inner radius section 1
  const float kRB24IpRFRi2 = 6.00 / 2.;  // Flange inner radius section 2
  const float kRB24IpRFRi3 = 5.84 / 2.;  // Flange inner radius section 3
  const float kRB24IpRFRi4 = 6.00 / 2.;  // Flange inner radius section 1
  const float kRB24IpRFRi5 = 10.50 / 2.; // Flange inner radius section 2

  TGeoPcon* shRB24IpRF = new TGeoPcon(0., 360., 9);
  z0 = 0.;
  shRB24IpRF->DefineSection(0, z0, kRB24IpRFRi1, kRB24IpRFRo);
  z0 += kRB24IpRFD1;
  shRB24IpRF->DefineSection(1, z0, kRB24IpRFRi2, kRB24IpRFRo);
  z0 += kRB24IpRFD2;
  shRB24IpRF->DefineSection(2, z0, kRB24IpRFRi2, kRB24IpRFRo);
  shRB24IpRF->DefineSection(3, z0, kRB24IpRFRi3, kRB24IpRFRo);
  z0 += kRB24IpRFD3;
  shRB24IpRF->DefineSection(4, z0, kRB24IpRFRi3, kRB24IpRFRo);
  shRB24IpRF->DefineSection(5, z0, kRB24IpRFRi4, kRB24IpRFRo);
  z0 += kRB24IpRFD4;
  shRB24IpRF->DefineSection(6, z0, kRB24IpRFRi4, kRB24IpRFRo);
  shRB24IpRF->DefineSection(7, z0, kRB24IpRFRi5, kRB24IpRFRo);
  z0 += kRB24IpRFD5;
  shRB24IpRF->DefineSection(8, z0, kRB24IpRFRi5, kRB24IpRFRo);

  TGeoVolume* voRB24IpRF = new TGeoVolume("RB24IpRF", shRB24IpRF, kMedSteel);

  //
  // Pos  2 RF Screen Tube         LHCVC2U_0005
  //

  //
  // Tube
  float kRB24IpSTTL = 32.84;      // Total length of the tube
  float kRB24IpSTTRi = 5.80 / 2.; // Inner Radius
  float kRB24IpSTTRo = 6.00 / 2.; // Outer Radius
  TGeoVolume* voRB24IpSTT = new TGeoVolume("RB24IpSTT", new TGeoTube(kRB24IpSTTRi, kRB24IpSTTRo, kRB24IpSTTL / 2.), kMedSteelNF);
  // Screen
  float kRB24IpSTCL = 0.4; // Lenth of the crochet detail
  // Length of the screen
  float kRB24IpSTSL = 9.00 - 2. * kRB24IpSTCL;
  // Rel. position of the screen
  float kRB24IpSTSZ = 7.00 + kRB24IpSTCL;
  TGeoVolume* voRB24IpSTS = new TGeoVolume("RB24IpSTS", new TGeoTube(kRB24IpSTTRi, kRB24IpSTTRo, kRB24IpSTSL / 2.), kMedSteelNF);
  //
  voRB24IpSTT->AddNode(voRB24IpSTS, 1, new TGeoTranslation(0., 0., kRB24IpSTSZ - kRB24IpSTTL / 2. + kRB24IpSTSL / 2.));

  // Crochets
  // Inner radius
  float kRB24IpSTCRi = kRB24IpSTTRo + 0.25;
  // Outer radius
  float kRB24IpSTCRo = kRB24IpSTTRo + 0.35;
  // Length of 1stsection
  float kRB24IpSTCL1 = 0.15;
  // Length of 2nd section
  float kRB24IpSTCL2 = 0.15;
  // Length of 3rd section
  float kRB24IpSTCL3 = 0.10;
  // Rel. position of 1st Crochet

  TGeoPcon* shRB24IpSTC = new TGeoPcon(0., 360., 5);
  z0 = 0;
  shRB24IpSTC->DefineSection(0, z0, kRB24IpSTCRi, kRB24IpSTCRo);
  z0 += kRB24IpSTCL1;
  shRB24IpSTC->DefineSection(1, z0, kRB24IpSTCRi, kRB24IpSTCRo);
  shRB24IpSTC->DefineSection(2, z0, kRB24IpSTTRo, kRB24IpSTCRo);
  z0 += kRB24IpSTCL2;
  shRB24IpSTC->DefineSection(3, z0, kRB24IpSTTRo, kRB24IpSTCRo);
  z0 += kRB24IpSTCL3;
  shRB24IpSTC->DefineSection(4, z0, kRB24IpSTTRo, kRB24IpSTTRo + 0.001);
  TGeoVolume* voRB24IpSTC = new TGeoVolume("RB24IpSTC", shRB24IpSTC, kMedSteel);

  // Pos  3 Shell                  LHCVC2U_0007
  // Pos  4 Extruded Shell         LHCVC2U_0006
  float kRB24IpShellL = 4.45;          // Length of the Shell
  float kRB24IpShellD = 0.10;          // Wall thickness of the shell
  float kRB24IpShellCTRi = 6.70 / 2.;  // Inner radius of the connection tube
  float kRB24IpShellCTL = 1.56;        // Length of the connection tube
  float kRB24IpShellCARi = 17.80 / 2.; // Inner radius of the cavity
  float kRB24IpShellCCRo = 18.20 / 2.; // Inner radius at the centre

  TGeoPcon* shRB24IpShell = new TGeoPcon(0., 360., 7);
  z0 = 0;
  shRB24IpShell->DefineSection(0, z0, kRB24IpShellCTRi, kRB24IpShellCTRi + kRB24IpShellD);
  z0 += kRB24IpShellCTL;
  shRB24IpShell->DefineSection(1, z0, kRB24IpShellCTRi, kRB24IpShellCTRi + kRB24IpShellD);
  shRB24IpShell->DefineSection(2, z0, kRB24IpShellCTRi, kRB24IpShellCARi + kRB24IpShellD);
  z0 += kRB24IpShellD;
  shRB24IpShell->DefineSection(3, z0, kRB24IpShellCARi, kRB24IpShellCARi + kRB24IpShellD);
  z0 = kRB24IpShellL - kRB24IpShellD;
  shRB24IpShell->DefineSection(4, z0, kRB24IpShellCARi, kRB24IpShellCARi + kRB24IpShellD);
  shRB24IpShell->DefineSection(5, z0, kRB24IpShellCARi, kRB24IpShellCCRo);
  z0 = kRB24IpShellL;
  shRB24IpShell->DefineSection(6, z0, kRB24IpShellCARi, kRB24IpShellCCRo);
  TGeoVolume* voRB24IpShell = new TGeoVolume("RB24IpShell", shRB24IpShell, kMedSteel);

  TGeoPcon* shRB24IpShellM = makeMotherFromTemplate(shRB24IpShell, 0, 6, kRB24IpShellCTRi, 13);

  for (int i = 0; i < 6; i++) {
    z = 2. * kRB24IpShellL - shRB24IpShellM->GetZ(5 - i);
    float rmin = shRB24IpShellM->GetRmin(5 - i);
    float rmax = shRB24IpShellM->GetRmax(5 - i);
    shRB24IpShellM->DefineSection(7 + i, z, rmin, rmax);
  }

  TGeoVolume* voRB24IpShellM = new TGeoVolume("RB24IpShellM", shRB24IpShellM, kMedVac);
  voRB24IpShellM->SetVisibility(0);
  voRB24IpShellM->AddNode(voRB24IpShell, 1, gGeoIdentity);
  voRB24IpShellM->AddNode(voRB24IpShell, 2, new TGeoCombiTrans(0., 0., 2. * kRB24IpShellL, rot180));
  //
  // Pos  8 Pumping Elements
  //
  //  Anode array
  TGeoVolume* voRB24IpPE = new TGeoVolume("voRB24IpPE", new TGeoTube(0.9, 1., 2.54 / 2.), kMedSteel);
  float kRB24IpPEAR = 5.5;

  for (int i = 0; i < 15; i++) {
    float phi = float(i) * 24.;
    float x = kRB24IpPEAR * TMath::Cos(kDegRad * phi);
    float y = kRB24IpPEAR * TMath::Sin(kDegRad * phi);
    voRB24IpShellM->AddNode(voRB24IpPE, i + 1, new TGeoTranslation(x, y, kRB24IpShellL));
  }

  //
  // Warm Module Type VMABC
  // LHCVMABC_0002
  //
  //
  //
  // Flange                  1.00
  // Central Piece          11.50
  // Bellow                 14.50
  // End Flange              1.00
  //===================================
  // Total                  28.00
  //
  // Pos 1 Warm Bellows DN100       LHCVBU__0016
  // Pos 2 Trans. Tube Flange       LHCVSR__0062
  // Pos 3 RF Contact   D63         LHCVSR__0057
  // [Pos 4 Hex. Countersunk Screw   Bossard BN4719]
  // [Pos 5 Tension spring           LHCVSR__00239]
  //

  // Pos 1 Warm Bellows DN100                   LHCVBU__0016
  // Pos 1.1 Right Body 2 Ports with Support    LHCVBU__0014
  //
  // Tube 1
  const float kRB24VMABCRBT1Ri = 10.0 / 2.;
  const float kRB24VMABCRBT1Ro = 10.3 / 2.;
  const float kRB24VMABCRBT1L = 11.5;
  const float kRB24VMABCRBT1L2 = 8.;
  const float kRB24VMABCL = 28.375;

  TGeoTube* shRB24VMABCRBT1 = new TGeoTube(kRB24VMABCRBT1Ri, kRB24VMABCRBT1Ro, kRB24VMABCRBT1L / 2.);
  shRB24VMABCRBT1->SetName("RB24VMABCRBT1");
  TGeoTube* shRB24VMABCRBT1o = new TGeoTube(0., kRB24VMABCRBT1Ro, kRB24VMABCRBT1L / 2.);
  shRB24VMABCRBT1o->SetName("RB24VMABCRBT1o");
  TGeoTube* shRB24VMABCRBT1o2 = new TGeoTube(0., kRB24VMABCRBT1Ro + 0.3, kRB24VMABCRBT1L / 2.);
  shRB24VMABCRBT1o2->SetName("RB24VMABCRBT1o2");
  // Lower inforcement
  TGeoVolume* voRB24VMABCRBT12 = new TGeoVolume(
    "RB24VMABCRBT12", new TGeoTubeSeg(kRB24VMABCRBT1Ro, kRB24VMABCRBT1Ro + 0.3, kRB24VMABCRBT1L2 / 2., 220., 320.),
    kMedSteelNF);
  //
  // Tube 2
  const float kRB24VMABCRBT2Ri = 6.0 / 2.;
  const float kRB24VMABCRBT2Ro = 6.3 / 2.;
  const float kRB24VMABCRBF2Ro = 11.4 / 2.;
  const float kRB24VMABCRBT2L = 5.95 + 2.; // 2. cm added for welding
  const float kRB24VMABCRBF2L = 1.75;
  TGeoTube* shRB24VMABCRBT2 = new TGeoTube(kRB24VMABCRBT2Ri, kRB24VMABCRBT2Ro, kRB24VMABCRBT2L / 2.);
  shRB24VMABCRBT2->SetName("RB24VMABCRBT2");
  TGeoTube* shRB24VMABCRBT2i = new TGeoTube(0., kRB24VMABCRBT2Ri, kRB24VMABCRBT2L / 2. + 2.);
  shRB24VMABCRBT2i->SetName("RB24VMABCRBT2i");
  TGeoCombiTrans* tRBT2 = new TGeoCombiTrans(-11.5 + kRB24VMABCRBT2L / 2., 0., 7.2 - kRB24VMABCRBT1L / 2., rotxz);
  tRBT2->SetName("tRBT2");
  tRBT2->RegisterYourself();
  TGeoCompositeShape* shRB24VMABCRBT2c = new TGeoCompositeShape("shRB24VMABCRBT2c", "RB24VMABCRBT2:tRBT2-RB24VMABCRBT1o");
  TGeoVolume* voRB24VMABCRBT2 = new TGeoVolume("shRB24VMABCRBT2", shRB24VMABCRBT2c, kMedSteelNF);
  // Flange
  // Pos 1.4 Flange DN63                        LHCVBU__0008
  TGeoVolume* voRB24VMABCRBF2 =
    new TGeoVolume("RB24VMABCRBF2", new TGeoTube(kRB24VMABCRBT2Ro, kRB24VMABCRBF2Ro, kRB24VMABCRBF2L / 2.), kMedSteelNF);
  // DN63 Blank Flange (my best guess)
  TGeoVolume* voRB24VMABCRBF2B = new TGeoVolume("RB24VMABCRBF2B", new TGeoTube(0., kRB24VMABCRBF2Ro, kRB24VMABCRBF2L / 2.), kMedSteelNF);
  //
  // Tube 3
  const float kRB24VMABCRBT3Ri = 3.5 / 2.;
  const float kRB24VMABCRBT3Ro = 3.8 / 2.;
  const float kRB24VMABCRBF3Ro = 7.0 / 2.;
  const float kRB24VMABCRBT3L = 4.95 + 2.; // 2. cm added for welding
  const float kRB24VMABCRBF3L = 1.27;
  TGeoTube* shRB24VMABCRBT3 = new TGeoTube(kRB24VMABCRBT3Ri, kRB24VMABCRBT3Ro, kRB24VMABCRBT3L / 2);
  shRB24VMABCRBT3->SetName("RB24VMABCRBT3");
  TGeoTube* shRB24VMABCRBT3i = new TGeoTube(0., kRB24VMABCRBT3Ri, kRB24VMABCRBT3L / 2. + 2.);
  shRB24VMABCRBT3i->SetName("RB24VMABCRBT3i");
  TGeoCombiTrans* tRBT3 = new TGeoCombiTrans(0., 10.5 - kRB24VMABCRBT3L / 2., 7.2 - kRB24VMABCRBT1L / 2., rotyz);
  tRBT3->SetName("tRBT3");
  tRBT3->RegisterYourself();
  TGeoCompositeShape* shRB24VMABCRBT3c =
    new TGeoCompositeShape("shRB24VMABCRBT3c", "RB24VMABCRBT3:tRBT3-RB24VMABCRBT1o");
  TGeoVolume* voRB24VMABCRBT3 = new TGeoVolume("shRB24VMABCRBT3", shRB24VMABCRBT3c, kMedSteel);
  // Flange
  // Pos 1.4 Flange DN35                        LHCVBU__0007
  TGeoVolume* voRB24VMABCRBF3 = new TGeoVolume("RB24VMABCRBF3", new TGeoTube(kRB24VMABCRBT3Ro, kRB24VMABCRBF3Ro, kRB24VMABCRBF3L / 2.), kMedSteelNF);
  //
  // Tube 4
  const float kRB24VMABCRBT4Ri = 6.0 / 2.;
  const float kRB24VMABCRBT4Ro = 6.4 / 2.;
  const float kRB24VMABCRBT4L = 6.6;
  TGeoTube* shRB24VMABCRBT4 = new TGeoTube(kRB24VMABCRBT4Ri, kRB24VMABCRBT4Ro, kRB24VMABCRBT4L / 2.);
  shRB24VMABCRBT4->SetName("RB24VMABCRBT4");
  TGeoCombiTrans* tRBT4 = new TGeoCombiTrans(0., -11. + kRB24VMABCRBT4L / 2., 7.2 - kRB24VMABCRBT1L / 2., rotyz);
  tRBT4->SetName("tRBT4");
  tRBT4->RegisterYourself();
  TGeoCompositeShape* shRB24VMABCRBT4c =
    new TGeoCompositeShape("shRB24VMABCRBT4c", "RB24VMABCRBT4:tRBT4-RB24VMABCRBT1o2");
  TGeoVolume* voRB24VMABCRBT4 = new TGeoVolume("shRB24VMABCRBT4", shRB24VMABCRBT4c, kMedSteelNF);
  TGeoCompositeShape* shRB24VMABCRB =
    new TGeoCompositeShape("shRB24VMABCRB", "RB24VMABCRBT1-(RB24VMABCRBT2i:tRBT2+RB24VMABCRBT3i:tRBT3)");
  TGeoVolume* voRB24VMABCRBI = new TGeoVolume("RB24VMABCRBI", shRB24VMABCRB, kMedSteelNF);
  //
  // Plate
  const float kRB24VMABCRBBx = 16.0;
  const float kRB24VMABCRBBy = 1.5;
  const float kRB24VMABCRBBz = 15.0;

  // Relative position of tubes
  const float kRB24VMABCTz = 7.2;
  // Relative position of plate
  const float kRB24VMABCPz = 3.6;
  const float kRB24VMABCPy = -12.5;

  TGeoVolume* voRB24VMABCRBP = new TGeoVolume(
    "RB24VMABCRBP", new TGeoBBox(kRB24VMABCRBBx / 2., kRB24VMABCRBBy / 2., kRB24VMABCRBBz / 2.), kMedSteelNF);
  //
  // Pirani Gauge (my best guess)
  //
  TGeoPcon* shRB24VMABCPirani = new TGeoPcon(0., 360., 15);
  // DN35/16 Coupling
  z = 0;
  shRB24VMABCPirani->DefineSection(0, z, 0.8, kRB24VMABCRBF3Ro);
  z += kRB24VMABCRBF3L; // 1.3
  shRB24VMABCPirani->DefineSection(1, z, 0.8, kRB24VMABCRBF3Ro);
  shRB24VMABCPirani->DefineSection(2, z, 0.8, 1.0);
  // Pipe
  z += 2.8;
  shRB24VMABCPirani->DefineSection(3, z, 0.8, 1.0);
  // Flange
  shRB24VMABCPirani->DefineSection(4, z, 0.8, 1.75);
  z += 1.6;
  shRB24VMABCPirani->DefineSection(5, z, 0.8, 1.75);
  shRB24VMABCPirani->DefineSection(6, z, 0.8, 1.0);
  z += 5.2;
  shRB24VMABCPirani->DefineSection(7, z, 0.8, 1.0);
  shRB24VMABCPirani->DefineSection(8, z, 0.8, 2.5);
  z += 2.0;
  shRB24VMABCPirani->DefineSection(9, z, 0.80, 2.50);
  shRB24VMABCPirani->DefineSection(10, z, 1.55, 1.75);
  z += 5.7;
  shRB24VMABCPirani->DefineSection(11, z, 1.55, 1.75);
  shRB24VMABCPirani->DefineSection(11, z, 0.00, 1.75);
  z += 0.2;
  shRB24VMABCPirani->DefineSection(12, z, 0.00, 1.75);
  shRB24VMABCPirani->DefineSection(13, z, 0.00, 0.75);
  z += 0.5;
  shRB24VMABCPirani->DefineSection(14, z, 0.00, 0.75);
  TGeoVolume* voRB24VMABCPirani = new TGeoVolume("RB24VMABCPirani", shRB24VMABCPirani, kMedSteelNF);
  //
  //
  //

  //
  // Positioning of elements
  TGeoVolumeAssembly* voRB24VMABCRB = new TGeoVolumeAssembly("RB24VMABCRB");
  //
  voRB24VMABCRB->AddNode(voRB24VMABCRBI, 1, gGeoIdentity);
  // Plate
  voRB24VMABCRB->AddNode(voRB24VMABCRBP, 1,
                         new TGeoTranslation(0., kRB24VMABCPy + kRB24VMABCRBBy / 2.,
                                             kRB24VMABCRBBz / 2. - kRB24VMABCRBT1L / 2. + kRB24VMABCPz));
  // Tube 2
  voRB24VMABCRB->AddNode(voRB24VMABCRBT2, 1, gGeoIdentity);
  // Flange Tube 2
  voRB24VMABCRB->AddNode(voRB24VMABCRBF2, 1, new TGeoCombiTrans(kRB24VMABCPy + kRB24VMABCRBF2L / 2., 0., kRB24VMABCTz - kRB24VMABCRBT1L / 2., rotxz));
  // Blank Flange Tube 2
  voRB24VMABCRB->AddNode(voRB24VMABCRBF2B, 1, new TGeoCombiTrans(kRB24VMABCPy - kRB24VMABCRBF2L / 2., 0., kRB24VMABCTz - kRB24VMABCRBT1L / 2., rotxz));
  // Tube 3
  voRB24VMABCRB->AddNode(voRB24VMABCRBT3, 1, gGeoIdentity);
  // Flange Tube 3
  voRB24VMABCRB->AddNode(voRB24VMABCRBF3, 1, new TGeoCombiTrans(0., 11.2 - kRB24VMABCRBF3L / 2., kRB24VMABCTz - kRB24VMABCRBT1L / 2., rotyz));
  // Pirani Gauge
  voRB24VMABCRB->AddNode(voRB24VMABCPirani, 1, new TGeoCombiTrans(0., 11.2, kRB24VMABCTz - kRB24VMABCRBT1L / 2., rotyz));
  // Tube 4
  voRB24VMABCRB->AddNode(voRB24VMABCRBT4, 1, gGeoIdentity);
  // Inforcement
  voRB24VMABCRB->AddNode(voRB24VMABCRBT12, 1, new TGeoTranslation(0., 0., kRB24VMABCRBT1L2 / 2. - kRB24VMABCRBT1L / 2. + 2.8));

  // Pos 1.3 Bellows with end part              LHCVBU__0002
  //
  // Connection Tube
  // Connection tube inner r
  const float kRB24VMABBEConTubeRin = 10.0 / 2.;
  // Connection tube outer r
  const float kRB24VMABBEConTubeRou = 10.3 / 2.;
  // Connection tube length
  const float kRB24VMABBEConTubeL1 = 0.9;
  const float kRB24VMABBEConTubeL2 = 2.6;

  // Mother volume
  TGeoPcon* shRB24VMABBEBellowM = new TGeoPcon(0., 360., 6);
  // Connection Tube and Flange
  z = 0.;
  shRB24VMABBEBellowM->DefineSection(0, z, kRB24VMABBEConTubeRin, kRB24VMABBEConTubeRou);
  z += kRB24VMABBEConTubeL1;
  shRB24VMABBEBellowM->DefineSection(1, z, kRB24VMABBEConTubeRin, kRB24VMABBEConTubeRou);
  shRB24VMABBEBellowM->DefineSection(2, z, kRB24B1BellowRi, kRB24B1BellowRo + kRB24B1ProtTubeThickness);
  z += newRB24B1BellowUndL;
  shRB24VMABBEBellowM->DefineSection(3, z, kRB24B1BellowRi, kRB24B1BellowRo + kRB24B1ProtTubeThickness);
  shRB24VMABBEBellowM->DefineSection(4, z, kRB24VMABBEConTubeRin, kRB24VMABBEConTubeRou);
  z += kRB24VMABBEConTubeL2;
  shRB24VMABBEBellowM->DefineSection(5, z, kRB24VMABBEConTubeRin, kRB24VMABBEConTubeRou);
  TGeoVolume* voRB24VMABBEBellowM = new TGeoVolume("RB24VMABBEBellowM", shRB24VMABBEBellowM, kMedVacNF);
  voRB24VMABBEBellowM->SetVisibility(0);

  //  Connection tube left
  TGeoVolume* voRB24VMABBECT1 = new TGeoVolume(
    "RB24VMABBECT1", new TGeoTube(kRB24VMABBEConTubeRin, kRB24VMABBEConTubeRou, kRB24VMABBEConTubeL1 / 2.), kMedSteelNF);
  //  Connection tube right
  TGeoVolume* voRB24VMABBECT2 = new TGeoVolume(
    "RB24VMABBECT2", new TGeoTube(kRB24VMABBEConTubeRin, kRB24VMABBEConTubeRou, kRB24VMABBEConTubeL2 / 2.), kMedSteelNF);
  z = kRB24VMABBEConTubeL1 / 2.;
  voRB24VMABBEBellowM->AddNode(voRB24VMABBECT1, 1, new TGeoTranslation(0., 0., z));
  z += kRB24VMABBEConTubeL1 / 2.;
  z += newRB24B1BellowUndL / 2.;
  voRB24VMABBEBellowM->AddNode(voRB24B1Bellow, 2, new TGeoTranslation(0., 0., z));
  z += newRB24B1BellowUndL / 2.;
  z += kRB24VMABBEConTubeL2 / 2.;
  voRB24VMABBEBellowM->AddNode(voRB24VMABBECT2, 1, new TGeoTranslation(0., 0., z));
  z += kRB24VMABBEConTubeL2 / 2.;

  voRB24VMABCRB->AddNode(voRB24VMABBEBellowM, 1, new TGeoTranslation(0., 0., kRB24VMABCRBT1L / 2.));

  // Pos 1.2 Rotable flange                     LHCVBU__0013[*]
  // Front
  voRB24VMABCRB->AddNode(voRB24B1RFlange, 3, new TGeoCombiTrans(0., 0., -kRB24VMABCRBT1L / 2. + 0.86, rot180));
  // End
  z = kRB24VMABCRBT1L / 2. + newRB24B1BellowUndL + kRB24VMABBEConTubeL1 + kRB24VMABBEConTubeL2;
  voRB24VMABCRB->AddNode(voRB24B1RFlange, 4, new TGeoTranslation(0., 0., z - 0.86));

  // Pos 2    Trans. Tube Flange       LHCVSR__0062
  // Pos 2.1  Transition Tube          LHCVSR__0063
  // Pos 2.2  Transition Flange        LHCVSR__0060
  //
  // Transition Tube with Flange
  TGeoPcon* shRB24VMABCTT = new TGeoPcon(0., 360., 7);
  z = 0.;
  shRB24VMABCTT->DefineSection(0, z, 6.3 / 2., 11.16 / 2.);
  z += 0.25;
  shRB24VMABCTT->DefineSection(1, z, 6.3 / 2., 11.16 / 2.);
  shRB24VMABCTT->DefineSection(2, z, 6.3 / 2., 9.30 / 2.);
  z += 0.25;
  shRB24VMABCTT->DefineSection(3, z, 6.3 / 2., 9.30 / 2.);
  shRB24VMABCTT->DefineSection(4, z, 6.3 / 2., 6.70 / 2.);
  z += (20.35 - 0.63);
  shRB24VMABCTT->DefineSection(5, z, 6.3 / 2., 6.7 / 2.);
  z += 0.63;
  shRB24VMABCTT->DefineSection(6, z, 6.3 / 2., 6.7 / 2.);
  TGeoVolume* voRB24VMABCTT = new TGeoVolume("RB24VMABCTT", shRB24VMABCTT, kMedSteelNF);
  voRB24VMABCRB->AddNode(voRB24VMABCTT, 1, new TGeoTranslation(0., 0., -kRB24VMABCRBT1L / 2. - 1.));

  // Pos 3   RF Contact   D63         LHCVSR__0057
  // Pos 3.1 RF Contact Flange        LHCVSR__0017
  //
  TGeoPcon* shRB24VMABCCTFlange = new TGeoPcon(0., 360., 6);
  const float kRB24VMABCCTFlangeRin = 6.36 / 2.; // Inner radius
  const float kRB24VMABCCTFlangeL = 1.30;        // Length

  z = 0.;
  shRB24VMABCCTFlange->DefineSection(0, z, kRB24VMABCCTFlangeRin, 6.5 / 2.);
  z += 0.15;
  shRB24VMABCCTFlange->DefineSection(1, z, kRB24VMABCCTFlangeRin, 6.5 / 2.);
  shRB24VMABCCTFlange->DefineSection(2, z, kRB24VMABCCTFlangeRin, 6.9 / 2.);
  z += 0.9;
  shRB24VMABCCTFlange->DefineSection(3, z, kRB24VMABCCTFlangeRin, 6.9 / 2.);
  shRB24VMABCCTFlange->DefineSection(4, z, kRB24VMABCCTFlangeRin, 11.16 / 2.);
  z += 0.25;
  shRB24VMABCCTFlange->DefineSection(5, z, kRB24VMABCCTFlangeRin, 11.16 / 2.);
  TGeoVolume* voRB24VMABCCTFlange = new TGeoVolume("RB24VMABCCTFlange", shRB24VMABCCTFlange, kMedCuNF);
  //
  // Pos 3.2 RF-Contact        LHCVSR__0056
  //
  TGeoPcon* shRB24VMABCCT = new TGeoPcon(0., 360., 4);
  const float kRB24VMABCCTRin = 6.30 / 2.;  // Inner radius
  const float kRB24VMABCCTCRin = 7.29 / 2.; // Max. inner radius conical section
  const float kRB24VMABCCTL = 11.88;        // Length
  const float kRB24VMABCCTSL = 10.48;       // Length of straight section
  const float kRB24VMABCCTd = 0.03;         // Thickness
  z = 0;
  shRB24VMABCCT->DefineSection(0, z, kRB24VMABCCTCRin, kRB24VMABCCTCRin + kRB24VMABCCTd);
  z = kRB24VMABCCTL - kRB24VMABCCTSL;
  shRB24VMABCCT->DefineSection(1, z, kRB24VMABCCTRin + 0.35, kRB24VMABCCTRin + 0.35 + kRB24VMABCCTd);
  z = kRB24VMABCCTL - kRB24VMABCCTFlangeL;
  shRB24VMABCCT->DefineSection(2, z, kRB24VMABCCTRin, kRB24VMABCCTRin + kRB24VMABCCTd);
  z = kRB24VMABCCTL;
  shRB24VMABCCT->DefineSection(3, z, kRB24VMABCCTRin, kRB24VMABCCTRin + kRB24VMABCCTd);

  TGeoVolume* voRB24VMABCCT = new TGeoVolume("RB24VMABCCT", shRB24VMABCCT, kMedCuNF);

  TGeoVolumeAssembly* voRB24VMABRFCT = new TGeoVolumeAssembly("RB24VMABRFCT");
  voRB24VMABRFCT->AddNode(voRB24VMABCCT, 1, gGeoIdentity);
  voRB24VMABRFCT->AddNode(voRB24VMABCCTFlange, 1, new TGeoTranslation(0., 0., kRB24VMABCCTL - kRB24VMABCCTFlangeL));

  z = kRB24VMABCRBT1L / 2. + newRB24B1BellowUndL + kRB24VMABBEConTubeL1 + kRB24VMABBEConTubeL2 - kRB24VMABCCTL + 1.;
  voRB24VMABCRB->AddNode(voRB24VMABRFCT, 1, new TGeoTranslation(0., 0., z));

  //
  // Assembling RB24/1
  //

  // part which is placed in the cave
  // ->
  TGeoVolumeAssembly* voRB24C = new TGeoVolumeAssembly("RB24C");
  voRB24C->AddNode(voRB24cCuTubeM, 1, gGeoIdentity);
  z = -kRB24cCuTubeL / 2 + kRB24CuTubeFL / 2.;
  voRB24C->AddNode(voRB24CuTubeF, 1, new TGeoTranslation(0., 0., z));
  // VMABC close to compensator magnet
  // z = -kRB24cCuTubeL / 2. - (kRB24VMABCL - kRB24VMABCRBT1L / 2) + 1.;
  // voRB24C->AddNode(voRB24VMABCRB, 2, new TGeoTranslation(0., 0., z));
  z = -kRB24cCuTubeL / 2. - kRB24B1L;
  voRB24C->AddNode(voRB24B1BellowM, 2, new TGeoTranslation(0., 0., z));

  // <-

  //
  //   RB24/2
  //
  // Copper Tube RB24/2
  // mainly inside the compensator magnet
  const float kRB242CuTubeL = 350.0;
  // 20 cm straight - 20 cm transition to final oval - 270 oval - 20 cm transition to final oval - 20 cm straight
  //
  // mother volume for transition region
  TGeoVolume* voRB242CuOvTransMo = new TGeoVolume("voRB24CuOvTransMo", new TGeoTube(0., 4.75, 10.), kMedAir);
  const int nTrans = 10;
  TGeoVolume* voRB242CuOvTransV[nTrans];
  TGeoVolume* voRB242CuOvTransI[nTrans];
  float dovX = 4.;
  float dovY = 4.;
  float dovZ = -9.0;
  for (int i = 0; i < nTrans; i++) {
    dovX -= 0.0625;
    dovY += 0.075;
    char vname[20];
    snprintf(vname, 20, "voRB242CuOvTransV%d", i);
    voRB242CuOvTransV[i] = new TGeoVolume(vname, new TGeoEltu(dovX, dovY, 1.0), kMedCuHC);
    snprintf(vname, 20, "voRB242CuOvTransI%d", i);
    voRB242CuOvTransI[i] = new TGeoVolume(vname, new TGeoEltu(dovX - 0.2, dovY - 0.2, 1.0), kMedVacHC);
    voRB242CuOvTransV[i]->AddNode(voRB242CuOvTransI[i], 1, gGeoIdentity);
    voRB242CuOvTransMo->AddNode(voRB242CuOvTransV[i], 1, new TGeoTranslation(0., 0., dovZ));
    dovZ += 2.;
  }
  //
  TGeoVolume* voRB242CuTubeM = new TGeoVolume("voRB242CuTubeM", new TGeoTube(0., kRB24CuTubeRo, 10.), kMedVacHC);
  TGeoVolume* voRB242CuTube = new TGeoVolume("voRB242CuTube", new TGeoTube(kRB24CuTubeRi, kRB24CuTubeRo, 10.), kMedCuHC);
  voRB242CuTubeM->AddNode(voRB242CuTube, 1, gGeoIdentity);
  TGeoVolume* voRB242CuOvalM = new TGeoVolume("voRB242CuOvalM", new TGeoEltu(3.375, 4.75, 135.), kMedCuHC);
  TGeoVolume* voRB242CuOval = new TGeoVolume("voRB242CuOval", new TGeoEltu(3.175, 4.55, 135.), kMedVacHC);
  voRB242CuOvalM->AddNode(voRB242CuOval, 1, gGeoIdentity);
  //
  TGeoVolumeAssembly* voRB242 = new TGeoVolumeAssembly("RB242");
  voRB242->AddNode(voRB242CuOvalM, 1, gGeoIdentity);
  z = -kRB242CuTubeL / 2 + kRB24CuTubeFL / 2.;
  voRB242->AddNode(voRB24CuTubeF, 3, new TGeoTranslation(0., 0., z));
  z = +kRB242CuTubeL / 2 - kRB24CuTubeFL / 2.;
  voRB242->AddNode(voRB24CuTubeF, 4, new TGeoTranslation(0., 0., z));
  z = 135. + 10.;
  voRB242->AddNode(voRB242CuOvTransMo, 1, new TGeoCombiTrans(0., 0., z, rot180));
  z = -135. - 10.;
  voRB242->AddNode(voRB242CuOvTransMo, 2, new TGeoTranslation(0., 0., z));
  z = -135. - 30.;
  voRB242->AddNode(voRB242CuTubeM, 1, new TGeoTranslation(0., 0., z));
  z = 135. + 30.;
  voRB242->AddNode(voRB242CuTubeM, 2, new TGeoTranslation(0., 0., z));
  z = -kRB24cCuTubeL / 2 - kRB24B1L - kRB242CuTubeL / 2.;
  voRB24C->AddNode(voRB242, 1, new TGeoTranslation(0., 0., z));
  //
  //   RB24/3
  //
  // Copper Tube RB24/3
  // the lenth of the tube is 296.85 on the drawing but this is inconsistent with the total length tube + bellow
  const float kRB243CuTubeL = 297.85 - (kRB24VMABCL - kRB24B1L);

  TGeoVolume* voRB243CuTubeM = new TGeoVolume("voRB243CuTubeM", new TGeoTube(0., kRB24CuTubeRo, (kRB243CuTubeL) / 2.), kMedVacNF);
  TGeoVolume* voRB243CuTube = new TGeoVolume("voRB243CuTube", new TGeoTube(kRB24CuTubeRi, kRB24CuTubeRo, (kRB243CuTubeL) / 2.), kMedCuNF);
  voRB243CuTubeM->AddNode(voRB243CuTube, 1, gGeoIdentity);

  TGeoVolumeAssembly* voRB243 = new TGeoVolumeAssembly("RB243");
  TGeoVolumeAssembly* voRB243A = new TGeoVolumeAssembly("RB243A");

  voRB243A->AddNode(voRB243CuTube, 1, gGeoIdentity);
  z = -kRB243CuTubeL / 2 + kRB24CuTubeFL / 2.;
  voRB243A->AddNode(voRB24CuTubeF, 5, new TGeoTranslation(0., 0., z));
  z = +kRB243CuTubeL / 2 - kRB24CuTubeFL / 2.;
  voRB243A->AddNode(voRB24CuTubeF, 6, new TGeoTranslation(0., 0., z));

  z = +kRB243CuTubeL / 2 + (kRB24VMABCRBT1L / 2) + 1;
  voRB243A->AddNode(voRB24VMABCRB, 2, new TGeoTranslation(0., 0., z));

  z = -kRB243CuTubeL / 2. - kRB24VMABCL;
  voRB243->AddNode(voRB243A, 1, new TGeoTranslation(0., 0., z));
  z = -(1.5 * kRB243CuTubeL + 2. * kRB24VMABCL);
  voRB243->AddNode(voRB243A, 2, new TGeoTranslation(0., 0., z));

  z = -2. * (kRB243CuTubeL + kRB24VMABCL) - (kRB24VMABCL - kRB24VMABCRBT1L / 2) + 1.;
  voRB243->AddNode(voRB24VMABCRB, 3, new TGeoTranslation(0., 0., z));

  z = -kRB24cCuTubeL / 2 - kRB24B1L - kRB242CuTubeL;
  voRB24C->AddNode(voRB243, 1, new TGeoTranslation(0., 0., z));

  //
  //
  caveRB24->AddNode(voRB24C, 1, new TGeoCombiTrans(0., 0., -kRB24CL / 2 + kRB24cCuTubeL / 2, rot180));

  //
  ////////////////////////////////////////////////////////////////////////////////
  //                                                                            //
  //                                  The Absorber Vacuum system                //
  //                                                                            //
  ////////////////////////////////////////////////////////////////////////////////
  //
  //    Rotable Flange starts at:            82.00 cm from IP
  //    Length of rotable flange section:    10.68 cm
  //    Weld                                  0.08 cm
  //    Length of straight section          207.21 cm
  //    =======================================================================
  //                                        299.97 cm  [0.03 cm missing ?]
  //    Length of opening cone              252.09 cm
  //    Weld                                  0.15 cm
  //    Length of compensator                30.54 cm
  //    Weld                                  0.15 cm
  //    Length of fixed flange  2.13 - 0.97   1.16 cm
  //    =======================================================================
  //                                        584.06 cm [584.80 installed] [0.74 cm missing]
  //    RB26/3
  //    Length of split flange  2.13 - 1.2    0.93 cm
  //    Weld                                  0.15 cm
  //    Length of fixed point section        16.07 cm
  //    Weld                                  0.15 cm
  //    Length of opening cone              629.20 cm
  //    Weld                                  0.30 cm
  //    Kength of the compensator            41.70 cm
  //    Weld                                  0.30 cm
  //    Length of fixed flange  2.99 - 1.72   1.27 cm
  // =================================================
  //    Length of RB26/3                    690.07 cm [689.20 installed] [0.87 cm too much]
  //
  //    RB26/4-5
  //    Length of split flange  2.13 - 1.2    0.93 cm
  //    Weld                                  0.15 cm
  //    Length of fixed point section        16.07 cm
  //    Weld                                  0.15 cm
  //    Length of opening cone              629.20 cm
  //    Weld                                  0.30 cm
  //    Length of closing cone
  //    Weld
  //    Lenth of straight section
  //    Kength of the compensator            41.70 cm
  //    Weld                                  0.30 cm
  //    Length of fixed flange  2.99 - 1.72   1.27 cm
  // =================================================
  //    Length of RB26/3                    690.07 cm [689.20 installed] [0.87 cm too much]

  ///////////////////////////////////////////
  //                                       //
  //    RB26/1-2                           //
  //    Drawing LHCV2a_0050 [as installed] //
  //    Drawing LHCV2a_0008                //
  //    Drawing LHCV2a_0001                //
  ///////////////////////////////////////////
  //    Pos1 Vacuum Tubes   LHCVC2A__0010
  //    Pos2 Compensator    LHCVC2A__0064
  //    Pos3 Rotable Flange LHCVFX___0016
  //    Pos4 Fixed Flange   LHCVFX___0006
  //    Pos5 Bellow Tooling LHCVFX___0003
  //
  //
  //
  ///////////////////////////////////
  //    RB26/1-2 Vacuum Tubes      //
  //    Drawing  LHCVC2a_0010      //
  ///////////////////////////////////
  const float kRB26s12TubeL0 = 459.45;                         // 0.15 cm added for welding
  const float kRB26s12TubeL2 = 47.21;                          // part of this tube outside barrel region
  const float kRB26s12TubeL = kRB26s12TubeL0 - kRB26s12TubeL2; // 392.115
  //
  // 184.905
  // 0.877
  // Add 1 cm on outer diameter for insulation
  //
  //
  // the section which is placed into the central barrel (ending at z = -505)
  TGeoPcon* shRB26s12Tube = new TGeoPcon(0., 360., 4);
  // Section 1: straight section
  shRB26s12Tube->DefineSection(0, 0.00, 5.84 / 2., 6.00 / 2.);
  shRB26s12Tube->DefineSection(1, 207.21, 5.84 / 2., 6.00 / 2.);
  // Section 2: 0.72 deg opening cone
  shRB26s12Tube->DefineSection(2, 207.21, 5.84 / 2., 6.14 / 2.);
  shRB26s12Tube->DefineSection(3, kRB26s12TubeL, 5.84 / 2 + 2.576, 6.14 / 2. + 2.576);

  // the section which is placed into the muon spectrometer (starting at z = -505)
  TGeoPcon* shRB26s12msTube = new TGeoPcon(0., 360., 3);
  // conical part
  shRB26s12msTube->DefineSection(0, 0.00, shRB26s12Tube->GetRmin(3), shRB26s12Tube->GetRmax(3));
  shRB26s12msTube->DefineSection(1, 452.30 - kRB26s12TubeL, 12.0 / 2., 12.3 / 2.);
  // straight part until compensator
  shRB26s12msTube->DefineSection(2, kRB26s12TubeL2, 12.0 / 2., 12.3 / 2.);

  TGeoVolume* voRB26s12Tube = new TGeoVolume("RB26s12Tube", shRB26s12Tube, kMedSteelHC);
  TGeoVolume* voRB26s12msTube = new TGeoVolume("RB26s12msTube", shRB26s12msTube, kMedSteelHC);
  // Add the insulation layer
  TGeoVolume* voRB26s12TubeIns = new TGeoVolume("RB26s12TubeIns", makeInsulationFromTemplate(shRB26s12Tube), kMedInsu);
  TGeoVolume* voRB26s12msTubeIns = new TGeoVolume("RB26s12msTubeIns", makeInsulationFromTemplate(shRB26s12msTube), kMedInsu);
  voRB26s12Tube->AddNode(voRB26s12TubeIns, 1, gGeoIdentity);
  voRB26s12msTube->AddNode(voRB26s12msTubeIns, 1, gGeoIdentity);

  TGeoVolume* voRB26s12TubeM = new TGeoVolume("RB26s12TubeM", makeMotherFromTemplate(shRB26s12Tube), kMedVacHC);
  voRB26s12TubeM->AddNode(voRB26s12Tube, 1, gGeoIdentity);
  TGeoVolume* voRB26s12msTubeM = new TGeoVolume("RB26s12msTubeM", makeMotherFromTemplate(shRB26s12msTube), kMedVacHC);
  voRB26s12msTubeM->AddNode(voRB26s12msTube, 1, gGeoIdentity);

  ///////////////////////////////////
  //    RB26/2   Axial Compensator //
  //    Drawing  LHCVC2a_0064      //
  ///////////////////////////////////
  const float kRB26s2CompL = 30.65;          // Length of the compensator
  const float kRB26s2BellowRo = 14.38 / 2.;  // Bellow outer radius        [Pos 1]
  const float kRB26s2BellowRi = 12.12 / 2.;  // Bellow inner radius        [Pos 1]
  const int kRB26s2NumberOfPlies = 14;       // Number of plies            [Pos 1]
  const float kRB26s2BellowUndL = 10.00;     // Length of undulated region [Pos 1]  [+10 mm installed including pretension ?]
  const float kRB26s2PlieThickness = 0.025;  // Plie thickness             [Pos 1]
  const float kRB26s2ConnectionPlieR = 0.21; // Connection plie radius     [Pos 1]
  //  Plie radius
  const float kRB26s2PlieR = (kRB26s2BellowUndL - 4. * kRB26s2ConnectionPlieR + 2. * kRB26s2PlieThickness +
                              (2. * kRB26s2NumberOfPlies - 2.) * kRB26s2PlieThickness) /
                             (4. * kRB26s2NumberOfPlies - 2.);
  const float kRB26s2CompTubeInnerR = 12.00 / 2.;    // Connection tubes inner radius     [Pos 2 + 3]
  const float kRB26s2CompTubeOuterR = 12.30 / 2.;    // Connection tubes outer radius     [Pos 2 + 3]
  const float kRB26s2WeldingTubeLeftL = 9.00 / 2.;   // Left connection tube half length  [Pos 2]
  const float kRB26s2WeldingTubeRightL = 11.65 / 2.; // Right connection tube half length [Pos 3]  [+ 0.15 cm for welding]
  const float kRB26s2RingOuterR = 18.10 / 2.;        // Ring inner radius                 [Pos 4]
  const float kRB26s2RingL = 0.40 / 2.;              // Ring half length                  [Pos 4]
  const float kRB26s2RingZ = 6.50;                   // Ring z-position                   [Pos 4]
  const float kRB26s2ProtOuterR = 18.20 / 2.;        // Protection tube outer radius      [Pos 5]
  const float kRB26s2ProtL = 15.00 / 2.;             // Protection tube half length       [Pos 5]
  const float kRB26s2ProtZ = 6.70;                   // Protection tube z-position        [Pos 5]

  // Mother volume
  //
  TGeoPcon* shRB26s2Compensator = new TGeoPcon(0., 360., 6);
  shRB26s2Compensator->DefineSection(0, 0.0, 0., kRB26s2CompTubeOuterR);
  shRB26s2Compensator->DefineSection(1, kRB26s2RingZ, 0., kRB26s2CompTubeOuterR);
  shRB26s2Compensator->DefineSection(2, kRB26s2RingZ, 0., kRB26s2ProtOuterR);
  shRB26s2Compensator->DefineSection(3, kRB26s2ProtZ + 2. * kRB26s2ProtL, 0., kRB26s2ProtOuterR);
  shRB26s2Compensator->DefineSection(4, kRB26s2ProtZ + 2. * kRB26s2ProtL, 0., kRB26s2CompTubeOuterR);
  shRB26s2Compensator->DefineSection(5, kRB26s2CompL, 0., kRB26s2CompTubeOuterR);
  TGeoVolume* voRB26s2Compensator = new TGeoVolume("RB26s2Compensator", shRB26s2Compensator, kMedVacHC);

  //
  // [Pos 1] Bellow
  //
  //
  TGeoVolume* voRB26s2Bellow =
    new TGeoVolume("RB26s2Bellow", new TGeoTube(kRB26s2BellowRi, kRB26s2BellowRo, kRB26s2BellowUndL / 2.), kMedVacHC);
  //
  //  Upper part of the undulation
  //
  TGeoTorus* shRB26s2PlieTorusU = new TGeoTorus(kRB26s2BellowRo - kRB26s2PlieR, kRB26s2PlieR - kRB26s2PlieThickness, kRB26s2PlieR);
  shRB26s2PlieTorusU->SetName("RB26s2TorusU");
  TGeoTube* shRB26s2PlieTubeU = new TGeoTube(kRB26s2BellowRo - kRB26s2PlieR, kRB26s2BellowRo, kRB26s2PlieR);
  shRB26s2PlieTubeU->SetName("RB26s2TubeU");
  TGeoCompositeShape* shRB26s2UpperPlie = new TGeoCompositeShape("RB26s2UpperPlie", "RB26s2TorusU*RB26s2TubeU");

  TGeoVolume* voRB26s2WiggleU = new TGeoVolume("RB26s2UpperPlie", shRB26s2UpperPlie, kMedSteelHC);
  //
  // Lower part of the undulation
  TGeoTorus* shRB26s2PlieTorusL = new TGeoTorus(kRB26s2BellowRi + kRB26s2PlieR, kRB26s2PlieR - kRB26s2PlieThickness, kRB26s2PlieR);
  shRB26s2PlieTorusL->SetName("RB26s2TorusL");
  TGeoTube* shRB26s2PlieTubeL = new TGeoTube(kRB26s2BellowRi, kRB26s2BellowRi + kRB26s2PlieR, kRB26s2PlieR);
  shRB26s2PlieTubeL->SetName("RB26s2TubeL");
  TGeoCompositeShape* shRB26s2LowerPlie = new TGeoCompositeShape("RB26s2LowerPlie", "RB26s2TorusL*RB26s2TubeL");

  TGeoVolume* voRB26s2WiggleL = new TGeoVolume("RB26s2LowerPlie", shRB26s2LowerPlie, kMedSteelHC);

  //
  // Connection between upper and lower part of undulation
  TGeoVolume* voRB26s2WiggleC1 = new TGeoVolume(
    "RB26s2PlieConn1",
    new TGeoTube(kRB26s2BellowRi + kRB26s2PlieR, kRB26s2BellowRo - kRB26s2PlieR, kRB26s2PlieThickness / 2.), kMedSteelHC);
  //
  // One wiggle
  TGeoVolumeAssembly* voRB26s2Wiggle = new TGeoVolumeAssembly("RB26s2Wiggle");
  z0 = -kRB26s2PlieThickness / 2.;
  voRB26s2Wiggle->AddNode(voRB26s2WiggleC1, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2PlieR - kRB26s2PlieThickness / 2.;
  voRB26s2Wiggle->AddNode(voRB26s2WiggleU, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2PlieR - kRB26s2PlieThickness / 2.;
  voRB26s2Wiggle->AddNode(voRB26s2WiggleC1, 2, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2PlieR - kRB26s2PlieThickness;
  voRB26s2Wiggle->AddNode(voRB26s2WiggleL, 1, new TGeoTranslation(0., 0., z0));
  // Positioning of the volumes
  z0 = -kRB26s2BellowUndL / 2. + kRB26s2ConnectionPlieR;
  voRB26s2Bellow->AddNode(voRB26s2WiggleL, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2ConnectionPlieR;
  zsh = 4. * kRB26s2PlieR - 2. * kRB26s2PlieThickness;
  for (int iw = 0; iw < kRB26s2NumberOfPlies; iw++) {
    float zpos = z0 + iw * zsh;
    voRB26s2Bellow->AddNode(voRB26s2Wiggle, iw + 1, new TGeoTranslation(0., 0., zpos - kRB26s2PlieThickness));
  }

  voRB26s2Compensator->AddNode(voRB26s2Bellow, 1, new TGeoTranslation(0., 0., 2. * kRB26s2WeldingTubeLeftL + kRB26s2BellowUndL / 2.));

  //
  // [Pos 2] Left Welding Tube
  //
  TGeoTube* shRB26s2CompLeftTube = new TGeoTube(kRB26s2CompTubeInnerR, kRB26s2CompTubeOuterR, kRB26s2WeldingTubeLeftL);
  TGeoVolume* voRB26s2CompLeftTube = new TGeoVolume("RB26s2CompLeftTube", shRB26s2CompLeftTube, kMedSteelHC);
  voRB26s2Compensator->AddNode(voRB26s2CompLeftTube, 1, new TGeoTranslation(0., 0., kRB26s2WeldingTubeLeftL));
  //
  // [Pos 3] Right Welding Tube
  //
  TGeoTube* shRB26s2CompRightTube =
    new TGeoTube(kRB26s2CompTubeInnerR, kRB26s2CompTubeOuterR, kRB26s2WeldingTubeRightL);
  TGeoVolume* voRB26s2CompRightTube = new TGeoVolume("RB26s2CompRightTube", shRB26s2CompRightTube, kMedSteelHC);
  voRB26s2Compensator->AddNode(voRB26s2CompRightTube, 1, new TGeoTranslation(0., 0., kRB26s2CompL - kRB26s2WeldingTubeRightL));
  //
  // [Pos 4] Ring
  //
  TGeoTube* shRB26s2CompRing = new TGeoTube(kRB26s2CompTubeOuterR, kRB26s2RingOuterR, kRB26s2RingL);
  TGeoVolume* voRB26s2CompRing = new TGeoVolume("RB26s2CompRing", shRB26s2CompRing, kMedSteelHC);
  voRB26s2Compensator->AddNode(voRB26s2CompRing, 1, new TGeoTranslation(0., 0., kRB26s2RingZ + kRB26s2RingL));

  //
  // [Pos 5] Outer Protecting Tube
  //
  TGeoTube* shRB26s2CompProtTube = new TGeoTube(kRB26s2RingOuterR, kRB26s2ProtOuterR, kRB26s2ProtL);
  TGeoVolume* voRB26s2CompProtTube = new TGeoVolume("RB26s2CompProtTube", shRB26s2CompProtTube, kMedSteelHC);
  voRB26s2Compensator->AddNode(voRB26s2CompProtTube, 1, new TGeoTranslation(0., 0., kRB26s2ProtZ + kRB26s2ProtL));

  ///////////////////////////////////
  //    Rotable Flange             //
  //    Drawing  LHCVFX_0016       //
  ///////////////////////////////////
  const float kRB26s1RFlangeTubeRi = 5.84 / 2.; // Tube inner radius
  const float kRB26s1RFlangeTubeRo = 6.00 / 2.; // Tube outer radius

  // Pos 1 Clamp Ring          LHCVFX__0015
  const float kRB26s1RFlangeCrL = 1.40;        // Lenth of the clamp ring
  const float kRB26s1RFlangeCrRi1 = 6.72 / 2.; // Ring inner radius section 1
  const float kRB26s1RFlangeCrRi2 = 6.06 / 2.; // Ring inner radius section 2
  const float kRB26s1RFlangeCrRo = 8.60 / 2.;  // Ring outer radius
  const float kRB26s1RFlangeCrD = 0.800;       // Width section 1

  TGeoPcon* shRB26s1RFlangeCr = new TGeoPcon(0., 360., 4);
  z0 = 0.;
  shRB26s1RFlangeCr->DefineSection(0, z0, kRB26s1RFlangeCrRi1, kRB26s1RFlangeCrRo);
  z0 += kRB26s1RFlangeCrD;
  shRB26s1RFlangeCr->DefineSection(1, z0, kRB26s1RFlangeCrRi1, kRB26s1RFlangeCrRo);
  shRB26s1RFlangeCr->DefineSection(2, z0, kRB26s1RFlangeCrRi2, kRB26s1RFlangeCrRo);
  z0 = kRB26s1RFlangeCrL;
  shRB26s1RFlangeCr->DefineSection(3, z0, kRB26s1RFlangeCrRi2, kRB26s1RFlangeCrRo);
  TGeoVolume* voRB26s1RFlangeCr = new TGeoVolume("RB26s1RFlangeCr", shRB26s1RFlangeCr, kMedSteelHC);

  // Pos 2 Insert              LHCVFX__0015
  const float kRB26s1RFlangeIsL = 4.88;      // Lenth of the insert
  const float kRB26s1RFlangeIsR = 6.70 / 2.; // Ring radius
  const float kRB26s1RFlangeIsD = 0.80;      // Ring Width

  TGeoPcon* shRB26s1RFlangeIs = new TGeoPcon(0., 360., 4);
  z0 = 0.;
  shRB26s1RFlangeIs->DefineSection(0, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeIsR);
  z0 += kRB26s1RFlangeIsD;
  shRB26s1RFlangeIs->DefineSection(1, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeIsR);
  shRB26s1RFlangeIs->DefineSection(2, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeTubeRo);
  z0 = kRB26s1RFlangeIsL;
  shRB26s1RFlangeIs->DefineSection(3, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeTubeRo);
  TGeoVolume* voRB26s1RFlangeIs = new TGeoVolume("RB26s1RFlangeIs", shRB26s1RFlangeIs, kMedSteelHC);
  // 4.88 + 3.7 = 8.58 (8.7 to avoid overlap)
  // Pos 3 Fixed Point Section LHCVC2A_0021
  const float kRB26s1RFlangeFpL = 5.88;      // Length of the fixed point section (0.08 cm added for welding)
  const float kRB26s1RFlangeFpZ = 3.82;      // Position of the ring
  const float kRB26s1RFlangeFpD = 0.59;      // Width of the ring
  const float kRB26s1RFlangeFpR = 7.00 / 2.; // Radius of the ring

  TGeoPcon* shRB26s1RFlangeFp = new TGeoPcon(0., 360., 6);
  z0 = 0.;
  shRB26s1RFlangeFp->DefineSection(0, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeTubeRo);
  z0 += kRB26s1RFlangeFpZ;
  shRB26s1RFlangeFp->DefineSection(1, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeTubeRo);
  shRB26s1RFlangeFp->DefineSection(2, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeFpR);
  z0 += kRB26s1RFlangeFpD;
  shRB26s1RFlangeFp->DefineSection(3, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeFpR);
  shRB26s1RFlangeFp->DefineSection(4, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeTubeRo);
  z0 = kRB26s1RFlangeFpL;
  shRB26s1RFlangeFp->DefineSection(5, z0, kRB26s1RFlangeTubeRi, kRB26s1RFlangeTubeRo);
  TGeoVolume* voRB26s1RFlangeFp = new TGeoVolume("RB26s1RFlangeFp", shRB26s1RFlangeFp, kMedSteelHC);

  // Put everything in a mother volume
  TGeoPcon* shRB26s1RFlange = new TGeoPcon(0., 360., 8);
  z0 = 0.;
  shRB26s1RFlange->DefineSection(0, z0, 0., kRB26s1RFlangeCrRo);
  z0 += kRB26s1RFlangeCrL;
  shRB26s1RFlange->DefineSection(1, z0, 0., kRB26s1RFlangeCrRo);
  shRB26s1RFlange->DefineSection(2, z0, 0., kRB26s1RFlangeTubeRo);
  z0 = kRB26s1RFlangeIsL + kRB26s1RFlangeFpZ;
  shRB26s1RFlange->DefineSection(3, z0, 0., kRB26s1RFlangeTubeRo);
  shRB26s1RFlange->DefineSection(4, z0, 0., kRB26s1RFlangeFpR);
  z0 += kRB26s1RFlangeFpD;
  shRB26s1RFlange->DefineSection(5, z0, 0., kRB26s1RFlangeFpR);
  shRB26s1RFlange->DefineSection(6, z0, 0., kRB26s1RFlangeTubeRo);
  z0 = kRB26s1RFlangeIsL + kRB26s1RFlangeFpL;
  shRB26s1RFlange->DefineSection(7, z0, 0., kRB26s1RFlangeTubeRo);
  TGeoVolume* voRB26s1RFlange = new TGeoVolume("RB26s1RFlange", shRB26s1RFlange, kMedVacHC);

  voRB26s1RFlange->AddNode(voRB26s1RFlangeIs, 1, gGeoIdentity);
  voRB26s1RFlange->AddNode(voRB26s1RFlangeCr, 1, gGeoIdentity);
  voRB26s1RFlange->AddNode(voRB26s1RFlangeFp, 1, new TGeoTranslation(0., 0., kRB26s1RFlangeIsL));

  ///////////////////////////////////
  //    Fixed Flange               //
  //    Drawing  LHCVFX_0006       //
  ///////////////////////////////////
  const float kRB26s2FFlangeL = 2.13;         // Length of the flange
  const float kRB26s2FFlangeD1 = 0.97;        // Length of section 1
  const float kRB26s2FFlangeD2 = 0.29;        // Length of section 2
  const float kRB26s2FFlangeD3 = 0.87;        // Length of section 3
  const float kRB26s2FFlangeRo = 17.15 / 2.;  // Flange outer radius
  const float kRB26s2FFlangeRi1 = 12.30 / 2.; // Flange inner radius section 1
  const float kRB26s2FFlangeRi2 = 12.00 / 2.; // Flange inner radius section 2
  const float kRB26s2FFlangeRi3 = 12.30 / 2.; // Flange inner radius section 3
  z0 = 0;
  TGeoPcon* shRB26s2FFlange = new TGeoPcon(0., 360., 6);
  z0 = 0.;
  shRB26s2FFlange->DefineSection(0, z0, kRB26s2FFlangeRi1, kRB26s2FFlangeRo);
  z0 += kRB26s2FFlangeD1;
  shRB26s2FFlange->DefineSection(1, z0, kRB26s2FFlangeRi1, kRB26s2FFlangeRo);
  shRB26s2FFlange->DefineSection(2, z0, kRB26s2FFlangeRi2, kRB26s2FFlangeRo);
  z0 += kRB26s2FFlangeD2;
  shRB26s2FFlange->DefineSection(3, z0, kRB26s2FFlangeRi2, kRB26s2FFlangeRo);
  shRB26s2FFlange->DefineSection(4, z0, kRB26s2FFlangeRi3, kRB26s2FFlangeRo);
  z0 += kRB26s2FFlangeD3;
  shRB26s2FFlange->DefineSection(5, z0, kRB26s2FFlangeRi3, kRB26s2FFlangeRo);
  TGeoVolume* voRB26s2FFlange = new TGeoVolume("RB26s2FFlange", shRB26s2FFlange, kMedSteelHC);

  TGeoVolume* voRB26s2FFlangeM = new TGeoVolume("RB26s2FFlangeM", makeMotherFromTemplate(shRB26s2FFlange, 2, 5), kMedVacHC);
  voRB26s2FFlangeM->AddNode(voRB26s2FFlange, 1, gGeoIdentity);

  ////////////////////////////////////////
  //                                    //
  //    RB26/3                          //
  //    Drawing LHCV2a_0048             //
  //    Drawing LHCV2a_0002             //
  ////////////////////////////////////////
  //
  //    Pos 1 Vacuum Tubes      LHCVC2A__0003
  //    Pos 2 Fixed Point       LHCVFX___0005
  //    Pos 3 Split Flange      LHCVFX___0007
  //    Pos 4 Fixed Flange      LHCVFX___0004
  //    Pos 5 Axial Compensator LHCVC2A__0065
  //
  //
  //
  //
  ///////////////////////////////////
  //    Vacuum Tube                //
  //    Drawing  LHCVC2A_0003      //
  ///////////////////////////////////
  const float kRB26s3TubeL = 629.35 + 0.3; // 0.3 cm added for welding
  const float kRB26s3TubeR1 = 12. / 2.;
  const float kRB26s3TubeR2 = kRB26s3TubeR1 + 215.8 * TMath::Tan(0.829 / 180. * TMath::Pi());

  TGeoPcon* shRB26s3Tube = new TGeoPcon(0., 360., 7);
  // Section 1: straight section
  shRB26s3Tube->DefineSection(0, 0.00, kRB26s3TubeR1, kRB26s3TubeR1 + 0.15);
  shRB26s3Tube->DefineSection(1, 2.00, kRB26s3TubeR1, kRB26s3TubeR1 + 0.15);
  // Section 2: 0.829 deg opening cone
  shRB26s3Tube->DefineSection(2, 2.00, kRB26s3TubeR1, kRB26s3TubeR1 + 0.20);

  shRB26s3Tube->DefineSection(3, 217.80, kRB26s3TubeR2, kRB26s3TubeR2 + 0.20);
  shRB26s3Tube->DefineSection(4, 217.80, kRB26s3TubeR2, kRB26s3TubeR2 + 0.30);

  shRB26s3Tube->DefineSection(5, 622.20, 30.00 / 2., 30.60 / 2.);
  shRB26s3Tube->DefineSection(6, kRB26s3TubeL, 30.00 / 2., 30.60 / 2.);

  TGeoVolume* voRB26s3Tube = new TGeoVolume("RB26s3Tube", shRB26s3Tube, kMedSteelHC);
  //    Add the insulation layer
  TGeoVolume* voRB26s3TubeIns = new TGeoVolume("RB26s3TubeIns", makeInsulationFromTemplate(shRB26s3Tube), kMedInsu);
  voRB26s3Tube->AddNode(voRB26s3TubeIns, 1, gGeoIdentity);

  TGeoVolume* voRB26s3TubeM = new TGeoVolume("RB26s3TubeM", makeMotherFromTemplate(shRB26s3Tube), kMedVacHC);
  voRB26s3TubeM->AddNode(voRB26s3Tube, 1, gGeoIdentity);

  ///////////////////////////////////
  //    Fixed Point                //
  //    Drawing  LHCVFX_0005       //
  ///////////////////////////////////
  const float kRB26s3FixedPointL = 16.37;        // Length of the fixed point section (0.3 cm added for welding)
  const float kRB26s3FixedPointZ = 9.72;         // Position of the ring (0.15 cm added for welding)
  const float kRB26s3FixedPointD = 0.595;        // Width of the ring
  const float kRB26s3FixedPointR = 13.30 / 2.;   // Radius of the ring
  const float kRB26s3FixedPointRi = 12.00 / 2.;  // Inner radius of the tube
  const float kRB26s3FixedPointRo1 = 12.30 / 2.; // Outer radius of the tube (in)
  const float kRB26s3FixedPointRo2 = 12.40 / 2.; // Outer radius of the tube (out)
  const float kRB26s3FixedPointDs = 1.5;         // Width of straight section behind ring
  const float kRB26s3FixedPointDc = 3.15;        // Width of conical  section behind ring (0.15 cm added for welding)

  TGeoPcon* shRB26s3FixedPoint = new TGeoPcon(0., 360., 8);
  z0 = 0.;
  shRB26s3FixedPoint->DefineSection(0, z0, kRB26s3FixedPointRi, kRB26s3FixedPointRo1);
  z0 += kRB26s3FixedPointZ;
  shRB26s3FixedPoint->DefineSection(1, z0, kRB26s3FixedPointRi, kRB26s3FixedPointRo1);
  shRB26s3FixedPoint->DefineSection(2, z0, kRB26s3FixedPointRi, kRB26s3FixedPointR);
  z0 += kRB26s3FixedPointD;
  shRB26s3FixedPoint->DefineSection(3, z0, kRB26s3FixedPointRi, kRB26s3FixedPointR);
  shRB26s3FixedPoint->DefineSection(4, z0, kRB26s3FixedPointRi, kRB26s3FixedPointRo1);
  z0 += kRB26s3FixedPointDs;
  shRB26s3FixedPoint->DefineSection(5, z0, kRB26s3FixedPointRi, kRB26s3FixedPointRo1);
  z0 += kRB26s3FixedPointDc;
  shRB26s3FixedPoint->DefineSection(6, z0, kRB26s3FixedPointRi, kRB26s3FixedPointRo2);
  z0 = kRB26s3FixedPointL;
  shRB26s3FixedPoint->DefineSection(7, z0, kRB26s3FixedPointRi, kRB26s3FixedPointRo2);
  TGeoVolume* voRB26s3FixedPoint = new TGeoVolume("RB26s3FixedPoint", shRB26s3FixedPoint, kMedSteelHC);

  TGeoVolume* voRB26s3FixedPointM = new TGeoVolume("RB26s3FixedPointM", makeMotherFromTemplate(shRB26s3FixedPoint), kMedVacHC);
  voRB26s3FixedPointM->AddNode(voRB26s3FixedPoint, 1, gGeoIdentity);

  ///////////////////////////////////
  //    Split Flange               //
  //    Drawing  LHCVFX_0005       //
  ///////////////////////////////////
  const float kRB26s3SFlangeL = 2.13;         // Length of the flange
  const float kRB26s3SFlangeD1 = 0.57;        // Length of section 1
  const float kRB26s3SFlangeD2 = 0.36;        // Length of section 2
  const float kRB26s3SFlangeD3 = 0.50 + 0.70; // Length of section 3
  const float kRB26s3SFlangeRo = 17.15 / 2.;  // Flange outer radius
  const float kRB26s3SFlangeRi1 = 12.30 / 2.; // Flange inner radius section 1
  const float kRB26s3SFlangeRi2 = 12.00 / 2.; // Flange inner radius section 2
  const float kRB26s3SFlangeRi3 = 12.30 / 2.; // Flange inner radius section 3
  z0 = 0;
  TGeoPcon* shRB26s3SFlange = new TGeoPcon(0., 360., 6);
  z0 = 0.;
  shRB26s3SFlange->DefineSection(0, z0, kRB26s3SFlangeRi1, kRB26s3SFlangeRo);
  z0 += kRB26s3SFlangeD1;
  shRB26s3SFlange->DefineSection(1, z0, kRB26s3SFlangeRi1, kRB26s3SFlangeRo);
  shRB26s3SFlange->DefineSection(2, z0, kRB26s3SFlangeRi2, kRB26s3SFlangeRo);
  z0 += kRB26s3SFlangeD2;
  shRB26s3SFlange->DefineSection(3, z0, kRB26s3SFlangeRi2, kRB26s3SFlangeRo);
  shRB26s3SFlange->DefineSection(4, z0, kRB26s3SFlangeRi3, kRB26s3SFlangeRo);
  z0 += kRB26s3SFlangeD3;
  shRB26s3SFlange->DefineSection(5, z0, kRB26s3SFlangeRi3, kRB26s3SFlangeRo);
  TGeoVolume* voRB26s3SFlange = new TGeoVolume("RB26s3SFlange", shRB26s3SFlange, kMedSteelHC);

  TGeoVolume* voRB26s3SFlangeM = new TGeoVolume("RB26s3SFlangeM", makeMotherFromTemplate(shRB26s3SFlange, 0, 3), kMedVacHC);
  voRB26s3SFlangeM->AddNode(voRB26s3SFlange, 1, gGeoIdentity);

  ///////////////////////////////////
  //    RB26/3   Fixed Flange      //
  //    Drawing  LHCVFX___0004     //
  ///////////////////////////////////
  const float kRB26s3FFlangeL = 2.99;         // Length of the flange
  const float kRB26s3FFlangeD1 = 1.72;        // Length of section 1
  const float kRB26s3FFlangeD2 = 0.30;        // Length of section 2
  const float kRB26s3FFlangeD3 = 0.97;        // Length of section 3
  const float kRB26s3FFlangeRo = 36.20 / 2.;  // Flange outer radius
  const float kRB26s3FFlangeRi1 = 30.60 / 2.; // Flange inner radius section 1
  const float kRB26s3FFlangeRi2 = 30.00 / 2.; // Flange inner radius section 2
  const float kRB26s3FFlangeRi3 = 30.60 / 2.; // Flange inner radius section 3
  z0 = 0;
  TGeoPcon* shRB26s3FFlange = new TGeoPcon(0., 360., 6);
  z0 = 0.;
  shRB26s3FFlange->DefineSection(0, z0, kRB26s3FFlangeRi1, kRB26s3FFlangeRo);
  z0 += kRB26s3FFlangeD1;
  shRB26s3FFlange->DefineSection(1, z0, kRB26s3FFlangeRi1, kRB26s3FFlangeRo);
  shRB26s3FFlange->DefineSection(2, z0, kRB26s3FFlangeRi2, kRB26s3FFlangeRo);
  z0 += kRB26s3FFlangeD2;
  shRB26s3FFlange->DefineSection(3, z0, kRB26s3FFlangeRi2, kRB26s3FFlangeRo);
  shRB26s3FFlange->DefineSection(4, z0, kRB26s3FFlangeRi3, kRB26s3FFlangeRo);
  z0 += kRB26s3FFlangeD3;
  shRB26s3FFlange->DefineSection(5, z0, kRB26s3FFlangeRi3, kRB26s3FFlangeRo);
  TGeoVolume* voRB26s3FFlange = new TGeoVolume("RB26s3FFlange", shRB26s3FFlange, kMedSteelHC);

  TGeoVolume* voRB26s3FFlangeM = new TGeoVolume("RB26s3FFlangeM", makeMotherFromTemplate(shRB26s3FFlange, 2, 5), kMedVacHC);
  voRB26s3FFlangeM->AddNode(voRB26s3FFlange, 1, gGeoIdentity);

  ///////////////////////////////////
  //    RB26/3   Axial Compensator //
  //    Drawing  LHCVC2a_0065      //
  ///////////////////////////////////
  const float kRB26s3CompL = 42.3;           // Length of the compensator (0.3 cm added for welding)
  const float kRB26s3BellowRo = 34.00 / 2.;  // Bellow outer radius        [Pos 1]
  const float kRB26s3BellowRi = 30.10 / 2.;  // Bellow inner radius        [Pos 1]
  const int kRB26s3NumberOfPlies = 13;       // Number of plies            [Pos 1]
  const float kRB26s3BellowUndL = 17.70;     // Length of undulated region [Pos 1]
  const float kRB26s3PlieThickness = 0.06;   // Plie thickness             [Pos 1]
  const float kRB26s3ConnectionPlieR = 0.21; // Connection plie radius     [Pos 1]
  //  Plie radius
  const float kRB26s3PlieR = (kRB26s3BellowUndL - 4. * kRB26s3ConnectionPlieR + 2. * kRB26s3PlieThickness +
                              (2. * kRB26s3NumberOfPlies - 2.) * kRB26s3PlieThickness) /
                             (4. * kRB26s3NumberOfPlies - 2.);

  //
  // The welding tubes have 3 sections with different radii and 2 transition regions.
  // Section 1: connection to the outside
  // Section 2: commection to the bellow
  // Section 3: between 1 and 2
  const float kRB26s3CompTubeInnerR1 = 30.0 / 2.; // Outer Connection tubes inner radius     [Pos 4 + 3]
  const float kRB26s3CompTubeOuterR1 = 30.6 / 2.; // Outer Connection tubes outer radius     [Pos 4 + 3]
  const float kRB26s3CompTubeInnerR2 = 29.4 / 2.; // Connection tubes inner radius           [Pos 4 + 3]
  const float kRB26s3CompTubeOuterR2 = 30.0 / 2.; // Connection tubes outer radius           [Pos 4 + 3]
  const float kRB26s3CompTubeInnerR3 = 30.6 / 2.; // Connection tubes inner radius at bellow [Pos 4 + 3]
  const float kRB26s3CompTubeOuterR3 = 32.2 / 2.; // Connection tubes outer radius at bellow [Pos 4 + 3]

  const float kRB26s3WeldingTubeLeftL1 = 2.0;   // Left connection tube length             [Pos 4]
  const float kRB26s3WeldingTubeLeftL2 = 3.4;   // Left connection tube length             [Pos 4]
  const float kRB26s3WeldingTubeLeftL = 7.0;    // Left connection tube total length       [Pos 4]
  const float kRB26s3WeldingTubeRightL1 = 2.3;  // Right connection tube length            [Pos 3] (0.3 cm added for welding)
  const float kRB26s3WeldingTubeRightL2 = 13.4; // Right connection tube length            [Pos 3]

  const float kRB26s3WeldingTubeT1 = 0.6; // Length of first r-transition            [Pos 4 + 3]
  const float kRB26s3WeldingTubeT2 = 1.0; // Length of 2nd   r-transition            [Pos 4 + 3]

  const float kRB26s3RingOuterR = 36.1 / 2.; // Ring inner radius                       [Pos 4]
  const float kRB26s3RingL = 0.8 / 2.;       // Ring half length                        [Pos 4]
  const float kRB26s3RingZ = 3.7;            // Ring z-position                         [Pos 4]
  const float kRB26s3ProtOuterR = 36.2 / 2.; // Protection tube outer radius            [Pos 2]
  const float kRB26s3ProtL = 27.0 / 2.;      // Protection tube half length             [Pos 2]
  const float kRB26s3ProtZ = 4.0;            // Protection tube z-position              [Pos 2]

  // Mother volume
  //
  TGeoPcon* shRB26s3Compensator = new TGeoPcon(0., 360., 6);
  shRB26s3Compensator->DefineSection(0, 0.0, 0., kRB26s3CompTubeOuterR1);
  shRB26s3Compensator->DefineSection(1, kRB26s3RingZ, 0., kRB26s3CompTubeOuterR1);
  shRB26s3Compensator->DefineSection(2, kRB26s3RingZ, 0., kRB26s3ProtOuterR);
  shRB26s3Compensator->DefineSection(3, kRB26s3ProtZ + 2. * kRB26s3ProtL, 0., kRB26s3ProtOuterR);
  shRB26s3Compensator->DefineSection(4, kRB26s3ProtZ + 2. * kRB26s3ProtL, 0., kRB26s3CompTubeOuterR1);
  shRB26s3Compensator->DefineSection(5, kRB26s3CompL, 0., kRB26s3CompTubeOuterR1);
  TGeoVolume* voRB26s3Compensator = new TGeoVolume("RB26s3Compensator", shRB26s3Compensator, kMedVacHC);

  //
  // [Pos 1] Bellow
  //
  //

  //
  //  Upper part of the undulation
  //
  TGeoTorus* shRB26s3PlieTorusU = new TGeoTorus(kRB26s3BellowRo - kRB26s3PlieR, kRB26s3PlieR - kRB26s3PlieThickness, kRB26s3PlieR);
  shRB26s3PlieTorusU->SetName("RB26s3TorusU");
  TGeoTube* shRB26s3PlieTubeU = new TGeoTube(kRB26s3BellowRo - kRB26s3PlieR, kRB26s3BellowRo, kRB26s3PlieR);
  shRB26s3PlieTubeU->SetName("RB26s3TubeU");
  TGeoCompositeShape* shRB26s3UpperPlie = new TGeoCompositeShape("RB26s3UpperPlie", "RB26s3TorusU*RB26s3TubeU");

  TGeoVolume* voRB26s3WiggleU = new TGeoVolume("RB26s3UpperPlie", shRB26s3UpperPlie, kMedSteelHC);
  //
  // Lower part of the undulation
  TGeoTorus* shRB26s3PlieTorusL = new TGeoTorus(kRB26s3BellowRi + kRB26s3PlieR, kRB26s3PlieR - kRB26s3PlieThickness, kRB26s3PlieR);
  shRB26s3PlieTorusL->SetName("RB26s3TorusL");
  TGeoTube* shRB26s3PlieTubeL = new TGeoTube(kRB26s3BellowRi, kRB26s3BellowRi + kRB26s3PlieR, kRB26s3PlieR);
  shRB26s3PlieTubeL->SetName("RB26s3TubeL");
  TGeoCompositeShape* shRB26s3LowerPlie = new TGeoCompositeShape("RB26s3LowerPlie", "RB26s3TorusL*RB26s3TubeL");

  TGeoVolume* voRB26s3WiggleL = new TGeoVolume("RB26s3LowerPlie", shRB26s3LowerPlie, kMedSteelHC);

  //
  // Connection between upper and lower part of undulation
  TGeoVolume* voRB26s3WiggleC1 = new TGeoVolume(
    "RB26s3PlieConn1",
    new TGeoTube(kRB26s3BellowRi + kRB26s3PlieR, kRB26s3BellowRo - kRB26s3PlieR, kRB26s3PlieThickness / 2.), kMedSteelHC);
  //
  // One wiggle
  TGeoVolumeAssembly* voRB26s3Wiggle = new TGeoVolumeAssembly("RB26s3Wiggle");
  z0 = -kRB26s3PlieThickness / 2.;
  voRB26s3Wiggle->AddNode(voRB26s3WiggleC1, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3PlieR - kRB26s3PlieThickness / 2.;
  voRB26s3Wiggle->AddNode(voRB26s3WiggleU, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3PlieR - kRB26s3PlieThickness / 2.;
  voRB26s3Wiggle->AddNode(voRB26s3WiggleC1, 2, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3PlieR - kRB26s3PlieThickness;
  voRB26s3Wiggle->AddNode(voRB26s3WiggleL, 1, new TGeoTranslation(0., 0., z0));
  voRB26s3Wiggle->GetShape()->ComputeBBox(); // enforce recomputing of BBox

  //
  // The bellow itself
  float zBellowTot = kRB26s3NumberOfPlies * (static_cast<TGeoBBox*>(voRB26s3Wiggle->GetShape()))->GetDZ();
  TGeoVolume* voRB26s3Bellow = new TGeoVolume("RB26s3Bellow", new TGeoTube(kRB26s3BellowRi, kRB26s3BellowRo, zBellowTot), kMedVacHC);

  // Positioning of the volumes
  z0 = -kRB26s2BellowUndL / 2. + kRB26s2ConnectionPlieR;
  voRB26s2Bellow->AddNode(voRB26s2WiggleL, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2ConnectionPlieR;
  zsh = 4. * kRB26s2PlieR - 2. * kRB26s2PlieThickness;
  for (int iw = 0; iw < kRB26s2NumberOfPlies; iw++) {
    float zpos = z0 + iw * zsh;
    voRB26s2Bellow->AddNode(voRB26s2Wiggle, iw + 1, new TGeoTranslation(0., 0., zpos - kRB26s2PlieThickness));
  }

  voRB26s3Compensator->AddNode(voRB26s3Bellow, 1, new TGeoTranslation(0., 0., kRB26s3WeldingTubeLeftL + zBellowTot));

  //
  // [Pos 2] Outer Protecting Tube
  //
  TGeoTube* shRB26s3CompProtTube = new TGeoTube(kRB26s3RingOuterR, kRB26s3ProtOuterR, kRB26s3ProtL);
  TGeoVolume* voRB26s3CompProtTube = new TGeoVolume("RB26s3CompProtTube", shRB26s3CompProtTube, kMedSteelHC);
  voRB26s3Compensator->AddNode(voRB26s3CompProtTube, 1, new TGeoTranslation(0., 0., kRB26s3ProtZ + kRB26s3ProtL));

  //
  // [Pos 3] Right Welding Tube
  //
  TGeoPcon* shRB26s3CompRightTube = new TGeoPcon(0., 360., 5);
  z0 = 0.;
  shRB26s3CompRightTube->DefineSection(0, z0, kRB26s3CompTubeInnerR3, kRB26s3CompTubeOuterR3);
  z0 += kRB26s3WeldingTubeT2;
  shRB26s3CompRightTube->DefineSection(1, z0, kRB26s3CompTubeInnerR2, kRB26s3CompTubeOuterR2);
  z0 += kRB26s3WeldingTubeRightL2;
  shRB26s3CompRightTube->DefineSection(2, z0, kRB26s3CompTubeInnerR2, kRB26s3CompTubeOuterR2);
  z0 += kRB26s3WeldingTubeT1;
  shRB26s3CompRightTube->DefineSection(3, z0, kRB26s3CompTubeInnerR1, kRB26s3CompTubeOuterR1);
  z0 += kRB26s3WeldingTubeRightL1;
  shRB26s3CompRightTube->DefineSection(4, z0, kRB26s3CompTubeInnerR1, kRB26s3CompTubeOuterR1);

  TGeoVolume* voRB26s3CompRightTube = new TGeoVolume("RB26s3CompRightTube", shRB26s3CompRightTube, kMedSteelHC);
  voRB26s3Compensator->AddNode(voRB26s3CompRightTube, 1, new TGeoTranslation(0., 0., kRB26s3CompL - z0));

  //
  // [Pos 4] Left Welding Tube
  //
  TGeoPcon* shRB26s3CompLeftTube = new TGeoPcon(0., 360., 5);
  z0 = 0.;
  shRB26s3CompLeftTube->DefineSection(0, z0, kRB26s3CompTubeInnerR1, kRB26s3CompTubeOuterR1);
  z0 += kRB26s3WeldingTubeLeftL1;
  shRB26s3CompLeftTube->DefineSection(1, z0, kRB26s3CompTubeInnerR1, kRB26s3CompTubeOuterR1);
  z0 += kRB26s3WeldingTubeT1;
  shRB26s3CompLeftTube->DefineSection(2, z0, kRB26s3CompTubeInnerR2, kRB26s3CompTubeOuterR2);
  z0 += kRB26s3WeldingTubeLeftL2;
  shRB26s3CompLeftTube->DefineSection(3, z0, kRB26s3CompTubeInnerR2, kRB26s3CompTubeOuterR2);
  z0 += kRB26s3WeldingTubeT2;
  shRB26s3CompLeftTube->DefineSection(4, z0, kRB26s3CompTubeInnerR3, kRB26s3CompTubeOuterR3);

  TGeoVolume* voRB26s3CompLeftTube = new TGeoVolume("RB26s3CompLeftTube", shRB26s3CompLeftTube, kMedSteelHC);
  voRB26s3Compensator->AddNode(voRB26s3CompLeftTube, 1, gGeoIdentity);
  //
  // [Pos 5] Ring
  //
  TGeoTube* shRB26s3CompRing = new TGeoTube(kRB26s3CompTubeOuterR2, kRB26s3RingOuterR, kRB26s3RingL);
  TGeoVolume* voRB26s3CompRing = new TGeoVolume("RB26s3CompRing", shRB26s3CompRing, kMedSteelHC);
  voRB26s3Compensator->AddNode(voRB26s3CompRing, 1, new TGeoTranslation(0., 0., kRB26s3RingZ + kRB26s3RingL));

  ///////////////////////////////////////////
  //                                       //
  //    RB26/4-5                           //
  //    Drawing LHCV2a_0012 [as installed] //
  ////////////////////////////////////////////
  //    Pos1 Vacuum Tubes        LHCVC2A__0014
  //    Pos2 Compensator         LHCVC2A__0066
  //    Pos3 Fixed Point Section LHCVC2A__0016
  //    Pos4 Split Flange        LHCVFX___0005
  //    Pos5 RotableFlange       LHCVFX___0009
  ////////////////////////////////////////////

  ///////////////////////////////////
  //    RB26/4-5 Vacuum Tubes      //
  //    Drawing  LHCVC2a_0014      //
  ///////////////////////////////////
  const float kRB26s45TubeL = 593.12 + 0.3; // 0.3 cm added for welding

  TGeoPcon* shRB26s45Tube = new TGeoPcon(0., 360., 11);
  // Section 1: straight section
  shRB26s45Tube->DefineSection(0, 0.00, 30.00 / 2., 30.60 / 2.);
  shRB26s45Tube->DefineSection(1, 1.20, 30.00 / 2., 30.60 / 2.);
  shRB26s45Tube->DefineSection(2, 1.20, 30.00 / 2., 30.80 / 2.);
  shRB26s45Tube->DefineSection(3, 25.10, 30.00 / 2., 30.80 / 2.);
  // Section 2: 0.932 deg opening cone
  shRB26s45Tube->DefineSection(4, 486.10, 45.00 / 2., 45.80 / 2.);
  // Section 3: straight section 4 mm
  shRB26s45Tube->DefineSection(5, 512.10, 45.00 / 2., 45.80 / 2.);
  // Section 4: straight section 3 mm
  shRB26s45Tube->DefineSection(6, 512.10, 45.00 / 2., 45.60 / 2.);
  shRB26s45Tube->DefineSection(7, 527.70, 45.00 / 2., 45.60 / 2.);
  // Section 4: closing cone
  shRB26s45Tube->DefineSection(8, 591.30, 10.00 / 2., 10.60 / 2.);
  shRB26s45Tube->DefineSection(9, 591.89, 10.00 / 2., 10.30 / 2.);

  shRB26s45Tube->DefineSection(10, kRB26s45TubeL, 10.00 / 2., 10.30 / 2.);
  TGeoVolume* voRB26s45Tube = new TGeoVolume("RB26s45Tube", shRB26s45Tube, kMedSteelHC);

  TGeoVolume* voRB26s45TubeM = new TGeoVolume("RB26s45TubeM", makeMotherFromTemplate(shRB26s45Tube), kMedVacHC);
  voRB26s45TubeM->AddNode(voRB26s45Tube, 1, gGeoIdentity);

  ///////////////////////////////////
  //    RB26/5   Axial Compensator //
  //    Drawing  LHCVC2a_0066      //
  ///////////////////////////////////
  const float kRB26s5CompL = 27.60;           // Length of the compensator (0.30 cm added for welding)
  const float kRB26s5BellowRo = 12.48 / 2.;   // Bellow outer radius        [Pos 1]
  const float kRB26s5BellowRi = 10.32 / 2.;   // Bellow inner radius        [Pos 1]
  const int kRB26s5NumberOfPlies = 15;        // Number of plies            [Pos 1]
  const float kRB26s5BellowUndL = 10.50;      // Length of undulated region [Pos 1]
  const float kRB26s5PlieThickness = 0.025;   // Plie thickness             [Pos 1]
  const float kRB26s5ConnectionPlieR = 0.21;  // Connection plie radius     [Pos 1]
  const float kRB26s5ConnectionR = 11.2 / 2.; // Bellow connection radius   [Pos 1]
  //  Plie radius
  const float kRB26s5PlieR = (kRB26s5BellowUndL - 4. * kRB26s5ConnectionPlieR + 2. * kRB26s5PlieThickness +
                              (2. * kRB26s5NumberOfPlies - 2.) * kRB26s5PlieThickness) /
                             (4. * kRB26s5NumberOfPlies - 2.);
  const float kRB26s5CompTubeInnerR = 10.00 / 2.;    // Connection tubes inner radius     [Pos 2 + 3]
  const float kRB26s5CompTubeOuterR = 10.30 / 2.;    // Connection tubes outer radius     [Pos 2 + 3]
  const float kRB26s5WeldingTubeLeftL = 3.70 / 2.;   // Left connection tube half length  [Pos 2]
  const float kRB26s5WeldingTubeRightL = 13.40 / 2.; // Right connection tube half length [Pos 3]   (0.3 cm added for welding)
  const float kRB26s5RingInnerR = 11.2 / 2.;         // Ring inner radius                 [Pos 4]
  const float kRB26s5RingOuterR = 16.0 / 2.;         // Ring inner radius                 [Pos 4]
  const float kRB26s5RingL = 0.4 / 2.;               // Ring half length                  [Pos 4]
  const float kRB26s5RingZ = 14.97;                  // Ring z-position                   [Pos 4]
  const float kRB26s5ProtOuterR = 16.2 / 2.;         // Protection tube outer radius      [Pos 5]
  const float kRB26s5ProtL = 13.0 / 2.;              // Protection tube half length       [Pos 5]
  const float kRB26s5ProtZ = 2.17;                   // Protection tube z-position        [Pos 5]
  const float kRB26s5DetailZR = 11.3 / 2.;           // Detail Z max radius

  // Mother volume
  //
  TGeoPcon* shRB26s5Compensator = new TGeoPcon(0., 360., 8);
  shRB26s5Compensator->DefineSection(0, 0.0, 0., kRB26s5CompTubeOuterR);
  shRB26s5Compensator->DefineSection(1, kRB26s5ProtZ, 0., kRB26s5CompTubeOuterR);
  shRB26s5Compensator->DefineSection(2, kRB26s5ProtZ, 0., kRB26s5ProtOuterR);
  shRB26s5Compensator->DefineSection(3, kRB26s5ProtZ + 2. * kRB26s5ProtL + 2. * kRB26s5RingL, 0., kRB26s5ProtOuterR);
  shRB26s5Compensator->DefineSection(4, kRB26s5ProtZ + 2. * kRB26s5ProtL + 2. * kRB26s5RingL, 0., kRB26s5DetailZR);
  shRB26s5Compensator->DefineSection(5, kRB26s5CompL - 8., 0., kRB26s5DetailZR);
  shRB26s5Compensator->DefineSection(6, kRB26s5CompL - 8., 0., kRB26s5CompTubeOuterR);
  shRB26s5Compensator->DefineSection(7, kRB26s5CompL, 0., kRB26s5CompTubeOuterR);
  TGeoVolume* voRB26s5Compensator = new TGeoVolume("RB26s5Compensator", shRB26s5Compensator, kMedVacHC);

  //
  // [Pos 1] Bellow
  //
  //
  TGeoVolume* voRB26s5Bellow =
    new TGeoVolume("RB26s5Bellow", new TGeoTube(kRB26s5BellowRi, kRB26s5BellowRo, kRB26s5BellowUndL / 2.), kMedVacHC);
  //
  //  Upper part of the undulation
  //
  TGeoTorus* shRB26s5PlieTorusU = new TGeoTorus(kRB26s5BellowRo - kRB26s5PlieR, kRB26s5PlieR - kRB26s5PlieThickness, kRB26s5PlieR);
  shRB26s5PlieTorusU->SetName("RB26s5TorusU");
  TGeoTube* shRB26s5PlieTubeU = new TGeoTube(kRB26s5BellowRo - kRB26s5PlieR, kRB26s5BellowRo, kRB26s5PlieR);
  shRB26s5PlieTubeU->SetName("RB26s5TubeU");
  TGeoCompositeShape* shRB26s5UpperPlie = new TGeoCompositeShape("RB26s5UpperPlie", "RB26s5TorusU*RB26s5TubeU");

  TGeoVolume* voRB26s5WiggleU = new TGeoVolume("RB26s5UpperPlie", shRB26s5UpperPlie, kMedSteelHC);
  //
  // Lower part of the undulation
  TGeoTorus* shRB26s5PlieTorusL = new TGeoTorus(kRB26s5BellowRi + kRB26s5PlieR, kRB26s5PlieR - kRB26s5PlieThickness, kRB26s5PlieR);
  shRB26s5PlieTorusL->SetName("RB26s5TorusL");
  TGeoTube* shRB26s5PlieTubeL = new TGeoTube(kRB26s5BellowRi, kRB26s5BellowRi + kRB26s5PlieR, kRB26s5PlieR);
  shRB26s5PlieTubeL->SetName("RB26s5TubeL");
  TGeoCompositeShape* shRB26s5LowerPlie = new TGeoCompositeShape("RB26s5LowerPlie", "RB26s5TorusL*RB26s5TubeL");

  TGeoVolume* voRB26s5WiggleL = new TGeoVolume("RB26s5LowerPlie", shRB26s5LowerPlie, kMedSteelHC);

  //
  // Connection between upper and lower part of undulation
  TGeoVolume* voRB26s5WiggleC1 = new TGeoVolume("RB26s5PlieConn1",
                                                new TGeoTube(kRB26s5BellowRi + kRB26s5PlieR, kRB26s5BellowRo - kRB26s5PlieR, kRB26s5PlieThickness / 2.), kMedSteelHC);
  //
  // One wiggle
  TGeoVolumeAssembly* voRB26s5Wiggle = new TGeoVolumeAssembly("RB26s5Wiggle");
  z0 = -kRB26s5PlieThickness / 2.;
  voRB26s5Wiggle->AddNode(voRB26s5WiggleC1, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s5PlieR - kRB26s5PlieThickness / 2.;
  voRB26s5Wiggle->AddNode(voRB26s5WiggleU, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s5PlieR - kRB26s5PlieThickness / 2.;
  voRB26s5Wiggle->AddNode(voRB26s5WiggleC1, 2, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s5PlieR - kRB26s5PlieThickness;
  voRB26s5Wiggle->AddNode(voRB26s5WiggleL, 1, new TGeoTranslation(0., 0., z0));
  // Positioning of the volumes
  z0 = -kRB26s5BellowUndL / 2. + kRB26s5ConnectionPlieR;
  voRB26s5Bellow->AddNode(voRB26s5WiggleL, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s5ConnectionPlieR;
  zsh = 4. * kRB26s5PlieR - 2. * kRB26s5PlieThickness;
  for (int iw = 0; iw < kRB26s5NumberOfPlies; iw++) {
    float zpos = z0 + iw * zsh;
    voRB26s5Bellow->AddNode(voRB26s5Wiggle, iw + 1, new TGeoTranslation(0., 0., zpos - kRB26s5PlieThickness));
  }

  voRB26s5Compensator->AddNode(voRB26s5Bellow, 1, new TGeoTranslation(0., 0., 2. * kRB26s5WeldingTubeLeftL + kRB26s5BellowUndL / 2.));

  //
  // [Pos 2] Left Welding Tube
  //
  TGeoPcon* shRB26s5CompLeftTube = new TGeoPcon(0., 360., 3);
  z0 = 0;
  shRB26s5CompLeftTube->DefineSection(0, z0, kRB26s5CompTubeInnerR, kRB26s5CompTubeOuterR);
  z0 += 2 * kRB26s5WeldingTubeLeftL - (kRB26s5ConnectionR - kRB26s5CompTubeOuterR);
  shRB26s5CompLeftTube->DefineSection(1, z0, kRB26s5CompTubeInnerR, kRB26s5CompTubeOuterR);
  z0 += (kRB26s5ConnectionR - kRB26s5CompTubeOuterR);
  shRB26s5CompLeftTube->DefineSection(2, z0, kRB26s5ConnectionR - 0.15, kRB26s5ConnectionR);
  TGeoVolume* voRB26s5CompLeftTube = new TGeoVolume("RB26s5CompLeftTube", shRB26s5CompLeftTube, kMedSteelHC);
  voRB26s5Compensator->AddNode(voRB26s5CompLeftTube, 1, gGeoIdentity);
  //
  // [Pos 3] Right Welding Tube
  //
  TGeoPcon* shRB26s5CompRightTube = new TGeoPcon(0., 360., 11);
  // Detail Z
  shRB26s5CompRightTube->DefineSection(0, 0., kRB26s5CompTubeInnerR + 0.22, 11.2 / 2.);
  shRB26s5CompRightTube->DefineSection(1, 0.05, kRB26s5CompTubeInnerR + 0.18, 11.2 / 2.);
  shRB26s5CompRightTube->DefineSection(2, 0.22, kRB26s5CompTubeInnerR, 11.2 / 2. - 0.22);
  shRB26s5CompRightTube->DefineSection(3, 0.44, kRB26s5CompTubeInnerR, 11.2 / 2.);
  shRB26s5CompRightTube->DefineSection(4, 1.70, kRB26s5CompTubeInnerR, 11.2 / 2.);
  shRB26s5CompRightTube->DefineSection(5, 2.10, kRB26s5CompTubeInnerR, kRB26s5CompTubeOuterR);
  shRB26s5CompRightTube->DefineSection(6, 2.80, kRB26s5CompTubeInnerR, kRB26s5CompTubeOuterR);
  shRB26s5CompRightTube->DefineSection(7, 2.80, kRB26s5CompTubeInnerR, 11.3 / 2.);
  shRB26s5CompRightTube->DefineSection(8, 3.40, kRB26s5CompTubeInnerR, 11.3 / 2.);
  // Normal pipe
  shRB26s5CompRightTube->DefineSection(9, 3.50, kRB26s5CompTubeInnerR, kRB26s5CompTubeOuterR);
  shRB26s5CompRightTube->DefineSection(10, 2. * kRB26s5WeldingTubeRightL, kRB26s5CompTubeInnerR, kRB26s5CompTubeOuterR);

  TGeoVolume* voRB26s5CompRightTube = new TGeoVolume("RB26s5CompRightTube", shRB26s5CompRightTube, kMedSteelHC);
  voRB26s5Compensator->AddNode(voRB26s5CompRightTube, 1, new TGeoTranslation(0., 0., kRB26s5CompL - 2. * kRB26s5WeldingTubeRightL));
  //
  // [Pos 4] Ring
  //
  TGeoTube* shRB26s5CompRing = new TGeoTube(kRB26s5RingInnerR, kRB26s5RingOuterR, kRB26s5RingL);
  TGeoVolume* voRB26s5CompRing = new TGeoVolume("RB26s5CompRing", shRB26s5CompRing, kMedSteelHC);
  voRB26s5Compensator->AddNode(voRB26s5CompRing, 1, new TGeoTranslation(0., 0., kRB26s5RingZ + kRB26s5RingL));

  //
  // [Pos 5] Outer Protecting Tube
  //
  TGeoTube* shRB26s5CompProtTube = new TGeoTube(kRB26s5RingOuterR, kRB26s5ProtOuterR, kRB26s5ProtL);
  TGeoVolume* voRB26s5CompProtTube = new TGeoVolume("RB26s5CompProtTube", shRB26s5CompProtTube, kMedSteelHC);
  voRB26s5Compensator->AddNode(voRB26s5CompProtTube, 1, new TGeoTranslation(0., 0., kRB26s5ProtZ + kRB26s5ProtL));

  ///////////////////////////////////////
  //    RB26/4   Fixed Point Section   //
  //    Drawing  LHCVC2a_0016          //
  ///////////////////////////////////////
  const float kRB26s4TubeRi = 30.30 / 2.;      // Tube inner radius  (0.3 cm added for welding)
  const float kRB26s4TubeRo = 30.60 / 2.;      // Tube outer radius
  const float kRB26s4FixedPointL = 12.63;      // Length of the fixed point section
  const float kRB26s4FixedPointZ = 10.53;      // Position of the ring (0.15 added for welding)
  const float kRB26s4FixedPointD = 0.595;      // Width of the ring
  const float kRB26s4FixedPointR = 31.60 / 2.; // Radius of the ring

  TGeoPcon* shRB26s4FixedPoint = new TGeoPcon(0., 360., 6);
  z0 = 0.;
  shRB26s4FixedPoint->DefineSection(0, z0, kRB26s4TubeRi, kRB26s4TubeRo);
  z0 += kRB26s4FixedPointZ;
  shRB26s4FixedPoint->DefineSection(1, z0, kRB26s4TubeRi, kRB26s4TubeRo);
  shRB26s4FixedPoint->DefineSection(2, z0, kRB26s4TubeRi, kRB26s4FixedPointR);
  z0 += kRB26s4FixedPointD;
  shRB26s4FixedPoint->DefineSection(3, z0, kRB26s4TubeRi, kRB26s4FixedPointR);
  shRB26s4FixedPoint->DefineSection(4, z0, kRB26s4TubeRi, kRB26s4TubeRo);
  z0 = kRB26s4FixedPointL;
  shRB26s4FixedPoint->DefineSection(5, z0, kRB26s4TubeRi, kRB26s4TubeRo);
  TGeoVolume* voRB26s4FixedPoint = new TGeoVolume("RB26s4FixedPoint", shRB26s4FixedPoint, kMedSteelHC);

  TGeoVolume* voRB26s4FixedPointM = new TGeoVolume("RB26s4FixedPointM", makeMotherFromTemplate(shRB26s4FixedPoint), kMedVacHC);
  voRB26s4FixedPointM->AddNode(voRB26s4FixedPoint, 1, gGeoIdentity);

  ///////////////////////////////////////
  //    RB26/4   Split Flange          //
  //    Drawing  LHCVFX__0005          //
  ///////////////////////////////////////
  const float kRB26s4SFlangeL = 2.99;         // Length of the flange
  const float kRB26s4SFlangeD1 = 0.85;        // Length of section 1
  const float kRB26s4SFlangeD2 = 0.36;        // Length of section 2
  const float kRB26s4SFlangeD3 = 0.73 + 1.05; // Length of section 3
  const float kRB26s4SFlangeRo = 36.20 / 2.;  // Flange outer radius
  const float kRB26s4SFlangeRi1 = 30.60 / 2.; // Flange inner radius section 1
  const float kRB26s4SFlangeRi2 = 30.00 / 2.; // Flange inner radius section 2
  const float kRB26s4SFlangeRi3 = 30.60 / 2.; // Flange inner radius section 3
  z0 = 0;
  TGeoPcon* shRB26s4SFlange = new TGeoPcon(0., 360., 6);
  z0 = 0.;
  shRB26s4SFlange->DefineSection(0, z0, kRB26s4SFlangeRi1, kRB26s4SFlangeRo);
  z0 += kRB26s4SFlangeD1;
  shRB26s4SFlange->DefineSection(1, z0, kRB26s4SFlangeRi1, kRB26s4SFlangeRo);
  shRB26s4SFlange->DefineSection(2, z0, kRB26s4SFlangeRi2, kRB26s4SFlangeRo);
  z0 += kRB26s4SFlangeD2;
  shRB26s4SFlange->DefineSection(3, z0, kRB26s4SFlangeRi2, kRB26s4SFlangeRo);
  shRB26s4SFlange->DefineSection(4, z0, kRB26s4SFlangeRi3, kRB26s4SFlangeRo);
  z0 += kRB26s4SFlangeD3;
  shRB26s4SFlange->DefineSection(5, z0, kRB26s4SFlangeRi3, kRB26s4SFlangeRo);
  TGeoVolume* voRB26s4SFlange = new TGeoVolume("RB26s4SFlange", shRB26s4SFlange, kMedSteelHC);

  TGeoVolume* voRB26s4SFlangeM = new TGeoVolume("RB26s4SFlangeM", makeMotherFromTemplate(shRB26s4SFlange, 0, 3), kMedVacHC);
  voRB26s4SFlangeM->AddNode(voRB26s4SFlange, 1, gGeoIdentity);

  ///////////////////////////////////////
  //    RB26/5   Rotable Flange        //
  //    Drawing  LHCVFX__0009          //
  ///////////////////////////////////////
  const float kRB26s5RFlangeL = 1.86;         // Length of the flange
  const float kRB26s5RFlangeD1 = 0.61;        // Length of section 1
  const float kRB26s5RFlangeD2 = 0.15;        // Length of section 2
  const float kRB26s5RFlangeD3 = 0.60;        // Length of section 3
  const float kRB26s5RFlangeD4 = 0.50;        // Length of section 4
  const float kRB26s5RFlangeRo = 15.20 / 2.;  // Flange outer radius
  const float kRB26s5RFlangeRi1 = 10.30 / 2.; // Flange inner radius section 1
  const float kRB26s5RFlangeRi2 = 10.00 / 2.; // Flange inner radius section 2
  const float kRB26s5RFlangeRi3 = 10.30 / 2.; // Flange inner radius section 3
  const float kRB26s5RFlangeRi4 = 10.50 / 2.; // Flange inner radius section 4

  z0 = 0;
  TGeoPcon* shRB26s5RFlange = new TGeoPcon(0., 360., 8);
  z0 = 0.;
  shRB26s5RFlange->DefineSection(0, z0, kRB26s5RFlangeRi4, kRB26s5RFlangeRo);
  z0 += kRB26s5RFlangeD4;
  shRB26s5RFlange->DefineSection(1, z0, kRB26s5RFlangeRi4, kRB26s5RFlangeRo);
  shRB26s5RFlange->DefineSection(2, z0, kRB26s5RFlangeRi3, kRB26s5RFlangeRo);
  z0 += kRB26s5RFlangeD3;
  shRB26s5RFlange->DefineSection(3, z0, kRB26s5RFlangeRi3, kRB26s5RFlangeRo);
  shRB26s5RFlange->DefineSection(4, z0, kRB26s5RFlangeRi2, kRB26s5RFlangeRo);
  z0 += kRB26s5RFlangeD2;
  shRB26s5RFlange->DefineSection(5, z0, kRB26s5RFlangeRi2, kRB26s5RFlangeRo);
  shRB26s5RFlange->DefineSection(6, z0, kRB26s5RFlangeRi1, kRB26s5RFlangeRo);
  z0 += kRB26s5RFlangeD1;
  shRB26s5RFlange->DefineSection(7, z0, kRB26s5RFlangeRi1, kRB26s5RFlangeRo);
  TGeoVolume* voRB26s5RFlange = new TGeoVolume("RB26s5RFlange", shRB26s5RFlange, kMedSteelHC);

  TGeoVolume* voRB26s5RFlangeM = new TGeoVolume("RB26s5RFlangeM", makeMotherFromTemplate(shRB26s5RFlange, 4, 7), kMedVacHC);
  voRB26s5RFlangeM->AddNode(voRB26s5RFlange, 1, gGeoIdentity);

  //
  // Assemble RB26/1-2
  //
  TGeoVolumeAssembly* asRB26s12 = new TGeoVolumeAssembly("RB26s12");
  z0 = 0.;
  //  asRB26s12->AddNode(voRB26s1RFlange, 1, gGeoIdentity);
  barrel->AddNode(voRB26s1RFlange, 1, new TGeoCombiTrans(0., 30., -82, rot180));
  z0 += kRB26s1RFlangeIsL + kRB26s1RFlangeFpL;
  barrel->AddNode(voRB26s12TubeM, 1, new TGeoCombiTrans(0., 30., -82. - z0, rot180));
  z0 += kRB26s12TubeL;
  asRB26s12->AddNode(voRB26s12msTubeM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s12TubeL2;
  asRB26s12->AddNode(voRB26s2Compensator, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2CompL;
  z0 -= kRB26s2FFlangeD1;
  asRB26s12->AddNode(voRB26s2FFlangeM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s2FFlangeL;
  const float kRB26s12L = z0;

  //
  // Assemble RB26/3
  //
  TGeoVolumeAssembly* asRB26s3 = new TGeoVolumeAssembly("RB26s3");
  z0 = 0.;
  asRB26s3->AddNode(voRB26s3SFlangeM, 1, gGeoIdentity);
  z0 += kRB26s3SFlangeL;
  z0 -= kRB26s3SFlangeD3;
  asRB26s3->AddNode(voRB26s3FixedPointM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3FixedPointL;
  asRB26s3->AddNode(voRB26s3TubeM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3TubeL;
  asRB26s3->AddNode(voRB26s3Compensator, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3CompL;
  z0 -= kRB26s3FFlangeD1;
  asRB26s3->AddNode(voRB26s3FFlangeM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3FFlangeL;
  const float kRB26s3L = z0;

  //
  // Assemble RB26/4-5
  //
  TGeoVolumeAssembly* asRB26s45 = new TGeoVolumeAssembly("RB26s45");
  z0 = 0.;
  asRB26s45->AddNode(voRB26s4SFlangeM, 1, gGeoIdentity);
  z0 += kRB26s4SFlangeL;
  z0 -= kRB26s4SFlangeD3;
  asRB26s45->AddNode(voRB26s4FixedPointM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s4FixedPointL;
  asRB26s45->AddNode(voRB26s45TubeM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s45TubeL;
  asRB26s45->AddNode(voRB26s5Compensator, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s5CompL;
  z0 -= kRB26s5RFlangeD3;
  z0 -= kRB26s5RFlangeD4;
  asRB26s45->AddNode(voRB26s5RFlangeM, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s5RFlangeL;
  const float kRB26s45L = z0;

  //
  // Assemble RB26
  //
  TGeoVolumeAssembly* asRB26Pipe = new TGeoVolumeAssembly("RB26Pipe");
  z0 = 0.;
  asRB26Pipe->AddNode(asRB26s12, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s12L;
  asRB26Pipe->AddNode(asRB26s3, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s3L;
  asRB26Pipe->AddNode(asRB26s45, 1, new TGeoTranslation(0., 0., z0));
  z0 += kRB26s45L;
  top->AddNode(asRB26Pipe, 1, new TGeoCombiTrans(0., 0., -82., rot180));
}

void PipeRun4::createMaterials()
{
  //
  // Define materials for beam pipe
  //
  int isxfld = 2.;
  float sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // Steel (Inox)
  float asteel[4] = {55.847, 51.9961, 58.6934, 28.0855};
  float zsteel[4] = {26., 24., 28., 14.};
  float wsteel[4] = {.715, .18, .1, .005};
  // AlBe - alloy
  float aAlBe[2] = {26.98, 9.01}; // al=2.702 be=1.8477
  float zAlBe[2] = {13.00, 4.00};
  float wAlBe[2] = {0.4, 0.6};
  // Polyamid
  float aPA[4] = {16., 14., 12., 1.};
  float zPA[4] = {8., 7., 6., 1.};
  float wPA[4] = {1., 1., 6., 11.};
  // Polyimide film
  float aPI[4] = {16., 14., 12., 1.};
  float zPI[4] = {8., 7., 6., 1.};
  float wPI[4] = {5., 2., 22., 10.};
  // Rohacell
  float aRohacell[4] = {16., 14., 12., 1.};
  float zRohacell[4] = {8., 7., 6., 1.};
  float wRohacell[4] = {2., 1., 9., 13.};
  // Air
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;
  float dAir1 = 1.20479E-11;
  // Insulation powder
  //                    Si         O       Ti     Al
  float ains[4] = {28.0855, 15.9994, 47.867, 26.982};
  float zins[4] = {14., 8., 22., 13.};
  float wins[4] = {0.3019, 0.4887, 0.1914, 0.018};
  //
  //
  // Anticorodal
  //
  // Al Si7 Mg 0.6
  //
  float aaco[3] = {26.982, 28.0855, 24.035};
  float zaco[3] = {13., 14., 12.};
  float waco[3] = {0.924, 0.07, 0.006};
  // Kapton
  //
  float aKapton[4] = {1.00794, 12.0107, 14.010, 15.9994};
  float zKapton[4] = {1., 6., 7., 8.};
  float wKapton[4] = {0.026362, 0.69113, 0.07327, 0.209235};
  float dKapton = 1.42;
  // NEG coating
  //                  Ti     V      Zr
  float aNEG[4] = {47.87, 50.94, 91.24};
  float zNEG[4] = {22.00, 23.00, 40.00};
  float wNEG[4] = {1. / 3., 1. / 3., 1. / 3.};
  float dNEG = 5.6; // ?

  //---------------------------------
  // Aluminium AA 5083 for MFT: Al Manganese(Mn) Magnesium(Mg) Chrome(Cr)
  float aALU5083[4] = {26.982, 54.938, 24.305, 51.996}; // Mg pas meme a que la ligne Anticorodal!
  float zALU5083[4] = {13., 25., 12., 24.};
  float wALU5083[4] = {0.947, 0.007, 0.044, 0.0015};
  // Aluminium AA 2219 for MFT: Al Cu Mn Ti V Zr
  float aALU2219[6] = {26.982, 63.546, 54.938, 47.867, 50.941, 91.224};
  float zALU2219[6] = {13., 29., 25., 22., 23., 40.};
  float wALU2219[6] = {0.93, 0.063, 0.003, 0.0006, 0.001, 0.0018};
  // Aluminium AA 7075 for beam pipe support (wings): Al Zn Mg Cu
  float aALU7075[4] = {26.982, 65.38, 24.305, 63.546};
  float zALU7075[4] = {13., 30., 12., 29.};
  float wALU7075[4] = {0.902, 0.06, 0.024, 0.014};
  //---------------------------------

  // ****************
  //     Defines tracking media parameters.
  //
  float epsil = .1;     // Tracking precision,
  float stemax = -0.01; // Maximum displacement for multiple scat
  float tmaxfd = -20.;  // Maximum angle due to field deflection
  float deemax = -.3;   // Maximum fractional energy loss, DLS
  float stmin = -.8;
  // ***************
  //

  auto& matmgr = o2::base::MaterialManager::Instance();

  //    Beryllium
  matmgr.Material("PIPE", 5, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7);
  matmgr.Medium("PIPE", 5, "BE", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Copper
  matmgr.Material("PIPE", 10, "COPPER", 63.55, 29, 8.96, 1.43, 85.6 / 8.96);
  matmgr.Material("PIPE", 30, "COPPER_NF", 63.55, 29, 8.96, 1.43, 85.6 / 8.96);
  matmgr.Material("PIPE", 50, "COPPER_HC", 63.55, 29, 8.96, 1.43, 85.6 / 8.96);
  matmgr.Material("PIPE", 70, "COPPER_NFHC", 63.55, 29, 8.96, 1.43, 85.6 / 8.96);

  matmgr.Medium("PIPE", 10, "CU", 10, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 30, "CU_NF", 30, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 50, "CU_HC", 50, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 70, "CU_NFHC", 70, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Air
  matmgr.Mixture("PIPE", 15, "AIR$      ", aAir, zAir, dAir, 4, wAir);
  matmgr.Mixture("PIPE", 35, "AIR_HIGH$ ", aAir, zAir, dAir, 4, wAir);
  matmgr.Mixture("PIPE", 55, "AIR_NF ", aAir, zAir, dAir, 4, wAir);
  matmgr.Medium("PIPE", 15, "AIR", 15, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 35, "AIR_HIGH", 35, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 55, "AIR_NF", 55, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Insulation
  matmgr.Mixture("PIPE", 14, "INSULATION0$", ains, zins, 0.41, 4, wins);
  matmgr.Medium("PIPE", 14, "INS_C0", 14, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //
  //    Vacuum
  matmgr.Mixture("PIPE", 16, "VACUUM$ ", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("PIPE", 36, "VACUUM$_NF", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("PIPE", 56, "VACUUM$_HC ", aAir, zAir, dAir1, 4, wAir);
  matmgr.Mixture("PIPE", 76, "VACUUM$_NFHC", aAir, zAir, dAir1, 4, wAir);

  matmgr.Medium("PIPE", 16, "VACUUM", 16, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 36, "VACUUM_NF", 36, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 56, "VACUUM_HC", 56, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 76, "VACUUM_NFHC", 76, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //
  //    Steel
  matmgr.Mixture("PIPE", 19, "STAINLESS STEEL$", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("PIPE", 39, "STAINLESS STEEL$_NF", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("PIPE", 59, "STAINLESS STEEL$_HC", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("PIPE", 79, "STAINLESS STEEL$_NFHC", asteel, zsteel, 7.88, 4, wsteel);

  matmgr.Medium("PIPE", 19, "INOX", 19, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 39, "INOX_NF", 39, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 59, "INOX_HC", 59, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 79, "INOX_NFHC", 79, 0, 0, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //----------------- for the MFT ----------------------
  matmgr.Mixture("PIPE", 63, "ALUMINIUM5083$", aALU5083, zALU5083, 2.66, 4, wALU5083); // from aubertduval.fr
  matmgr.Mixture("PIPE", 64, "ALUMINIUM2219$", aALU2219, zALU2219, 2.84, 6, wALU2219); // from aubertduval.fr
  matmgr.Medium("PIPE", 63, "AA5083", 63, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("PIPE", 64, "AA2219", 64, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //----------------------------------------------------
  matmgr.Mixture("PIPE", 65, "PI$", aPI, zPI, 1.42, -4, wPI);
  matmgr.Medium("PIPE", 65, "POLYIMIDE", 65, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //---------------------------------
  //     Carbon Fiber M55J
  matmgr.Material("PIPE", 66, "M55J6K$", 12.0107, 6, 1.92, 999, 999);
  matmgr.Medium("PIPE", 66, "M55J6K", 66, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // Rohacell
  matmgr.Mixture("PIPE", 67, "Rohacell$", aRohacell, zRohacell, 0.03, -4, wRohacell);
  matmgr.Medium("PIPE", 67, "ROHACELL", 67, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // Titanium
  matmgr.Material("PIPE", 22, "Titanium$", 47.867, 22, 4.54, 3.560, 27.80);
  matmgr.Medium("PIPE", 22, "TITANIUM", 22, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // Alu 7075 (ZICRAL)
  matmgr.Mixture("PIPE", 68, "ALUMINIUM7075$", aALU7075, zALU7075, 2.810, -4, wALU7075);
  matmgr.Medium("PIPE", 68, "AA7075", 68, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // Al-Be alloy
  matmgr.Mixture("PIPE", 11, "AlBe$", aAlBe, zAlBe, 2.07, 2, wAlBe);
  matmgr.Medium("PIPE", 11, "AlBe", 11, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

TGeoPcon* PipeRun4::makeMotherFromTemplate(const TGeoPcon* shape, int imin, int imax, float r0, int nz)
{
  //
  //  Create a mother shape from a template setting some min radii to 0
  //
  int nz0 = shape->GetNz();
  // if nz > -1 the number of planes is given by nz
  if (nz != -1) {
    nz0 = nz;
  }
  TGeoPcon* mother = new TGeoPcon(0., 360., nz0);

  if (imin == -1 || imax == -1) {
    imin = 0;
    imax = shape->GetNz();
  } else if (imax >= nz0) {
    imax = nz0 - 1;
    printf("Warning: imax reset to nz-1 %5d %5d %5d %5d\n", imin, imax, nz, nz0);
  }

  // construct the sections dynamically since duplications have to be avoided
  std::vector<double> pconparams;
  pconparams.reserve(nz0);
  pconparams.push_back(0.);
  pconparams.push_back(360);
  pconparams.push_back(nz0);
  int zplanecounter = 0;

  auto addSection = [&pconparams, &zplanecounter](double z, double rmin, double rmax) {
    pconparams.push_back(z);
    pconparams.push_back(rmin);
    pconparams.push_back(rmax);
    zplanecounter++;
  };

  double zlast, rminlast, rmaxlast;
  for (int i = 0; i < shape->GetNz(); i++) {
    double rmin = shape->GetRmin(i);
    if ((i >= imin) && (i <= imax)) {
      rmin = r0;
    }
    double rmax = shape->GetRmax(i);
    double z = shape->GetZ(i);
    if (i == 0 || (z != zlast || rmin != rminlast || rmax != rmaxlast)) {
      addSection(z, rmin, rmax);
    }
    zlast = z;
    rminlast = rmin;
    rmaxlast = rmax;
  }
  // correct dimension (unless the user chose the number of sections)
  if (nz == -1) {
    pconparams[2] = zplanecounter;
    // reinit polycon from parameters
    mother->SetDimensions(pconparams.data());
  } else {
    for (int i = 0; i < zplanecounter; i++) {
      mother->DefineSection(i, pconparams[3 + 3 * i], pconparams[4 + 3 * i], pconparams[5 + 3 * i]);
    }
  }

  return mother;
}

TGeoPcon* PipeRun4::makeInsulationFromTemplate(TGeoPcon* shape)
{
  //
  //  Create an beam pipe insulation layer shape from a template
  //
  int nz = shape->GetNz();
  TGeoPcon* insu = new TGeoPcon(0., 360., nz);

  for (int i = 0; i < nz; i++) {
    double z = shape->GetZ(i);
    double rmin = shape->GetRmin(i);
    double rmax = shape->GetRmax(i);
    rmax += 0.5;
    shape->DefineSection(i, z, rmin, rmax);
    rmin = rmax - 0.5;
    insu->DefineSection(i, z, rmin, rmax);
  }
  return insu;
}

TGeoVolume* PipeRun4::makeBellow(const char* ext, int nc, float rMin, float rMax, float dU, float rPlie,
                                 float dPlie)
{
  // nc     Number of convolution
  // rMin   Inner radius of the bellow
  // rMax   Outer radius of the bellow
  // dU     Undulation length
  // rPlie  Plie radius
  // dPlie  Plie thickness
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* kMedVac = matmgr.getTGeoMedium("PIPE_VACUUM");
  const TGeoMedium* kMedSteel = matmgr.getTGeoMedium("PIPE_INOX");
  //
  //  Upper part of the undulation
  //
  std::string name, nameA, nameB;
  TGeoTorus* shPlieTorusU = new TGeoTorus(rMax - rPlie, rPlie - dPlie, rPlie);
  nameA = fmt::format("{:s}TorusU", ext);
  shPlieTorusU->SetName(nameA.c_str());
  TGeoTube* shPlieTubeU = new TGeoTube(rMax - rPlie, rMax, rPlie);
  nameB = fmt::format("{:s}TubeU", ext);
  shPlieTubeU->SetName(nameB.c_str());
  name = fmt::format("{:s}UpperPlie", ext);
  TGeoCompositeShape* shUpperPlie = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}*{:s}", nameA, nameB).c_str());

  TGeoVolume* voWiggleU = new TGeoVolume(name.c_str(), shUpperPlie, kMedSteel);
  //
  // Lower part of the undulation
  TGeoTorus* shPlieTorusL = new TGeoTorus(rMin + rPlie, rPlie - dPlie, rPlie);
  nameA = fmt::format("{:s}TorusL", ext);
  shPlieTorusL->SetName(nameA.c_str());
  TGeoTube* shPlieTubeL = new TGeoTube(rMin, rMin + rPlie, rPlie);
  nameB = fmt::format("{:s}TubeL", ext);
  shPlieTubeL->SetName(nameB.c_str());
  name = fmt::format("{:s}LowerPlie", ext);
  TGeoCompositeShape* shLowerPlie = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}*{:s}", nameA, nameB).c_str());

  TGeoVolume* voWiggleL = new TGeoVolume(name.c_str(), shLowerPlie, kMedSteel);

  //
  // Connection between upper and lower part of undulation
  TGeoVolume* voWiggleC1 = new TGeoVolume(fmt::format("{:s}PlieConn1", ext).c_str(), new TGeoTube(rMin + rPlie, rMax - rPlie, dPlie / 2.), kMedSteel);
  //
  // One wiggle
  float dz = rPlie - dPlie / 2.;
  float z0 = -dPlie / 2.;
  TGeoVolumeAssembly* asWiggle = new TGeoVolumeAssembly(fmt::format("{:s}Wiggle", ext).c_str());
  asWiggle->AddNode(voWiggleC1, 1, new TGeoTranslation(0., 0., z0));
  z0 += dz;
  asWiggle->AddNode(voWiggleU, 1, new TGeoTranslation(0., 0., z0));
  z0 += dz;
  asWiggle->AddNode(voWiggleC1, 2, new TGeoTranslation(0., 0., z0));
  z0 += dz;
  asWiggle->AddNode(voWiggleL, 1, new TGeoTranslation(0., 0., z0));
  asWiggle->GetShape()->ComputeBBox(); // enforce recomputing of BBox
  //
  float zBellowTot = nc * (static_cast<TGeoBBox*>(asWiggle->GetShape()))->GetDZ();
  TGeoVolume* voBellow = new TGeoVolume(fmt::format("{:s}BellowUS", ext).c_str(), new TGeoTube(rMin, rMax, zBellowTot), kMedVac);
  // Positioning of the volumes
  z0 = -dU / 2. + rPlie;
  voBellow->AddNode(voWiggleL, 2, new TGeoTranslation(0., 0., z0));
  z0 += rPlie;
  float zsh = 4. * rPlie - 2. * dPlie;
  for (int iw = 0; iw < nc; iw++) {
    float zpos = z0 + iw * zsh;
    voBellow->AddNode(asWiggle, iw + 1, new TGeoTranslation(0., 0., zpos - dPlie));
  }
  return voBellow;
}

TGeoVolume* PipeRun4::makeBellowCside(const char* ext, int nc, float rMin, float rMax, float rPlie, float dPlie)
{
  // nc     Number of convolution
  // rMin   Inner radius of the bellow
  // rMax   Outer radius of the bellow
  // dU     Undulation length
  // rPlie  Plie radius
  // dPlie  Plie thickness
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* kMedVac = matmgr.getTGeoMedium("PIPE_VACUUM");
  const TGeoMedium* kMedAlu5083 = matmgr.getTGeoMedium("PIPE_AA5083"); // fm

  float dU = nc * (4. * rPlie - 2. * dPlie);

  std::string name, nameA, nameB;
  name = fmt::format("{:s}BellowUS", ext);
  //  TGeoVolume* voBellow = new TGeoVolume(name, new TGeoTube(rMin, rMax, dU/2.), kMedVac);
  TGeoVolumeAssembly* voBellow = new TGeoVolumeAssembly(name.c_str());
  //
  //  Upper part of the undulation
  //

  TGeoTorus* shPlieTorusU = new TGeoTorus(rMax - rPlie, rPlie - dPlie, rPlie);
  nameA = fmt::format("{:s}TorusU", ext);
  shPlieTorusU->SetName(nameA.c_str());
  TGeoTube* shPlieTubeU = new TGeoTube(rMax - rPlie, rMax, rPlie);
  nameB = fmt::format("{:s}TubeU", ext);
  shPlieTubeU->SetName(nameB.c_str());
  name = fmt::format("{:s}UpperPlie", ext);
  TGeoCompositeShape* shUpperPlie = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}*{:s}", nameA, nameB).c_str());

  TGeoVolume* voWiggleU = new TGeoVolume(name.c_str(), shUpperPlie, kMedAlu5083);
  voWiggleU->SetLineColor(kOrange); // fm

  // First Lower part of the ondulation
  TGeoTorus* shPlieTorusL = new TGeoTorus(rMin + rPlie, rPlie - dPlie, rPlie);
  nameA = fmt::format("{:s}TorusL", ext);
  shPlieTorusL->SetName(nameA.c_str());
  TGeoTranslation* t1 = new TGeoTranslation("t1", 0, 0, -rPlie / 2.);
  t1->RegisterYourself();

  TGeoTube* shPlieTubeL = new TGeoTube(rMin, rMin + rPlie, rPlie / 2.);
  nameB = fmt::format("{:s}TubeL", ext);
  shPlieTubeL->SetName(nameB.c_str());
  name = fmt::format("{:s}LowerPlie", ext);
  TGeoCompositeShape* shLowerPlie1 = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}*{:s}:t1", nameA, nameB).c_str());

  TGeoVolume* voWiggleL1 = new TGeoVolume(name.c_str(), shLowerPlie1, kMedAlu5083);
  voWiggleL1->SetLineColor(kOrange); // fm

  // Second Lower part of the undulation
  TGeoTranslation* t2 = new TGeoTranslation("t2", 0, 0, rPlie / 2.);
  t2->RegisterYourself();

  TGeoCompositeShape* shLowerPlie2 = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}*{:s}:t2", nameA, nameB).c_str());

  TGeoVolume* voWiggleL2 = new TGeoVolume(name.c_str(), shLowerPlie2, kMedAlu5083);
  voWiggleL2->SetLineColor(kOrange); // fm

  // Connection between upper and lower part of undulation
  name = fmt::format("{:s}PlieConn1", ext);
  TGeoVolume* voWiggleC1 = new TGeoVolume(name.c_str(), new TGeoTube(rMin + rPlie, rMax - rPlie, dPlie / 2.), kMedAlu5083);
  voWiggleC1->SetLineColor(kOrange); // fm

  //
  // Vacuum Part
  //

  //--Upper part of the ondulation

  TGeoTorus* vacPlieTorusU = new TGeoTorus(rMax - rPlie, 0., rPlie - dPlie);
  nameA = fmt::format("{:s}vacTorusU", ext);
  vacPlieTorusU->SetName(nameA.c_str());
  TGeoTube* vacPlieTubeU = new TGeoTube(0., rMax - rPlie, rPlie - dPlie);
  nameB = fmt::format("{:s}vacTubeU", ext);
  vacPlieTubeU->SetName(nameB.c_str());
  name = fmt::format("{:s}vacUpperPlie", ext);
  TGeoCompositeShape* vacUpperPlie = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}+{:s}", nameA, nameB).c_str());

  TGeoVolume* voVacWiggleU = new TGeoVolume(name.c_str(), vacUpperPlie, kMedVac);
  voVacWiggleU->SetVisibility(0);

  // First Lower part of the undulation
  TGeoTorus* vacPlieTorusL = new TGeoTorus(rMin + rPlie, 0., rPlie);
  nameA = fmt::format("{:s}vacTorusL", ext);
  vacPlieTorusL->SetName(nameA.c_str());

  TGeoTube* vacPlieTubeL = new TGeoTube(0., rMin + rPlie, rPlie / 2.);
  nameB = fmt::format("{:s}vacTubeL", ext);
  vacPlieTubeL->SetName(nameB.c_str());
  name = fmt::format("{:s}vacLowerPlie", ext);
  TGeoCompositeShape* vacLowerPlie1 = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}:t1-{:s}", nameB, nameA).c_str());

  TGeoVolume* voVacWiggleL1 = new TGeoVolume(name.c_str(), vacLowerPlie1, kMedVac);
  voVacWiggleL1->SetVisibility(0);

  // Second Lower part of the undulation
  TGeoCompositeShape* vacLowerPlie2 = new TGeoCompositeShape(name.c_str(), fmt::format("{:s}:t2-{:s}", nameB, nameA).c_str());

  TGeoVolume* voVacWiggleL2 = new TGeoVolume(name.c_str(), vacLowerPlie2, kMedVac);
  voVacWiggleL2->SetVisibility(0);

  // One wiggle
  float dz = rPlie - dPlie / 2.;
  float z0 = 2. * rPlie;
  name = fmt::format("{:s}Wiggle", ext);
  TGeoVolumeAssembly* asWiggle = new TGeoVolumeAssembly(name.c_str());

  asWiggle->AddNode(voWiggleL1, 1, new TGeoTranslation(0., 0., z0));
  asWiggle->AddNode(voVacWiggleL1, 1, new TGeoTranslation(0., 0., z0));
  z0 -= dz;
  asWiggle->AddNode(voWiggleC1, 1, new TGeoTranslation(0., 0., z0));
  z0 -= dz;
  asWiggle->AddNode(voWiggleU, 1, new TGeoTranslation(0., 0., z0));
  asWiggle->AddNode(voVacWiggleU, 1, new TGeoTranslation(0., 0., z0));
  z0 -= dz;
  asWiggle->AddNode(voWiggleC1, 2, new TGeoTranslation(0., 0., z0));
  z0 -= dz;
  asWiggle->AddNode(voWiggleL2, 1, new TGeoTranslation(0., 0., z0));
  asWiggle->AddNode(voVacWiggleL2, 1, new TGeoTranslation(0., 0., z0));

  // Positioning of the volumes
  z0 = +dU / 2.;
  float zsh = 4. * dz;
  // for (int iw = 0; iw < 1; iw++) {
  for (int iw = 0; iw < nc; iw++) {
    float zpos = z0 - iw * zsh;
    voBellow->AddNode(asWiggle, iw + 1, new TGeoTranslation(0., 0., zpos));
  }
  return voBellow;
}

TGeoVolume* PipeRun4::makeSupportBar(const char* tag, float Rin, float Rout, float length, float skinLength)
{
  //
  // make a support bar with the specified dimensions of the collar and arms
  //

  // Dimensions :

  const float kSupportXdim = length; // 20.67;
  const float kBeamPipeRingZdim = 5.25 / 2;
  /* thin layer of material between pipe and support; to be put back later */
  const float kVespelRmax = Rin + 0.08;
  const float kVespelRmin = Rin;
  const float kBeampipeCarbonCollarRmin = Rin + 0.18; // 2.4;
  const float kBeampipeCarbonCollarRmax = Rout;       // 2.7;

  const float kFixationCarbonCollarRmin = 1.5;
  const float kFixationCarbonCollarRmax = 1.7;
  const float kFixationCarbonCollarDZ = 2.5;

  const float kSkinThickness = 0.1;
  const float kSkinXdim = skinLength; // 14.25;
  const float kSkinYdim = 1.;
  const float kSkinZdim = kFixationCarbonCollarDZ;
  const float kCarbonEarsXdim = 1.01;
  const float kCarbonEarsYdim = 0.2;
  const float kCarbonEarsZdim = kFixationCarbonCollarDZ;

  const TGeoMedium* kMedRohacell = gGeoManager->GetMedium("PIPE_ROHACELL");
  const TGeoMedium* kMedPolyimide = gGeoManager->GetMedium("PIPE_POLYIMIDE");
  const TGeoMedium* kMedCarbonFiber = gGeoManager->GetMedium("PIPE_M55J6K");

  TGeoVolume* beamPipeSupport = new TGeoVolumeAssembly(Form("BeampipeSupport_%s", tag));

  // Support Bar
  TGeoVolumeAssembly* supportBar = new TGeoVolumeAssembly(Form("BPS_SupportBar_%s", tag));

  TGeoBBox* carbonSkinBPS = new TGeoBBox(kSkinXdim / 2., kSkinYdim / 2., kSkinZdim / 2.);
  carbonSkinBPS->SetName(Form("carbonSkinBPS_%s", tag));

  TGeoBBox* foambarBPS = new TGeoBBox(Form("foambarBPS_%s", tag), kSkinXdim / 2. - kSkinThickness, kSkinYdim / 2. - kSkinThickness,
                                      kSkinZdim / 2. - kSkinThickness / 2.);
  TGeoBBox* carbonEarsBPS = new TGeoBBox(kCarbonEarsXdim / 2., kCarbonEarsYdim / 2., kCarbonEarsZdim / 2.);
  carbonEarsBPS->SetName(Form("carbonEarsBPS_%s", tag));

  // TODO: could reuse those?..
  TGeoTranslation* transBP1 = new TGeoTranslation(Form("transBP1_%s", tag), (kSkinXdim + kCarbonEarsXdim) / 2., 0., 0.);
  transBP1->RegisterYourself();
  TGeoTranslation* transBP2 = new TGeoTranslation(Form("transBP2_%s", tag), -(kSkinXdim + kCarbonEarsXdim) / 2., 0., 0.);
  transBP2->RegisterYourself();
  TGeoCompositeShape* supportBarCarbon = new TGeoCompositeShape(
    Form("BPS_supportBarCarbon_%s", tag), Form("(carbonSkinBPS_%s-foambarBPS_%s)+carbonEarsBPS_%s:transBP1_%s+carbonEarsBPS_%s:transBP2_%s", tag, tag, tag, tag, tag, tag));

  TGeoVolume* supportBarCarbonVol = new TGeoVolume(Form("BPS_supportBarCarbon_%s", tag), supportBarCarbon, kMedCarbonFiber);
  supportBarCarbonVol->SetLineColor(kGray + 3);

  supportBar->AddNode(supportBarCarbonVol, 1, new TGeoTranslation(kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax, 0, 0));
  supportBar->AddNode(supportBarCarbonVol, 2, new TGeoTranslation(-(kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax), 0, 0));

  TGeoVolume* foamVol = new TGeoVolume(Form("supportBarFoam_%s", tag), foambarBPS, kMedRohacell);
  foamVol->SetLineColor(kGray);
  supportBar->AddNode(foamVol, 1, new TGeoTranslation(kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax, 0, 0));
  supportBar->AddNode(foamVol, 2, new TGeoTranslation(-(kSkinXdim / 2. + kCarbonEarsXdim + kBeampipeCarbonCollarRmax), 0, 0));

  beamPipeSupport->AddNode(supportBar, 1);

  // Fixation to wings
  TGeoVolumeAssembly* fixationToWings = new TGeoVolumeAssembly(Form("BPS_fixationToWings_%s", tag));

  float delatX = 0.1;

  TGeoTubeSeg* fixationTube = new TGeoTubeSeg(kFixationCarbonCollarRmin, kFixationCarbonCollarRmax, kFixationCarbonCollarDZ / 2., -90., 90.);
  fixationTube->SetName(Form("fixationTube_%s", tag));
  TGeoBBox* fixationToBar = new TGeoBBox(kCarbonEarsXdim / 2. + delatX, kCarbonEarsYdim / 2., kCarbonEarsZdim / 2.);
  fixationToBar->SetName(Form("fixationToBar_%s", tag));

  TGeoTranslation* transBP3 = new TGeoTranslation(Form("transBP3_%s", tag), kFixationCarbonCollarRmax + kCarbonEarsXdim / 2. - delatX, kCarbonEarsYdim, 0.);
  transBP3->RegisterYourself();
  TGeoTranslation* transBP4 = new TGeoTranslation(Form("transBP4_%s", tag), kFixationCarbonCollarRmax + kCarbonEarsXdim / 2. - delatX, -kCarbonEarsYdim, 0.);
  transBP4->RegisterYourself();
  TGeoCompositeShape* fixationToWing = new TGeoCompositeShape(Form("fixationToWing_%s", tag), Form("fixationTube_%s+fixationToBar_%s:transBP3_%s+fixationToBar_%s:transBP4_%s", tag, tag, tag, tag, tag));

  TGeoVolume* fixationToWingVol = new TGeoVolume(Form("fixationToWing_%s", tag), fixationToWing, kMedCarbonFiber);
  fixationToWingVol->SetLineColor(kGray + 2);

  fixationToWings->AddNode(fixationToWingVol, 1, new TGeoTranslation(-kSupportXdim, 0, 0));
  fixationToWings->AddNode(fixationToWingVol, 2, new TGeoCombiTrans(+kSupportXdim, 0, 0, new TGeoRotation("rot", 0., 0., 180.)));

  beamPipeSupport->AddNode(fixationToWings, 1);

  // Fixation to pipe

  TGeoVolumeAssembly* fixationToPipe = new TGeoVolumeAssembly(Form("fixationToPipe_%s", tag));

  TGeoTubeSeg* pipeSupportTubeCarbon = new TGeoTubeSeg(kBeampipeCarbonCollarRmin, kBeampipeCarbonCollarRmax, kFixationCarbonCollarDZ / 2., 0., 180.);
  pipeSupportTubeCarbon->SetName(Form("pipeSupportTubeCarbon_%s", tag));

  TGeoBBox* fixationTubeToBar = new TGeoBBox(kCarbonEarsXdim / 2. + delatX, kCarbonEarsYdim / 2., kCarbonEarsZdim / 2.);
  fixationTubeToBar->SetName(Form("fixationTubeToBar_%s", tag));
  TGeoBBox* hole = new TGeoBBox((kBeampipeCarbonCollarRmax - kVespelRmin) / 2., kCarbonEarsYdim / 2., kCarbonEarsZdim / 2. + 1e-3);
  hole->SetName(Form("hole_%s", tag));

  TGeoTranslation* transBP5 = new TGeoTranslation(Form("transBP5_%s", tag), kBeampipeCarbonCollarRmax + kCarbonEarsXdim / 2. - delatX, kCarbonEarsYdim, 0.);
  transBP5->RegisterYourself();
  TGeoTranslation* transBP6 = new TGeoTranslation(Form("transBP6_%s", tag), -(kBeampipeCarbonCollarRmax + kCarbonEarsXdim / 2. - delatX), kCarbonEarsYdim, 0.);
  transBP6->RegisterYourself();
  TGeoTranslation* transBP7 = new TGeoTranslation(Form("transBP7_%s", tag), (kBeampipeCarbonCollarRmax + kVespelRmin) / 2., 0., 0.);
  transBP7->RegisterYourself();
  TGeoTranslation* transBP8 = new TGeoTranslation(Form("transBP8_%s", tag), -((kBeampipeCarbonCollarRmax + kVespelRmin) / 2.), 0., 0.);
  transBP8->RegisterYourself();
  TGeoCompositeShape* halfFixationToPipe = new TGeoCompositeShape(
    Form("halfFixationToPipe_%s", tag),
    Form("(pipeSupportTubeCarbon_%s-hole_%s:transBP7_%s-hole_%s:transBP8_%s)+fixationTubeToBar_%s:transBP5_%s+fixationTubeToBar_%s:transBP6_%s", tag, tag, tag, tag, tag, tag, tag, tag, tag));

  TGeoVolume* halfFixationToPipeVol = new TGeoVolume(Form("halfFixationToPipe_%s", tag), halfFixationToPipe, kMedCarbonFiber);
  halfFixationToPipeVol->SetLineColor(kRed + 2);

  fixationToPipe->AddNode(halfFixationToPipeVol, 1);
  fixationToPipe->AddNode(halfFixationToPipeVol, 2, new TGeoCombiTrans(0, 0, 0, new TGeoRotation("rot", 0., 0., 180.)));

  beamPipeSupport->AddNode(fixationToPipe, 1);

  // Beam Pipe Ring

  TGeoVolumeAssembly* beamPipeRing = new TGeoVolumeAssembly(Form("beamPipeRing_%s", tag));

  TGeoTube* beamPipeRingCarbon = new TGeoTube(kVespelRmax, kBeampipeCarbonCollarRmin, kBeamPipeRingZdim / 2.);
  TGeoVolume* beamPipeRingCarbonVol = new TGeoVolume(Form("beamPipeRingCarbon_%s", tag), beamPipeRingCarbon, kMedCarbonFiber);
  beamPipeRingCarbonVol->SetLineColor(kGreen + 2);
  beamPipeRing->AddNode(beamPipeRingCarbonVol, 1, new TGeoTranslation(0., 0, (kBeamPipeRingZdim - kFixationCarbonCollarDZ) / 2.));

  TGeoTube* beamPipeRingVespel = new TGeoTube(kVespelRmin, kVespelRmax, kBeamPipeRingZdim / 2.);
  TGeoVolume* beamPipeRingVespelVol = new TGeoVolume(Form("beamPipeRingVespel_%s", tag), beamPipeRingVespel, kMedPolyimide);
  beamPipeRingVespelVol->SetLineColor(kGreen + 4);
  beamPipeRing->AddNode(beamPipeRingVespelVol, 1, new TGeoTranslation(0., 0, (kBeamPipeRingZdim - kFixationCarbonCollarDZ) / 2.));

  beamPipeSupport->AddNode(beamPipeRing, 1);
  beamPipeSupport->SetVisibility(0);

  return beamPipeSupport;
}

// ----------------------------------------------------------------------------
FairModule* PipeRun4::CloneModule() const { return new PipeRun4(*this); }
ClassImp(o2::passive::PipeRun4);
