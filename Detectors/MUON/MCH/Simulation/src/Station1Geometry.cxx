// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Station1Geometry.cxx
/// \brief  Implementation of the station 1 geometry (copied and adapted from AliMUONSt1GeometryBuilder)
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   16 may 2018

#include "Materials.h"
#include "Station1Geometry.h"

#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoShape.h>
#include <TGeoArb8.h>
#include <TGeoXtru.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>

#include <iostream>
#include <string>
#include <array>

namespace o2
{
namespace mch
{

/// Constants

// chamber z position (from AliMUONConstants)
const float kChamberZPos[2] = { -526.16, -545.24 };

// quadrant z position w.r.t the chamber center
const float kQuadrantZPos = 7.5 / 2;

// thickness
const float kHzPadPlane = 0.0148 / 2; // pad plane
const float kHzFoam = 2.503 / 2;      // foam of mechanical plane
const float kHzFR4 = 0.062 / 2;       // FR4 of mechanical plane
const float kTotalHzPlane = kHzFoam + kHzFR4;
const float kHzSnPb = 0.0091 / 2;        // pad / kapton connection (66 pt)
const float kHzKapton = 0.0122 / 2;      // kapton
const float kHzBergPlastic = 0.3062 / 2; // Berg connector
const float kHzBergCopper = 0.1882 / 2;  // Berg connector
const float kHzDaughter = 0.0156 / 2;    // daughter board
const float kHzGas = 0.42 / 2;           // gas

// spacers
const float kHxBoxSpacer = 0.51;
const float kHySpacer5A = 0.17;
const float kHzSpacer5A = 1.1515;
const float kHySpacer6 = 1.5;
const float kHzSpacer6 = 0.1;
const float kRSpacer7A = 0.3;
const float kHzSpacer7A = 0.1;

// quadrant mother volume
const float kMotherPhiL = 0.;
const float kMotherPhiU = 90.;

// TUBS1 - Middle layer of model
const float kMotherIR1 = 18.3;
const float kMotherOR1 = 105.673;
const float kMotherThick1 = 6.5 / 2;

// TUBS2 - near and far layers of model
const float kMotherIR2 = 20.7;
const float kMotherOR2 = 100.073;
const float kMotherThick2 = 1.5;

// sensitive copper pads, foam layer, PCB and electronics model parameters
const float kHxHole = 1.5 / 2;
const float kHyHole = 3.;
const float kHxBergPlastic = 0.74 / 2;
const float kHyBergPlastic = 5.09 / 2;
const float kHxBergCopper = 0.25 / 2;
const float kHyBergCopper = 3.6 / 2;
const float kHxKapton = 0.4;
const float kHyKapton = 5.7 / 2;
const float kHxDaughter = 2.3 / 2;
const float kHyDaughter = 6.3 / 2;
const float kOffsetX = 1.46;
const float kOffsetY = 0.71;
const float kDeltaFilleEtamX = 1.;
const float kDeltaFilleEtamY = 0.051;

// lateral positionner parameters
const float kLateralXPosShift = 92.175;
const float kLateralYPosShift = 5.;

// trapezoid angles
const float kThetaTrap = 0.;
const float kPhiTrap = 0.;

// parameters relative to the LHC beam pipe
const float kNearFarLHC = 2.4;   // Near and Far TUBS Origin wrt LHC Origin
const float kDeltaQuadLHC = 2.6; // LHC Origin wrt Quadrant Origin
const float kFrameOffset = 5.2;

// pad plane offsets
const float kPadXOffsetBP = 0.50 - 0.63 / 2; // = 0.185
const float kPadYOffsetBP = 0.31 + 0.42 / 2; // = 0.52
const int kFoamBoxNameOffset = 200;
const int kFR4BoxNameOffset = 400;
const int kDaughterCopyNoOffset = 1000;

// volume names
const char* kHoleName = "SCHL";
const char* kDaughterName = "SCDB";
const char* kQuadrantMLayerName = "SQM";
const char* kQuadrantNLayerName = "SQN";
const char* kQuadrantFLayerName = "SQF";
const char* kQuadrantMFLayerName = "SQMF";

//______________________________________________________________________________
void createHole()
{
  /// Create all the elements found inside a foam hole

  auto hole = new TGeoVolumeAssembly(kHoleName);

  hole->AddNode(gGeoManager->MakeBox("SNPB", assertMedium(Medium::Copper), kHxKapton, kHyKapton, kHzSnPb), 1,
                new TGeoTranslation(0., 0., -kHzFoam + kHzSnPb));

  hole->AddNode(gGeoManager->MakeBox("SKPT", assertMedium(Medium::Copper), kHxHole, kHyBergPlastic, kHzKapton), 1);
}

//______________________________________________________________________________
void createDaughterBoard()
{
  /// Create all the elements in a daughter board

  auto daughter = new TGeoVolumeAssembly(kDaughterName);

  daughter->AddNode(gGeoManager->MakeBox("SBGP", assertMedium(Medium::Plastic), kHxBergPlastic, kHyBergPlastic, kHzBergPlastic),
                    1, new TGeoTranslation(0., 0., -kHzDaughter));

  daughter->AddNode(gGeoManager->MakeBox("SBGC", assertMedium(Medium::Copper), kHxBergCopper, kHyBergCopper, kHzBergCopper), 1);

  daughter->AddNode(gGeoManager->MakeBox("SDGH", assertMedium(Medium::Copper), kHxDaughter, kHyDaughter, kHzDaughter),
                    1, new TGeoTranslation(0., 0., -kHzBergPlastic));
}

//______________________________________________________________________________
void createInnerLayers()
{
  /// Create the layer of sensitive volumes with gas and the copper layer.
  /// The shape of the sensitive area is defined as an extruded solid substracted with tube (to get inner circular shape).

  new TGeoTube("cutTube", 0., kMotherIR1, kHzPadPlane + kHzGas);

  // Xtru parameters
  double maxXY = 89.;
  double xy1 = 77.33;
  double xy2 = 48.77;
  double dxy1 = maxXY - xy1;

  const int nz = 2;
  const int nv = 6;
  double vx[nv] = { 0., 0., xy2, maxXY, maxXY, dxy1 };
  double vy[nv] = { dxy1, maxXY, maxXY, xy2, 0., 0. };

  // layer parameters
  const auto kGasMed = assertMedium(Medium::Gas);
  const auto kPadMed = assertMedium(Medium::Copper);

  const int kNLayers = 3;
  const std::string kLayerName[kNLayers] = { "SA1G", "SA2G", "SA1C" };
  const std::array<const TGeoMedium*, kNLayers> kLayerMedium = { kGasMed, kGasMed, kPadMed };
  const double kLayerZpos[kNLayers] = { kHzGas, kHzGas, kHzPadPlane };

  for (int i = 0; i < kNLayers; i++) {
    TGeoXtru* xtruS = new TGeoXtru(nz);
    xtruS->SetName(Form("xtruS%d", i + 1));
    xtruS->DefinePolygon(nv, vx, vy);
    xtruS->DefineSection(0, -kLayerZpos[i], 0., 0., 1.);
    xtruS->DefineSection(1, kLayerZpos[i], 0., 0., 1.);
    new TGeoVolume(kLayerName[i].data(), new TGeoCompositeShape(Form("layerS%d", i + 1), Form("xtruS%d-cutTube", i + 1)),
                   kLayerMedium[i]);
  }
}

//______________________________________________________________________________
void createSpacer()
{
  /// The spacer volumes are defined according to the input prepared by Nicole Williswith modifications needed to fit into existing geometry.

  /// No.    Type  Material Center (mm)            Dimensions (mm) (half lengths)
  ///  5     BOX   EPOXY    408.2  430.4 522.41    5.75  1.5   25.5
  ///  5P    BOX   EPOXY    408.2  445.4 522.41    5.75  1.5   25.5
  ///  6     BOX   EPOXY    408.2  437.9 519.76    5.75  15.0   1.0
  ///  6P    BOX   EPOXY    408.2  437.9 525.06    5.75  15.0   1.0
  ///  7     CYL   INOX     408.2  437.9 522.41    r=3.0  hz=20.63

  /// To fit in existing volumes the volumes 5 and 7 are represented by 2 volumes
  /// with half size in z (5A, 7A); the dimensions of the volume 5A were also modified
  /// to avoid overlaps (x made smaller, y larger to obtain the identical volume)

  new TGeoVolume("Spacer5A", new TGeoBBox(kHxBoxSpacer, kHySpacer5A, kHzSpacer5A), assertMedium(Medium::Epoxy));

  new TGeoVolume("Spacer6", new TGeoBBox(kHxBoxSpacer, kHySpacer6, kHzSpacer6), assertMedium(Medium::Epoxy));

  new TGeoVolume("Spacer7A", new TGeoTube(0., kRSpacer7A, kHzSpacer7A), assertMedium(Medium::Inox));
}

//______________________________________________________________________________
void createFrame(int chamber)
{
  /// Create the non-sensitive elements of the frame for the \a chamber
  ///
  /// Model and notation:                                                     \n
  ///                                                                         \n
  /// The Quadrant volume name starts with SQ                                 \n
  /// The volume segments are numbered 00 to XX                               \n
  ///                                                                         \n
  ///                              OutTopFrame                                \n
  ///                               (SQ02-16)                                 \n
  ///                              ------------                               \n
  ///             OutEdgeFrame   /              |                             \n
  ///             (SQ17-24)     /               |  InVFrame (SQ00-01)         \n
  ///                          /                |                             \n
  ///                          |                |                             \n
  ///               OutVFrame  |            _- -                              \n
  ///               (SQ25-39)  |           |   InArcFrame (SQ42-45)           \n
  ///                          |           |                                  \n
  ///                          -------------                                  \n
  ///                        InHFrame (SQ40-41)                               \n
  ///                                                                         \n
  ///                                                                         \n
  /// 06 February 2003 - Overlapping volumes resolved.                        \n
  /// One quarter chamber is comprised of three TUBS volumes: SQMx, SQNx, and SQFx,
  /// where SQMx is the Quadrant Middle layer for chamber \a chamber ( z in [-3.25,3.25]),
  /// SQNx is the Quadrant Near side layer for chamber \a chamber ( z in [-6.25,3-.25) ), and
  /// SQFx is the Quadrant Far side layer for chamber \a chamber ( z in (3.25,6.25] ).

  auto Mlayer = new TGeoVolumeAssembly(Form("%s%d", kQuadrantMLayerName, chamber));
  auto Flayer = new TGeoVolumeAssembly(Form("%s%d", kQuadrantFLayerName, chamber));
  auto MFlayer = new TGeoVolumeAssembly(Form("%s%d", kQuadrantMFLayerName, chamber));
  auto Nlayer = new TGeoVolumeAssembly(Form("%s%d", kQuadrantNLayerName, chamber));

  // rotation matrices
  auto rot1 = new TGeoRotation("rot1", 90., 90., 90., 180., 0., 0.); // +90 deg in x-y plane
  auto rot4 = new TGeoRotation("rot4", 90., 315., 90., 45., 0., 0.); // -45 deg in x-y plane

  /// Volume thicknesses

  const float kHzFrameThickness = 1.59 / 2;
  const float kHzOuterFrameEpoxy = 1.19 / 2;
  const float kHzOuterFrameInox = 0.1 / 2;
  const float kHzFoam2 = 2.083 / 2;

  // Pertaining to the top outer area
  const float kHzTopAnodeSteel1 = 0.185 / 2;
  const float kHzTopAnodeSteel2 = 0.51 / 2;
  const float kHzAnodeFR4 = 0.08 / 2;
  const float kHzTopEarthFaceCu = 0.364 / 2;
  const float kHzTopEarthProfileCu = 1.1 / 2;
  const float kHzTopPositionerSteel = 1.45 / 2; // should really be 2.125/2.;
  const float kHzTopGasSupportAl = 0.85 / 2;

  // Pertaining to the vertical outer area
  const float kHzVerticalCradleAl = 0.8 / 2;
  const float kHzLateralSightAl = 0.975 / 2;
  const float kHzLateralPosnInoxFace = 2.125 / 2;
  const float kHzLatPosInoxProfM = 6.4 / 2;
  const float kHzLatPosInoxProfNF = 1.45 / 2;
  const float kHzLateralPosnAl = 0.5 / 2;
  const float kHzVertEarthFaceCu = 0.367 / 2;
  const float kHzVertBarSteel = 0.198 / 2;
  const float kHzVertEarthProfCu = 1.1 / 2;

  // parameter definitions in sequence

  // InVFrame parameters
  const float kHxInVFrame = 1.85 / 2;
  const float kHyInVFrame = 73.95 / 2;
  const float kHzInVFrame = kHzFrameThickness;

  // Flat 7.5mm vertical section
  const float kHxV1mm = 0.75 / 2;
  const float kHyV1mm = 1.85 / 2;
  const float kHzV1mm = kHzFrameThickness;

  // OuterTopFrame Structure
  //
  // FRAME
  // The frame is composed of a cuboid and two trapezoids
  // (TopFrameAnode, TopFrameAnodeA, TopFrameAnodeB).
  // Each shape is composed of two layers (Epoxy and Inox) and
  // takes the frame's inner anode circuitry into account in the material budget.
  //
  // ANODE
  // The overhanging anode part is composed froma cuboid and two trapezoids
  // (TopAnode, TopAnode1, and TopAnode2). These surfaces neglect implanted
  // resistors, but accounts for the major Cu, Pb/Sn, and FR4 material
  // contributions.
  // The stainless steel anode supports have been included.
  //
  // EARTHING (TopEarthFace, TopEarthProfile)
  // Al GAS SUPPORT (TopGasSupport)
  //
  // ALIGNMENT (TopPositioner) - Alignment system, three sights per quarter
  // chamber. This sight is forseen for the alignment of the horizontal level
  // (parallel to the OY axis of LHC). Its position will be evaluated relative
  // to a system of sights places on the cradles;

  // TopFrameAnode parameters - cuboid, 2 layers
  const float kHxTFA = 34.1433 / 2;
  const float kHyTFA = 7.75 / 2;
  const float kHzTFAE = kHzOuterFrameEpoxy; // layer 1 thickness
  const float kHzTFAI = kHzOuterFrameInox;  // layer 3 thickness

  // TopFrameAnode parameters - 2 trapezoids, 2 layers (redefined with TGeoXtru shape)
  const float kH1FAA = 8.7 / 2;
  const float kTl1FAB = 4.35 / 2;
  const float kTl1FAA = 7.75 / 2;

  // TopAnode parameters - cuboid (part 1 of 3 parts)
  const float kHxTA1 = 16.2 / 2;
  const float kHyTA1 = 3.5 / 2;
  const float kHzTA11 = kHzTopAnodeSteel1; // layer 1
  const float kHzTA12 = kHzAnodeFR4;       // layer 2

  // TopAnode parameters - trapezoid 1 (part 2 of 3 parts)
  const float kHzTA21 = kHzTopAnodeSteel2; // layer 1
  const float kHzTA22 = kHzAnodeFR4;       // layer 2
  const float kHTA2 = 7.268 / 2;
  const float kBlTA2 = 2.03 / 2;
  const float kTlTA2 = 3.5 / 2;
  const float kAlpTA2 = 5.78;

  // TopAnode parameters - trapezoid 2 (part 3 of 3 parts)
  const float kHzTA3 = kHzAnodeFR4; // layer 1
  const float kHTA3 = 7.268 / 2;
  const float kBlTA3 = 0.;
  const float kTlTA3 = 2.03 / 2;
  const float kAlpTA3 = 7.95;

  // TopEarthFace parameters - single trapezoid
  const float kHzTEF = kHzTopEarthFaceCu;
  const float kHTEF = 1.2 / 2;
  const float kBlTEF = 21.323 / 2;
  const float kTlTEF = 17.963 / 2;
  const float kAlpTEF = -54.46;

  // TopEarthProfile parameters - single trapezoid
  const float kHzTEP = kHzTopEarthProfileCu;
  const float kHTEP = 0.2;
  const float kBlTEP = 31.766 / 2;
  const float kTlTEP = 30.535 / 2;
  const float kAlpTEP = -56.98;

  // TopPositioner parameters - single Stainless Steel trapezoid
  const float kHzTP = kHzTopPositionerSteel;
  const float kHTP = 1.5;
  const float kBlTP = 7.023 / 2;
  const float kTlTP = 7.314 / 2;
  const float kAlpTP = 2.78;

  // TopGasSupport parameters - single cuboid
  const float kHxTGS = 8.5 / 2;
  const float kHyTGS = 1.5;
  const float kHzTGS = kHzTopGasSupportAl;

  // OutEdgeFrame parameters - 4 trapezoidal sections, 2 layers of material (redefined with TGeoXtru shape)
  const float kH1OETF = 7.196 / 2;   // common to all 4 trapezoids
  const float kTl1OETF1 = 3.996 / 2; // Trapezoid 1
  const float kTl1OETF2 = 3.75 / 2;  // Trapezoid 2
  const float kTl1OETF3 = 3.01 / 2;  // Trapezoid 3
  const float kTl1OETF4 = 1.77 / 2;  // Trapezoid 4

  /// Frame Structure (OutVFrame):

  // OutVFrame and corner (OutVFrame cuboid, OutVFrame trapezoid)
  // EARTHING (VertEarthFaceCu,VertEarthSteel,VertEarthProfCu),
  // DETECTOR POSITIONNING (SuppLateralPositionner, LateralPositionner),
  // CRADLE (VertCradle), and
  // ALIGNMENT (LateralSightSupport, LateralSight)

  // OutVFrame parameters - cuboid
  const float kHxOutVFrame = 1.85 / 2;
  const float kHyOutVFrame = 46.23 / 2;
  const float kHzOutVFrame = kHzFrameThickness;

  // OutVFrame corner parameters - trapezoid
  const float kHzOCTF = kHzFrameThickness;
  const float kHOCTF = 1.85 / 2;
  const float kBlOCTF = 0.;
  const float kTlOCTF = 3.66 / 2;
  const float kAlpOCTF = 44.67;

  // VertEarthFaceCu parameters - single trapezoid
  const float kHzVFC = kHzVertEarthFaceCu;
  const float kHVFC = 0.6;
  const float kBlVFC = 46.11 / 2;
  const float kTlVFC = 48.236 / 2;
  const float kAlpVFC = 41.54;

  // VertEarthSteel parameters - single trapezoid
  const float kHzVES = kHzVertBarSteel;
  const float kHVES = 0.6;
  const float kBlVES = 30.486 / 2;
  const float kTlVES = 32.777 / 2;
  const float kAlpVES = 43.67;

  // VertEarthProfCu parameters - single trapezoid
  const float kHzVPC = kHzVertEarthProfCu;
  const float kHVPC = 0.2;
  const float kBlVPC = 29.287 / 2;
  const float kTlVPC = 30.091 / 2;
  const float kAlpVPC = 45.14;

  // SuppLateralPositionner - single cuboid
  const float kHxSLP = 1.4;
  const float kHySLP = 2.5;
  const float kHzSLP = kHzLateralPosnAl;

  // LateralPositionner - squared off U bend, face view
  const float kHxLPF = 2.6;
  const float kHyLPF = 1.5;
  const float kHzLPF = kHzLateralPosnInoxFace;

  // LateralPositionner - squared off U bend, profile view
  const float kHxLPP = 0.425 / 2;
  const float kHyLPP = 1.5;
  const float kHzLPP = kHzLatPosInoxProfM;   // middle layer
  const float kHzLPNF = kHzLatPosInoxProfNF; // near and far layers

  // VertCradle, 3 layers (copies), each composed of 4 trapezoids (redefined with TGeoXtru shape)
  const float kH1VC1 = 10.25 / 2;  // all cradles
  const float kBl1VC1 = 3.7 / 2;   // VertCradleA
  const float kBl1VC2 = 6.266 / 2; // VertCradleB
  const float kBl1VC3 = 7.75 / 2;  // VertCradleC

  // VertCradleD
  const float kHzVC4 = kHzVerticalCradleAl;
  const float kHVC4 = 10.27 / 2;
  const float kBlVC4 = 8.273 / 2;
  const float kTlVC4 = 7.75 / 2;
  const float kAlpVC4 = -1.46;

  // LateralSightSupport - single trapezoid
  const float kHzVSS = kHzLateralSightAl;
  const float kHVSS = 2.5;
  const float kBlVSS = 7.747 / 2;
  const float kTlVSS = 7.188 / 2;
  const float kAlpVSS = -3.2;

  // LateralSight (reference point) - 3 per quadrant, only 1 programmed for now
  const float kVSInRad = 0.6;
  const float kVSOutRad = 1.3;
  const float kVSLen = kHzFrameThickness;

  // InHFrame parameters
  const float kHxInHFrame = 75.8 / 2;
  const float kHyInHFrame = 1.85 / 2;
  const float kHzInHFrame = kHzFrameThickness;

  // Flat 7.5mm horizontal section
  const float kHxH1mm = 1.85 / 2;
  const float kHyH1mm = 0.75 / 2;
  const float kHzH1mm = kHzFrameThickness;

  // InArcFrame parameters
  const float kIAF = 15.7;
  const float kOAF = 17.55;
  const float kHzAF = kHzFrameThickness;
  const float kAFphi1 = 0.;
  const float kAFphi2 = 90.;

  // ScrewsInFrame parameters HEAD
  const float kSCRUHMI = 0.;
  const float kSCRUHMA = 0.69 / 2;
  const float kSCRUHLE = 0.2;
  // ScrewsInFrame parameters MIDDLE
  const float kSCRUMMI = 0.;
  const float kSCRUMMA = 0.39 / 2;
  const float kSCRUMLE = kHzFrameThickness;
  // ScrewsInFrame parameters NUT
  const float kSCRUNMI = 0.;
  const float kSCRUNMA = 0.78 / 2;
  const float kSCRUNLE = 0.4;

  const int npar = 11;
  float par[npar];

  if (chamber == 1) {
    // materials
    const auto kEpoxyMed = assertMedium(Medium::Epoxy);
    const auto kInoxMed = assertMedium(Medium::Inox);
    const auto kCopperMed = assertMedium(Medium::Copper);
    const auto kAluMed = assertMedium(Medium::Aluminium);
    const auto kFR4Med = assertMedium(Medium::FR4);

    // InVFrame
    new TGeoVolume("SQ00", new TGeoBBox(kHxInVFrame, kHyInVFrame, kHzInVFrame), kEpoxyMed);

    // Flat 1mm vertical section
    new TGeoVolume("SQ01", new TGeoBBox(kHxV1mm, kHyV1mm, kHzV1mm), kEpoxyMed);

    /// OutTopFrame - 3 components (a cuboid and 2 trapezes) and 2 layers (Epoxy/Inox)

    // TopFrameAnode - layer 1 of 2
    new TGeoVolume("SQ02", new TGeoBBox(kHxTFA, kHyTFA, kHzTFAE), kEpoxyMed);

    // TopFrameAnode - layer 2 of 2
    new TGeoVolume("SQ03", new TGeoBBox(kHxTFA, kHyTFA, kHzTFAI), kInoxMed);

    // common declarations for TGeoXtru parameters
    double dx, dx0, dx1, dx2, dx3, dy, dy1, dy2, dy3, dy4;
    double vx[16], vy[16];
    int nz = 2, nv = 5;

    // SQ04to06 and SQ05to07

    dx = 2 * kH1FAA;
    dy1 = 2 * kTl1FAA;
    dy2 = 2 * kTl1FAB;

    vx[0] = 0.;
    vy[0] = 0.;
    vx[1] = 0.;
    vy[1] = dy1;
    vx[2] = dx;
    vy[2] = dy2;
    vx[3] = 2 * dx;
    vy[3] = 0.;
    vx[4] = dx;
    vy[4] = 0.;

    // shift center in the middle
    for (int i = 0; i < nv; i++) {
      vx[i] -= dx;
      vy[i] -= dy1 / 2;
    }

    TGeoXtru* xtruS5 = new TGeoXtru(nz);
    xtruS5->DefinePolygon(nv, vx, vy);
    xtruS5->DefineSection(0, -kHzOuterFrameEpoxy, 0., 0., 1.);
    xtruS5->DefineSection(1, kHzOuterFrameEpoxy, 0., 0., 1.);
    new TGeoVolume("SQ04toSQ06", xtruS5, kEpoxyMed);

    TGeoXtru* xtruS6 = new TGeoXtru(nz);
    xtruS6->DefinePolygon(nv, vx, vy);
    xtruS6->DefineSection(0, -kHzOuterFrameInox, 0., 0., 1.);
    xtruS6->DefineSection(1, kHzOuterFrameInox, 0., 0., 1.);
    new TGeoVolume("SQ05toSQ07", xtruS6, kInoxMed);

    // TopAnode1 -  layer 1 of 2
    new TGeoVolume("SQ08", new TGeoBBox(kHxTA1, kHyTA1, kHzTA11), kInoxMed);

    // TopAnode1 -  layer 2 of 2
    new TGeoVolume("SQ09", new TGeoBBox(kHxTA1, kHyTA1, kHzTA12), kFR4Med);

    // TopAnode2 -  layer 1 of 2
    par[0] = kHzTA21;
    par[1] = kThetaTrap; // defined once for all here !
    par[2] = kPhiTrap;   // defined once for all here !
    par[3] = kHTA2;
    par[4] = kBlTA2;
    par[5] = kTlTA2;
    par[6] = kAlpTA2;
    par[7] = kHTA2;
    par[8] = kBlTA2;
    par[9] = kTlTA2;
    par[10] = kAlpTA2;
    gGeoManager->Volume("SQ10", "TRAP", kInoxMed->GetId(), par, npar);

    // TopAnode2 -  layer 2 of 2
    par[0] = kHzTA22;
    gGeoManager->Volume("SQ11", "TRAP", kFR4Med->GetId(), par, npar);

    // TopAnode3 -  layer 1 of 1
    par[0] = kHzTA3;
    par[3] = kHTA3;
    par[4] = kBlTA3;
    par[5] = kTlTA3;
    par[6] = kAlpTA3;
    par[7] = kHTA3;
    par[8] = kBlTA3;
    par[9] = kTlTA3;
    par[10] = kAlpTA3;
    gGeoManager->Volume("SQ12", "TRAP", kFR4Med->GetId(), par, npar);

    // TopEarthFace
    par[0] = kHzTEF;
    par[3] = kHTEF;
    par[4] = kBlTEF;
    par[5] = kTlTEF;
    par[6] = kAlpTEF;
    par[7] = kHTEF;
    par[8] = kBlTEF;
    par[9] = kTlTEF;
    par[10] = kAlpTEF;
    gGeoManager->Volume("SQ13", "TRAP", kCopperMed->GetId(), par, npar);

    // TopEarthProfile
    par[0] = kHzTEP;
    par[3] = kHTEP;
    par[4] = kBlTEP;
    par[5] = kTlTEP;
    par[6] = kAlpTEP;
    par[7] = kHTEP;
    par[8] = kBlTEP;
    par[9] = kTlTEP;
    par[10] = kAlpTEP;
    gGeoManager->Volume("SQ14", "TRAP", kCopperMed->GetId(), par, npar);

    // TopGasSupport
    new TGeoVolume("SQ15", new TGeoBBox(kHxTGS, kHyTGS, kHzTGS), assertMedium(Medium::Aluminium));

    // TopPositioner parameters - single Stainless Steel trapezoid
    par[0] = kHzTP;
    par[3] = kHTP;
    par[4] = kBlTP;
    par[5] = kTlTP;
    par[6] = kAlpTP;
    par[7] = kHTP;
    par[8] = kBlTP;
    par[9] = kTlTP;
    par[10] = kAlpTP;
    gGeoManager->Volume("SQ16", "TRAP", kInoxMed->GetId(), par, npar);

    // OutEdgeTrapFrame Epoxy = (4 trapezes)*2 copies*2 layers (Epoxy/Inox) (redefined with TGeoXtru shape )

    dx = 2 * kH1OETF;
    dy1 = 2 * kTl1OETF4;
    dy2 = 2 * kTl1OETF3;
    dy3 = 2 * kTl1OETF2;
    dy4 = 2 * kTl1OETF1;

    nz = 2;
    nv = 16;
    vx[0] = -4 * dx;
    vy[0] = 0.;
    vx[1] = -3 * dx;
    vy[1] = dy1;
    vx[2] = -2 * dx;
    vy[2] = dy2;
    vx[3] = -dx;
    vy[3] = dy3;
    vx[4] = 0.;
    vy[4] = dy4;
    vx[5] = dx;
    vy[5] = dy3;
    vx[6] = 2 * dx;
    vy[6] = dy2;
    vx[7] = 3 * dx;
    vy[7] = dy1;
    vx[8] = 4 * dx;
    vy[8] = 0.;
    vx[9] = 3 * dx;
    vy[9] = 0.;
    vx[10] = 2 * dx;
    vy[10] = 0.;
    vx[11] = dx;
    vy[11] = 0.;
    vx[12] = 0.;
    vy[12] = 0.;
    vx[13] = -dx;
    vy[13] = 0.;
    vx[14] = -2 * dx;
    vy[14] = 0.;
    vx[15] = -3 * dx;
    vy[15] = 0.;

    // shift center in the middle
    for (int i = 0; i < nv; i++)
      vy[i] += dy4 / 2;

    TGeoXtru* xtruS1 = new TGeoXtru(nz);
    xtruS1->DefinePolygon(nv, vx, vy);
    xtruS1->DefineSection(0, -kHzOuterFrameEpoxy, 0., 0., 1.);
    xtruS1->DefineSection(1, kHzOuterFrameEpoxy, 0., 0., 1.);
    new TGeoVolume("SQ17to23", xtruS1, kEpoxyMed);

    TGeoXtru* xtruS2 = new TGeoXtru(nz);
    xtruS2->DefinePolygon(nv, vx, vy);
    xtruS2->DefineSection(0, -kHzOuterFrameInox, 0., 0., 1.);
    xtruS2->DefineSection(1, kHzOuterFrameInox, 0., 0., 1.);
    new TGeoVolume("SQ18to24", xtruS2, kInoxMed);

    // OutEdgeTrapFrame Epoxy = (4 trapezes)*2 copies*2 layers (Epoxy/Inox)

    // OutVFrame
    new TGeoVolume("SQ25", new TGeoBBox(kHxOutVFrame, kHyOutVFrame, kHzOutVFrame), kEpoxyMed);

    // OutVFrame corner
    par[0] = kHzOCTF;
    par[3] = kHOCTF;
    par[4] = kBlOCTF;
    par[5] = kTlOCTF;
    par[6] = kAlpOCTF;
    par[7] = kHOCTF;
    par[8] = kBlOCTF;
    par[9] = kTlOCTF;
    par[10] = kAlpOCTF;
    gGeoManager->Volume("SQ26", "TRAP", kEpoxyMed->GetId(), par, npar);

    // EarthFaceCu trapezoid
    par[0] = kHzVFC;
    par[3] = kHVFC;
    par[4] = kBlVFC;
    par[5] = kTlVFC;
    par[6] = kAlpVFC;
    par[7] = kHVFC;
    par[8] = kBlVFC;
    par[9] = kTlVFC;
    par[10] = kAlpVFC;
    gGeoManager->Volume("SQ27", "TRAP", kCopperMed->GetId(), par, npar);

    // VertEarthSteel trapezoid
    par[0] = kHzVES;
    par[3] = kHVES;
    par[4] = kBlVES;
    par[5] = kTlVES;
    par[6] = kAlpVES;
    par[7] = kHVES;
    par[8] = kBlVES;
    par[9] = kTlVES;
    par[10] = kAlpVES;
    gGeoManager->Volume("SQ28", "TRAP", kInoxMed->GetId(), par, npar);

    // VertEarthProfCu trapezoid
    par[0] = kHzVPC;
    par[3] = kHVPC;
    par[4] = kBlVPC;
    par[5] = kTlVPC;
    par[6] = kAlpVPC;
    par[7] = kHVPC;
    par[8] = kBlVPC;
    par[9] = kTlVPC;
    par[10] = kAlpVPC;
    gGeoManager->Volume("SQ29", "TRAP", kCopperMed->GetId(), par, npar);

    // SuppLateralPositionner cuboid
    new TGeoVolume("SQ30", new TGeoBBox(kHxSLP, kHySLP, kHzSLP), kAluMed);

    // LateralPositionerFace
    new TGeoVolume("SQ31", new TGeoBBox(kHxLPF, kHyLPF, kHzLPF), kInoxMed);

    // LateralPositionerProfile
    new TGeoVolume("SQ32", new TGeoBBox(kHxLPP, kHyLPP, kHzLPP), kInoxMed); // middle layer

    new TGeoVolume("SQ33", new TGeoBBox(kHxLPP, kHyLPP, kHzLPNF), kInoxMed); // near and far layers

    dy = 2 * kH1VC1;
    dx0 = 2 * kBlVC4;
    dx1 = 2 * kBl1VC3;
    dx2 = 2 * kBl1VC2;
    dx3 = 2 * kBl1VC1;

    // VertCradle (trapezoids SQ34 to SQ36 or SQ37 redefined with TGeoXtru shape)

    nz = 2;
    nv = 7;
    vx[0] = 0.;
    vy[0] = 0.;
    vx[1] = 0.;
    vy[1] = dy;
    vx[2] = 0.;
    vy[2] = 2 * dy;
    vx[3] = 0.;
    vy[3] = 3 * dy;
    vx[4] = dx3;
    vy[4] = 2 * dy;
    vx[5] = dx2;
    vy[5] = dy;
    vx[6] = dx1;
    vy[6] = 0.;

    // shift center in the middle
    for (int i = 0; i < nv; i++) {
      vx[i] -= dx1 / 2;
      vy[i] -= 1.5 * dy;
    }

    TGeoXtru* xtruS3 = new TGeoXtru(nz);
    xtruS3->DefinePolygon(nv, vx, vy);
    xtruS3->DefineSection(0, -kHzVerticalCradleAl, 0., 0., 1.);
    xtruS3->DefineSection(1, kHzVerticalCradleAl, 0., 0., 1.);
    new TGeoVolume("SQ34to36", xtruS3, kAluMed);

    // Trapezoids SQ34 to SQ37 (keeping the same coordinate system as for SQ34to36)

    nz = 2;
    nv = 9;
    vx[0] = 0.;
    vy[0] = -dy;
    vx[1] = 0.;
    vy[1] = 0.;
    vx[2] = 0.;
    vy[2] = dy;
    vx[3] = 0.;
    vy[3] = 2 * dy;
    vx[4] = 0.;
    vy[4] = 3 * dy;
    vx[5] = dx3;
    vy[5] = 2 * dy;
    vx[6] = dx2;
    vy[6] = dy;
    vx[7] = dx1;
    vy[7] = 0.;
    vx[8] = dx0;
    vy[8] = -dy;

    // shift center in the middle (of SQ34to36!!)
    for (int i = 0; i < nv; i++) {
      vx[i] -= dx1 / 2;
      vy[i] -= 1.5 * dy;
    }

    TGeoXtru* xtruS4 = new TGeoXtru(nz);
    xtruS4->DefinePolygon(nv, vx, vy);
    xtruS4->DefineSection(0, -kHzVerticalCradleAl, 0., 0., 1.);
    xtruS4->DefineSection(1, kHzVerticalCradleAl, 0., 0., 1.);
    new TGeoVolume("SQ34to37", xtruS4, kAluMed);

    // VertCradleD - 4th trapezoid
    par[0] = kHzVC4;
    par[3] = kHVC4;
    par[4] = kBlVC4;
    par[5] = kTlVC4;
    par[6] = kAlpVC4;
    par[7] = kHVC4;
    par[8] = kBlVC4;
    par[9] = kTlVC4;
    par[10] = kAlpVC4;
    gGeoManager->Volume("SQ37", "TRAP", kAluMed->GetId(), par, npar);

    // LateralSightSupport trapezoid
    par[0] = kHzVSS;
    par[3] = kHVSS;
    par[4] = kBlVSS;
    par[5] = kTlVSS;
    par[6] = kAlpVSS;
    par[7] = kHVSS;
    par[8] = kBlVSS;
    par[9] = kTlVSS;
    par[10] = kAlpVSS;
    gGeoManager->Volume("SQ38", "TRAP", kAluMed->GetId(), par, npar);

    // LateralSight
    new TGeoVolume("SQ39", new TGeoTube(kVSInRad, kVSOutRad, kVSLen), kEpoxyMed);

    // InHFrame
    new TGeoVolume("SQ40", new TGeoBBox(kHxInHFrame, kHyInHFrame, kHzInHFrame), kEpoxyMed);

    // Flat 7.5mm horizontal section
    new TGeoVolume("SQ41", new TGeoBBox(kHxH1mm, kHyH1mm, kHzH1mm), kEpoxyMed);

    // InArcFrame
    new TGeoVolume("SQ42", new TGeoTubeSeg(kIAF, kOAF, kHzAF, kAFphi1, kAFphi2), kEpoxyMed);

    // ScrewsInFrame - 3 sections in order to avoid overlapping volumes
    // Screw Head, in air
    new TGeoVolume("SQ43", new TGeoTube(kSCRUHMI, kSCRUHMA, kSCRUHLE), kInoxMed);

    // Middle part, in the Epoxy
    new TGeoVolume("SQ44", new TGeoTube(kSCRUMMI, kSCRUMMA, kSCRUMLE), kInoxMed);

    // Screw nut, in air
    new TGeoVolume("SQ45", new TGeoTube(kSCRUNMI, kSCRUNMA, kSCRUNLE), kInoxMed);
  }

  /// Place volumes in the quadrant

  // InVFrame
  float x = kHxInVFrame;
  float y = 2 * (kHyInHFrame + kHyH1mm) + kIAF + kHyInVFrame;
  float z = 0.;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ00"), 1, new TGeoTranslation(x, y, z));

  // keep memory of the mid position (to place screws)
  const float kMidVXPos = x;
  const float kMidVYPos = y;
  const float kMidVZPos = z;

  // Flat 7.5mm vertical section
  x = 2 * kHxInVFrame + kHxV1mm;
  y = 2 * (kHyInHFrame + kHyH1mm) + kIAF + kHyV1mm;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ01"), 1, new TGeoTranslation(x, y, z));

  // TopFrameAnode - place 2 layers of TopFrameAnode cuboids
  x = kHxTFA;
  y = 2 * (kHyInHFrame + kHyH1mm + kHyInVFrame) + kIAF + kHyTFA;
  z = kHzOuterFrameInox;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ02"), 1, new TGeoTranslation(x, y, -z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ03"), 1, new TGeoTranslation(x, y, z));

  // TopFrameAnode - place 2 layers of 2 trapezoids (SQ04 - SQ07)
  x += kHxTFA + 2 * kH1FAA;
  z = kHzOuterFrameInox;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ04toSQ06"), 1, new TGeoTranslation(x, y, -z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ05toSQ07"), 1, new TGeoTranslation(x, y, z));

  // TopAnode1 - place 2 layers
  x = 6.8 + kDeltaQuadLHC;
  y = 99.85 + kDeltaQuadLHC;
  z = -kHzAnodeFR4;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ08"), 1, new TGeoTranslation(x, y, z));
  z = kHzTopAnodeSteel1;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ09"), 1, new TGeoTranslation(x, y, z));

  // TopAnode2 - place 2 layers
  x = 18.534 + kDeltaQuadLHC;
  y = 99.482 + kDeltaQuadLHC;
  z = -kHzAnodeFR4;
  // shift up to solve overlap with SQ14
  y += 0.1;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ10"), 1, new TGeoCombiTrans(x, y, z, rot1));
  z = kHzTopAnodeSteel2;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ11"), 1, new TGeoCombiTrans(x, y, z, rot1));

  // TopAnode3 - place 1 layer
  x = 25.804 + kDeltaQuadLHC;
  y = 98.61 + kDeltaQuadLHC;
  z = 0.;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ12"), 1, new TGeoCombiTrans(x, y, z, rot1));

  // TopEarthFace - 2 copies
  x = 23.122 + kDeltaQuadLHC;
  y = 96.9 + kDeltaQuadLHC;
  z = kHzOuterFrameEpoxy + kHzOuterFrameInox + kHzTopEarthFaceCu;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ13"), 1, new TGeoTranslation(x, y, z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ13"), 2, new TGeoTranslation(x, y, -z));

  // TopEarthProfile
  x = 14.475 + kDeltaQuadLHC;
  y = 97.9 + kDeltaQuadLHC;
  z = kHzTopEarthProfileCu;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ14"), 1, new TGeoTranslation(x, y, z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ14"), 2, new TGeoTranslation(x, y, -z));

  // TopGasSupport - 2 copies
  x = 4.95 + kDeltaQuadLHC;
  y = 96.2 + kDeltaQuadLHC;
  z = kHzOuterFrameEpoxy + kHzOuterFrameInox + kHzTopGasSupportAl;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ15"), 1, new TGeoTranslation(x, y, z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ15"), 2, new TGeoTranslation(x, y, -z));

  // TopPositioner parameters - single Stainless Steel trapezoid - 2 copies
  x = 7.6 + kDeltaQuadLHC;
  y = 98.98 + kDeltaQuadLHC;
  z = kHzOuterFrameEpoxy + kHzOuterFrameInox + 2 * kHzTopGasSupportAl + kHzTopPositionerSteel;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ16"), 1, new TGeoTranslation(x, y, z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ16"), 2, new TGeoTranslation(x, y, -z));

  // OutEdgeFrame
  z = -kHzOuterFrameInox;
  float xCenterAll = 70.5, yCenterAll = 70.35;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ17to23"), 1, new TGeoCombiTrans(xCenterAll, yCenterAll, z, rot4));

  z = kHzOuterFrameEpoxy;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ18to24"), 1, new TGeoCombiTrans(xCenterAll, yCenterAll, z, rot4));

  // OutVFrame
  x = 2 * (kHxInVFrame + kHxInHFrame + kHxV1mm) + kIAF - kHxOutVFrame;
  y = 2 * kHyInHFrame + kHyOutVFrame;
  z = 0.;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ25"), 1, new TGeoTranslation(x, y, z));

  // keep memory of the mid position (to place screws)
  const float kMidOVXPos = x;
  const float kMidOVYPos = y;
  const float kMidOVZPos = z;

  // OutVFrame corner
  y += kHyOutVFrame + (kBlOCTF + kTlOCTF) / 2;
  // shift to solve overlap with SQ17to23 and SQ18to24
  x += 0.02;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ26"), 1, new TGeoCombiTrans(x, y, z, rot1));

  // VertEarthFaceCu - 2 copies
  x = 89.4 + kDeltaQuadLHC;
  y = 25.79 + kDeltaQuadLHC;
  z = kHzFrameThickness + 2 * kHzFoam2 + kHzVertEarthFaceCu;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ27"), 1, new TGeoCombiTrans(x, y, z, rot1));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ27"), 2, new TGeoCombiTrans(x, y, -z, rot1));

  // VertEarthSteel - 2 copies
  x = 91 + kDeltaQuadLHC;
  y = 30.616 + kDeltaQuadLHC;
  z = kHzFrameThickness + 2 * kHzFoam2 + kHzVertBarSteel;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ28"), 1, new TGeoCombiTrans(x, y, z, rot1));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ28"), 2, new TGeoCombiTrans(x, y, -z, rot1));

  // VertEarthProfCu - 2 copies
  x = 92 + kDeltaQuadLHC;
  y = 29.64 + kDeltaQuadLHC;
  z = kHzFrameThickness;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ29"), 1, new TGeoCombiTrans(x, y, z, rot1));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ29"), 2, new TGeoCombiTrans(x, y, -z, rot1));

  // SuppLateralPositionner - 2 copies
  x = 90.2 - kNearFarLHC;
  y = kLateralYPosShift - kNearFarLHC;
  z = kHzLateralPosnAl - kMotherThick2;
  Flayer->AddNode(gGeoManager->GetVolume("SQ30"), 1, new TGeoTranslation(x, y, z));
  Nlayer->AddNode(gGeoManager->GetVolume("SQ30"), 2, new TGeoTranslation(x, y, -z));

  /// Lateral positionners

  // Face view
  x = kLateralXPosShift - kNearFarLHC - 2 * kHxLPP;
  y = kLateralYPosShift - kNearFarLHC;
  z = 2 * kHzLateralPosnAl + kHzLateralPosnInoxFace - kMotherThick2;
  Flayer->AddNode(gGeoManager->GetVolume("SQ31"), 1, new TGeoTranslation(x, y, z));
  Nlayer->AddNode(gGeoManager->GetVolume("SQ31"), 2, new TGeoTranslation(x, y, -z));

  // Profile view
  x = kLateralXPosShift + kDeltaQuadLHC + kHxLPF - kHxLPP;
  y = kLateralYPosShift + kDeltaQuadLHC;
  z = 0.;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ32"), 1, new TGeoTranslation(x, y, z));

  x = kLateralXPosShift - kNearFarLHC + kHxLPF - kHxLPP;
  y = kLateralYPosShift - kNearFarLHC;
  z = kMotherThick2 - kHzLPNF;
  Nlayer->AddNode(gGeoManager->GetVolume("SQ33"), 1, new TGeoTranslation(x, y, z));
  Flayer->AddNode(gGeoManager->GetVolume("SQ33"), 2, new TGeoTranslation(x, y, -z));

  // VertCradle - 3 (or 4 ) trapezoids redefined with TGeoXtru shape
  const float kVertCradleX = 97.29;
  const float kVertCradleXshift = 1.39311;
  const float kVertCradleY = 23.02;

  x = kVertCradleX + kDeltaQuadLHC + kVertCradleXshift;
  y = kVertCradleY + kDeltaQuadLHC;
  z = 0.;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ34to37"), 2, new TGeoTranslation(x, y, z));

  x = kVertCradleX - kNearFarLHC + kVertCradleXshift;
  y = kVertCradleY - kNearFarLHC;
  z = 2 * kHzLateralSightAl + kHzVerticalCradleAl - kMotherThick2;
  Nlayer->AddNode(gGeoManager->GetVolume("SQ34to36"), 1, new TGeoTranslation(x, y, z));
  Flayer->AddNode(gGeoManager->GetVolume("SQ34to36"), 3, new TGeoTranslation(x, y, -z));

  // OutVertCradleD 4th Trapeze - 3 copies
  x = 98.81 + kDeltaQuadLHC;
  y = 2.52 + kDeltaQuadLHC;
  z = kMotherThick1 - kHzVerticalCradleAl;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ37"), 1, new TGeoTranslation(x, y, z));
  Mlayer->AddNode(gGeoManager->GetVolume("SQ37"), 3, new TGeoTranslation(x, y, -z));

  // LateralSightSupport - 2 copies
  x = 98.33 - kNearFarLHC;
  y = 10 - kNearFarLHC;
  z = kHzLateralSightAl - kMotherThick2;

  Nlayer->AddNode(gGeoManager->GetVolume("SQ38"), 1, new TGeoTranslation(x, y, z));
  Flayer->AddNode(gGeoManager->GetVolume("SQ38"), 2, new TGeoTranslation(x, y, -z));

  // mire placement
  x = 92.84 + kDeltaQuadLHC;
  y = 8.13 + kDeltaQuadLHC;
  z = 0.;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ39"), 1, new TGeoTranslation(x, y, z));

  /// InHFrame
  x = 2 * (kHxInVFrame + kHxV1mm) + kIAF + kHxInHFrame;
  y = kHyInHFrame;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ40"), 1, new TGeoTranslation(x, y, z));

  // keep memory of the mid position (to place screws)
  const float kMidHXPos = x;
  const float kMidHYPos = y;
  const float kMidHZPos = z;

  // flat 7.5 mm horizontal section
  x = 2 * (kHxInVFrame + kHxV1mm) + kIAF + kHxH1mm;
  y = 2 * kHyInHFrame + kHyH1mm;
  Mlayer->AddNode(gGeoManager->GetVolume("SQ41"), 1, new TGeoTranslation(x, y, z));

  // InArcFrame
  x = 2 * (kHxInVFrame + kHxV1mm);
  y = 2 * (kHyInHFrame + kHyH1mm);
  Mlayer->AddNode(gGeoManager->GetVolume("SQ42"), 1, new TGeoTranslation(x, y, z));

  // keep memory of the mid position (to place screws)
  const float kMidArcXPos = x;
  const float kMidArcYPos = y;
  const float kMidArcZPos = z;

  // ScrewsInFrame - in sensitive volume
  const int kNScrews = 64;

  float scruX[kNScrews], scruY[kNScrews];
  // screw volumes
  auto vol43 = gGeoManager->GetVolume("SQ43"), vol44 = gGeoManager->GetVolume("SQ44"), vol45 = gGeoManager->GetVolume("SQ45");

  const float kSpecScrewPos = -2.23;

  // screws on IHEpoxyFrame
  const int kNScrewsIH = 14; // number of screws on the IHEpoxyFrame
  const float kOffX = 5.;    // inter-screw distance

  // first screw coordinates
  scruX[0] = 21.07;
  scruY[0] = kSpecScrewPos;
  // other screw coordinates
  for (int i = 1; i < kNScrewsIH; i++) {
    scruX[i] = scruX[i - 1] + kOffX;
    scruY[i] = scruY[0];
  }

  // place the volumes on the frames
  z = 0.;
  for (int i = 0; i < kNScrewsIH; i++) {
    x = kDeltaQuadLHC + scruX[i] + 0.1;
    y = kDeltaQuadLHC + scruY[i] + 0.1;
    Mlayer->AddNode(vol43, i + 1, new TGeoTranslation(x, y, z - kHzInHFrame - kSCRUHLE));
    if (chamber == 1)
      gGeoManager->GetVolume("SQ40")->AddNode(vol44, i + 1, new TGeoTranslation(x - kMidHXPos, y - kMidHYPos, z - kMidHZPos));

    Mlayer->AddNode(vol45, i + 1, new TGeoTranslation(x, y, z + kHzInHFrame + kSCRUNLE));
  }

  // special screw coordinates
  scruX[63] = 16.3;
  scruY[63] = kSpecScrewPos;
  x = kDeltaQuadLHC + scruX[63] + 0.1;
  y = kDeltaQuadLHC + scruY[63] + 0.1;
  z = 0.;

  Mlayer->AddNode(vol43, kNScrews, new TGeoTranslation(x, y, z - kHzInHFrame - kSCRUHLE));

  if (chamber == 1)
    gGeoManager->GetVolume("SQ40")->AddNode(vol44, kNScrews, new TGeoTranslation(x - kMidHXPos, y - kMidHYPos, z - kMidHZPos));

  Mlayer->AddNode(vol45, kNScrews, new TGeoTranslation(x, y, z + kHzInHFrame + kSCRUNLE));

  // screws on the IVEpoxyFrame
  const float kOffY = 5.; // inter-screw distance
  int firstScrew = 58, lastScrew = 44;

  // first (special) screw coordinates
  scruX[firstScrew - 1] = kSpecScrewPos;
  scruY[firstScrew - 1] = 16.3;
  // second (repetitive) screw coordinates
  scruX[firstScrew - 2] = kSpecScrewPos;
  scruY[firstScrew - 2] = 21.07;
  // other screw coordinates
  for (int i = firstScrew - 3; i > lastScrew - 2; i--) {
    scruX[i] = scruX[firstScrew - 2];
    scruY[i] = scruY[i + 1] + kOffY;
  }

  z = 0.;
  for (int i = lastScrew; i <= firstScrew; i++) {
    x = kDeltaQuadLHC + scruX[i - 1] + 0.1;
    y = kDeltaQuadLHC + scruY[i - 1] + 0.1;

    Mlayer->AddNode(vol43, i, new TGeoTranslation(x, y, z - kHzInHFrame - kSCRUHLE));

    if (chamber == 1)
      gGeoManager->GetVolume("SQ00")->AddNode(vol44, i, new TGeoTranslation(x - kMidVXPos, y - kMidVYPos, z - kMidVZPos));

    Mlayer->AddNode(vol45, i, new TGeoTranslation(x, y, z + kHzInHFrame + kSCRUNLE));
  }

  // screws on the OVEpoxyFrame

  firstScrew = 15;
  lastScrew = 25;

  // first (repetitive) screw coordinates
  scruX[firstScrew - 1] = 90.9;
  scruY[firstScrew - 1] = kSpecScrewPos;

  // other screw coordinates
  for (int i = firstScrew; i < lastScrew; i++) {
    scruX[i] = scruX[firstScrew - 1];
    scruY[i] = scruY[i - 1] + kOffY;
  }

  z = 0.;
  for (int i = 1 + firstScrew; i < lastScrew; i++) {
    x = kDeltaQuadLHC + scruX[i - 1] + 0.1;
    y = kDeltaQuadLHC + scruY[i - 1] + 0.1;

    Mlayer->AddNode(vol43, i, new TGeoTranslation(x, y, z - kHzInHFrame - kSCRUHLE));
    if (chamber == 1)
      gGeoManager->GetVolume("SQ25")->AddNode(vol44, i, new TGeoTranslation(x - kMidOVXPos, y - kMidOVYPos, z - kMidOVZPos));

    Mlayer->AddNode(vol45, i, new TGeoTranslation(x, y, z + kHzInHFrame + kSCRUNLE));
  }

  // special case for 1st screw, inside the horizontal frame (volume 40)
  x = kDeltaQuadLHC + scruX[firstScrew - 1] + 0.1 - kMidHXPos;
  y = kDeltaQuadLHC + scruY[firstScrew - 1] + 0.1 - kMidHYPos;
  z = -kMidHZPos;

  if (chamber == 1)
    gGeoManager->GetVolume("SQ40")->AddNode(vol44, firstScrew, new TGeoTranslation(x, y, z));

  // inner arc of Frame, screw positions and numbers
  firstScrew = 58;
  scruX[62] = 16.009;
  scruY[62] = 1.401;
  scruX[61] = 14.564;
  scruY[61] = 6.791;
  scruX[60] = 11.363;
  scruY[60] = scruX[60];
  scruX[59] = scruY[61];
  scruY[59] = scruX[61];
  scruX[58] = scruY[62];
  scruY[58] = scruX[62];

  z = 0.;
  for (int i = firstScrew; i < firstScrew + 5; i++) {
    x = kDeltaQuadLHC + scruX[i] + 0.1;
    y = kDeltaQuadLHC + scruY[i] + 0.1;

    Mlayer->AddNode(vol43, i + 1, new TGeoTranslation(x, y, z - kHzInHFrame - kSCRUHLE));

    if (chamber == 1)
      gGeoManager->GetVolume("SQ42")->AddNode(vol44, i + 1, new TGeoTranslation(x - kMidArcXPos, y - kMidArcYPos, z - kMidArcZPos));

    Mlayer->AddNode(vol45, i + 1, new TGeoTranslation(x, y, z + kHzInHFrame + kSCRUNLE));
  }
}

//______________________________________________________________________________
TGeoVolume* createPlaneSegment(int iSegment, float halfLength, float halfHeight, int nHoles)
{
  /// create a plane segment (this includes a foam layer, holes in the foam to feed the kaptons through, kapton connectors and the mother board)

  auto segment = new TGeoVolumeAssembly(Form("S%d", iSegment));

  // variables
  float x = 0., y = 0., z = 0.;

  // foam layer
  const int kFoamNumber = iSegment + kFoamBoxNameOffset;
  const char* kFoamBoxName = Form("S%d", kFoamNumber);

  auto foam = gGeoManager->MakeBox(kFoamBoxName, assertMedium(Medium::St1Rohacell), halfLength, halfHeight, kHzFoam);

  // place spacers in the concrete plane segments:
  // S225 (in S025), S267 (in S067) in chamber1 and S309 (in S109). S351(in S151) in chamber2
  // The segments were found as those which caused overlaps when we placed the spacer in global coordinates via PlaceSpacer0

  //    <posXYZ   X_Y_Z=" 12.6000;   0.75000;   0.0000"> <volume name="Spacer5A"/>
  //    <posXYZ   X_Y_Z=" 12.6000;  -0.75000;   0.0000"> <volume name="Spacer5A"/>
  //    <posXYZ   X_Y_Z=" 12.6000;   0.0000;    1.1515"> <volume name="Spacer6"/>
  //    <posXYZ   X_Y_Z=" 12.6000;   0.0000;    0.0000"> <volume name="Spacer7A"/>
  bool is267or351 = (kFoamNumber == 267 || kFoamNumber == 351);

  if (kFoamNumber == 225 || is267or351 || kFoamNumber == 309) {
    x = 12.6;
    y = 0.75;
    z = -0.1;

    if (is267or351)
      y += kPadYOffsetBP;

    foam->AddNode(gGeoManager->GetVolume("Spacer5A"), 1, new TGeoTranslation(x, y, z));

    y = -0.75;
    if (is267or351)
      y += kPadYOffsetBP;

    foam->AddNode(gGeoManager->GetVolume("Spacer5A"), 2, new TGeoTranslation(x, y, z));

    y = 0.;
    z = 1.1515;
    if (is267or351)
      y += kPadYOffsetBP;

    foam->AddNode(gGeoManager->GetVolume("Spacer6"), 1, new TGeoTranslation(x, y, z));

    y = 0.;
    z = 0.;
    if (is267or351)
      y += kPadYOffsetBP;

    foam->AddNode(gGeoManager->GetVolume("Spacer7A"), 1, new TGeoTranslation(x, y, z));
  }

  y = 0.;
  z = 0.;

  auto hole = gGeoManager->GetVolume(kHoleName);
  for (int holeNum = 0; holeNum < nHoles; holeNum++) {
    x = ((2 * holeNum + 1) / nHoles - 1) * halfLength;

    foam->AddNode(hole, holeNum, new TGeoTranslation(x, y, z));
  }

  z = -kTotalHzPlane + kHzFoam; // - kHzFR4
  segment->AddNode(foam, 1, new TGeoTranslation(0., 0., z));

  // mechanical plane (FR4 layer)
  z = kHzFoam;
  segment->AddNode(gGeoManager->MakeBox(Form("S%d", iSegment + kFR4BoxNameOffset), assertMedium(Medium::FR4), halfLength, halfHeight, kHzFR4),
                   1, new TGeoTranslation(0., 0., z));

  return segment;
}

/*
//______________________________________________________________________________
void placeSector(const AliMpSector* sector, TExMap specialMap, const TVector3& where, Bool_t reflectZ, Int_t chamber)
{
  /// Place all the segments in the mother volume, at the position defined
  /// by the sector's data.
  /// The lines with comments COMMENT OUT BEGIN/END indicates blocks
  /// which can be commented out in order to reduce the number of volumes
  /// in a sector to the plane segments corresponding to regular motifs only.
  static Int_t segNum = 1;
  Int_t sgn;
  Int_t reflZ;
  Int_t rotMat;
  if (!reflectZ) {
    sgn = 1;
    reflZ = 0;                                            // no reflection along z... nothing
    fMUON->AliMatrix(rotMat, 90., 90., 90, 180., 0., 0.); // 90� rotation around z, NO reflection along z
  } else {
    sgn = -1;
    fMUON->AliMatrix(reflZ, 90., 0., 90, 90., 180., 0.);    // reflection along z
    fMUON->AliMatrix(rotMat, 90., 90., 90, 180., 180., 0.); // 90� rotation around z AND reflection along z
  }
  GReal_t x, y, z;
  TArrayI alreadyDone(20);
  Int_t nofAlreadyDone = 0;
  for (Int_t irow = 0; irow < sector->GetNofRows(); irow++) { // for each row
    AliMpRow* row = sector->GetRow(irow);
    for (Int_t iseg = 0; iseg < row->GetNofRowSegments(); iseg++) { // for each row segment
      AliMpVRowSegment* seg = row->GetRowSegment(iseg);
      Long_t value = specialMap.GetValue(seg->GetMotifPositionId(0));
      if (value == 0) { //if this is a normal segment (ie. not part of <specialMap>)
        // create the cathode part
        auto segment = createPlaneSegment(segNum, seg->GetDimensionX()/2,seg->GetDimensionY()/2,seg->GetNofMotifs());
        x = where.X() + seg->GetPositionX();
        y = where.Y() + seg->GetPositionY();
        z = where.Z() + sgn * (kTotalHzPlane + kHzGas + 2. * kHzPadPlane);
        auto MLayer = gGeoManager->GetVolume(Form("%s%d", kQuadrantMLayerName, chamber));
        MLayer->AddNode(segment,1,new TGeoCombiTrans(x, y, z, reflZ));
        // and place all the daughter boards of this segment
        // COMMENT OUT BEGIN
        for (Int_t motifNum = 0; motifNum < seg->GetNofMotifs(); motifNum++) {
          // Copy number
          Int_t motifPosId = seg->GetMotifPositionId(motifNum);
          AliMpMotifPosition* motifPos =
            sector->GetMotifMap()->FindMotifPosition(motifPosId);
          Int_t copyNo = motifPosId;
          if (sector->GetDirection() == AliMp::kX)
            copyNo += kDaughterCopyNoOffset;
          // Position
          x = where.X() + motifPos->GetPositionX() + fgkOffsetX;
          y = where.Y() + motifPos->GetPositionY() + fgkOffsetY;
          z = where.Z() + sgn * (kMotherThick1 - TotalHzDaughter());
          TVirtualMC::GetMC()->Gspos(kDaughterName, copyNo, QuadrantMLayerName(chamber), x, y, z, reflZ, "ONLY");
        }
        // COMMENT OUT END
        segNum++;
      } else {
        // COMMENT OUT BEGIN
        // if this is a special segment
        for (Int_t motifNum = 0; motifNum < seg->GetNofMotifs(); motifNum++) { // for each motif
          Int_t motifPosId = seg->GetMotifPositionId(motifNum);
          Bool_t isDone = false;
          Int_t i = 0;
          while (i < nofAlreadyDone && !isDone) {
            if (alreadyDone.At(i) == motifPosId)
              isDone = true;
            i++;
          }
          if (isDone)
            continue; // don't treat the same motif twice
          AliMUONSt1SpecialMotif spMot = *((AliMUONSt1SpecialMotif*)specialMap.GetValue(motifPosId));
          AliDebugStream(2) << chamber << " processing special motif: " << motifPosId << endl;
          AliMpMotifPosition* motifPos = sector->GetMotifMap()->FindMotifPosition(motifPosId);
          // Copy number
          Int_t copyNo = motifPosId;
          if (sector->GetDirection() == AliMp::kX)
            copyNo += kDaughterCopyNoOffset;
          // place the hole for the motif, wrt the requested rotation angle
          Int_t rot = (spMot.GetRotAngle() < 0.1) ? reflZ : rotMat;
          x = where.X() + motifPos->GetPositionX() + spMot.GetDelta().X();
          y = where.Y() + motifPos->GetPositionY() + spMot.GetDelta().Y();
          z = where.Z() + sgn * (kTotalHzPlane + kHzGas + 2. * kHzPadPlane);
          // Shift the hole for special motif 46 to avoid debording into S047
          if (copyNo == 2070) {
            x -= 0.1;
            y -= 0.1;
          }
          TVirtualMC::GetMC()->Gspos(kHoleName, copyNo, QuadrantMLayerName(chamber), x, y, z, rot, "ONLY");
          // then place the daughter board for the motif, wrt the requested rotation angle
          x = x + kDeltaFilleEtamX;
          y = y + kDeltaFilleEtamY;
          // Do not shift the daughter board
          if (copyNo == 2070) {
            x += 0.1;
            y += 0.1;
          }
          z = where.Z() + sgn * (kMotherThick1 - TotalHzDaughter());
          TVirtualMC::GetMC()->Gspos(kDaughterName, copyNo, QuadrantMLayerName(chamber), x, y, z, rot, "ONLY");
          if (nofAlreadyDone == alreadyDone.GetSize())
            alreadyDone.Set(2 * nofAlreadyDone);
          alreadyDone.AddAt(motifPosId, nofAlreadyDone++);
          AliDebugStream(2) << chamber << " processed motifPosId: " << motifPosId << endl;
        }
        // COMMENT OUT END
      } // end of special motif case
    }
  }
}
*/
//______________________________________________________________________________

void placeInnerLayers(int chamber)
{
  /// place the gas and the copper layers for the specified chamber.

  float x = kDeltaQuadLHC, y = kDeltaQuadLHC, zc = kHzGas + kHzPadPlane;
  int dpos = 2 * (chamber - 1);

  auto layer = gGeoManager->GetVolume(Form("%s%d", kQuadrantMLayerName, chamber));
  layer->AddNode(gGeoManager->GetVolume(Form("SA%dG", chamber)), 1, new TGeoTranslation(x, y, 0.));
  layer->AddNode(gGeoManager->GetVolume("SA1C"), 1 + dpos, new TGeoTranslation(x, y, zc));
  layer->AddNode(gGeoManager->GetVolume("SA1C"), 2 + dpos, new TGeoTranslation(x, y, -zc));
}

//______________________________________________________________________________
TGeoVolumeAssembly* createQuadrant(int chamber)
{
  /// create the quadrant (bending and non-bending planes) for the given chamber
  auto quadrant = new TGeoVolumeAssembly(Form("Quadrant (chamber %d)", chamber));

  createFrame(chamber);

  /*  TODO
  TExMap specialMap;
  specialMap.Add(76, (Long_t) new AliMUONSt1SpecialMotif(TVector2(0.1, 0.72), 90.));
  specialMap.Add(75, (Long_t) new AliMUONSt1SpecialMotif(TVector2(0.7, 0.36)));
  specialMap.Add(47, (Long_t) new AliMUONSt1SpecialMotif(TVector2(1.01, 0.36)));
  // Load mapping from OCDB
  if (!AliMpSegmentation::Instance()) {
    AliFatal("Mapping has to be loaded first !");
  }
  const AliMpSector* kSector1 =
    AliMpSegmentation::Instance()->GetSector(100, AliMpDEManager::GetCathod(100, AliMp::kBendingPlane));
  if (!kSector1) {
    AliFatal("Could not access sector segmentation !");
  }
  // Bool_t reflectZ = true;
  Bool_t reflectZ = false;
  // TVector3 where = TVector3(2.5+0.1+0.56+0.001, 2.5+0.1+0.001, 0.);
  TVector3 where = TVector3(fgkDeltaQuadLHC + fgkPadXOffsetBP, fgkDeltaQuadLHC + fgkPadYOffsetBP, 0.);
  PlaceSector(kSector1, specialMap, where, reflectZ, chamber);
  int nb = AliMpConstants::ManuMask(AliMp::kNonBendingPlane);
  TExMapIter it(&specialMap);
  #if (defined(ROOT_SVN_REVISION) && ROOT_SVN_REVISION >= 29598) || \
  (defined(ROOT_VERSION_CODE) && ROOT_VERSION_CODE >= ROOT_VERSION(5, 25, 02))
  Long64_t key;
  Long64_t value;
  #else
  Long_t key;
  Long_t value;
  #endif
  while (it.Next(key, value) == kTRUE) {
    delete reinterpret_cast<AliMUONSt1SpecialMotif*>(value);
  }
  specialMap.Delete();
  specialMap.Add(76 | nb, (Long_t) new AliMUONSt1SpecialMotif(TVector2(1.01, 0.51), 90.));
  specialMap.Add(75 | nb, (Long_t) new AliMUONSt1SpecialMotif(TVector2(2.20, -0.08)));
  specialMap.Add(47 | nb, (Long_t) new AliMUONSt1SpecialMotif(TVector2(2.40, -1.11)));
  specialMap.Add(20 | nb, (Long_t) new AliMUONSt1SpecialMotif(TVector2(0.2, -0.08)));
  specialMap.Add(46 | nb, (Long_t) new AliMUONSt1SpecialMotif(TVector2(0.92, 0.17)));
  specialMap.Add(74 | nb, (Long_t) new AliMUONSt1SpecialMotif(TVector2(0.405, -0.10)));
  // Fix (7) - overlap of SQ42 with MCHL (after moving the whole sector
  // in the true position)
  const AliMpSector* kSector2 =
    AliMpSegmentation::Instance()->GetSector(100, AliMpDEManager::GetCathod(100, AliMp::kNonBendingPlane));
  if (!kSector2) {
    AliFatal("Could not access sector !");
  }
  // reflectZ = false;
  reflectZ = true;
  TVector2 offset = TVector2(kSector2->GetPositionX(), kSector2->GetPositionY());
  where = TVector3(where.X() + offset.X(), where.Y() + offset.Y(), 0.);
  // Add the half-pad shift of the non-bending plane wrt bending plane
  // (The shift is defined in the mapping as sector offset)
  // Fix (4) - was TVector3(where.X()+0.63/2, ... - now it is -0.63/2
  PlaceSector(kSector2, specialMap, where, reflectZ, chamber);
  it.Reset();
  while (it.Next(key, value) == kTRUE) {
    delete reinterpret_cast<AliMUONSt1SpecialMotif*>(value);
  }
  specialMap.Delete();
  */

  // place gas volumes
  placeInnerLayers(chamber);

  // middle layers
  float x = -(kDeltaQuadLHC + kPadXOffsetBP), y = -(kDeltaQuadLHC + kPadYOffsetBP), z = 0.;
  quadrant->AddNode(gGeoManager->GetVolume(Form("%s%d", kQuadrantMLayerName, chamber)), 1, new TGeoTranslation(x, y, z));
  quadrant->AddNode(gGeoManager->GetVolume(Form("%s%d", kQuadrantMFLayerName, chamber)), 1, new TGeoTranslation(x, y, z));

  // near/far layers
  x += kFrameOffset;
  y += kFrameOffset;
  quadrant->AddNode(gGeoManager->GetVolume(Form("%s%d", kQuadrantFLayerName, chamber)), 1, new TGeoTranslation(x, y, z));

  z -= kMotherThick1 + kMotherThick2;
  quadrant->AddNode(gGeoManager->GetVolume(Form("%s%d", kQuadrantNLayerName, chamber)), 1, new TGeoTranslation(x, y, z));

  return quadrant;
}

//______________________________________________________________________________
void createStation1Geometry(TGeoVolume& topVolume)
{

  // create basic volumes
  createHole();
  createDaughterBoard();
  createInnerLayers();
  createSpacer();

  const int kNQuadrants = 4;

  auto rot0 = new TGeoRotation();
  auto rot1 = new TGeoRotation("reflXZ", 90., 180., 90., 90., 180., 0.);
  auto rot2 = new TGeoRotation("reflXY", 90., 180., 90., 270., 0., 0.);
  auto rot3 = new TGeoRotation("reflYZ", 90., 0., 90., -90., 180., 0.);
  std::array<TGeoRotation*, kNQuadrants> rot = { rot0, rot1, rot2, rot3 };

  // initialize the quadrant positions
  float x[kNQuadrants] = { -1, 1, 1, -1 };
  float y[kNQuadrants] = { -1, -1, 1, 1 };

  for (int i = 0; i < kNQuadrants; i++) {
    x[i] *= kPadXOffsetBP;
    y[i] *= kPadYOffsetBP;
  }

  // build the two chambers
  int detElemID = 0;
  float z = kQuadrantZPos;

  for (int ich = 1; ich < 3; ich++) {

    // create two half-chambers
    auto* in = new TGeoVolumeAssembly(Form("SC0%dI", ich));
    auto* out = new TGeoVolumeAssembly(Form("SC0%dO", ich));

    // create a quadrant volume
    auto* quadrant = createQuadrant(ich);

    // place the quadrants in the half-chambers
    for (int i = 0; i < kNQuadrants; i++) {
      // alternate the z position
      z *= -1.;

      // compute the detection element ID
      detElemID = 100 * ich + i;

      if (x[i] < 0) {
        in->AddNode(quadrant, detElemID, new TGeoCombiTrans(x[i], y[i], z, rot[i]));
      } else
        out->AddNode(quadrant, detElemID, new TGeoCombiTrans(x[i], y[i], z, rot[i]));
    }

    // place the half-chambers in the top volume
    topVolume.AddNode(in, 2 * (ich - 1), new TGeoTranslation(0., 0., kChamberZPos[ich - 1]));
    topVolume.AddNode(out, 2 * ich - 1, new TGeoTranslation(0., 0., kChamberZPos[ich - 1]));

  } // end of the chamber loop
}

//______________________________________________________________________________
std::vector<TGeoVolume*> getStation1SensitiveVolumes()
{
  /// Create a vector containing the sensitive volume name of the quadrants for the Detector class

  std::vector<TGeoVolume*> sensitiveVolumeNames;
  for (int i = 1; i <= 2; i++) {

    auto vol = gGeoManager->GetVolume(Form("SA%dG", i));

    if (!vol) {
      throw std::runtime_error(Form("could not get expected volume SA%dG", i));
    } else {
      sensitiveVolumeNames.push_back(vol);
    }
  }
  return sensitiveVolumeNames;
}

} // namespace mch
} // namespace o2
