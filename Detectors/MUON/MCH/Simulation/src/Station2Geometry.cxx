// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Station2Geometry.cxx
/// \brief  Implementation of the station 2 geometry (copied and adapted from AliMUONSt2GeometryBuilderv2)
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   23 mai 2018

/// TODO : clean constants in the functions

#include "Materials.h"
#include "Station2Geometry.h"

#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoShape.h>
#include <TGeoPgon.h>
#include <TGeoVolume.h>

#include <iostream>
#include <array>

namespace o2
{
namespace mch
{

/// Constants (to be checked !!!!)

// Chamber z positions (from AliMUONConstants)
const float kChamberZpos[2] = { -676.4, -695.4 };

// Quadrant z position w.r.t the chamber center
const float kQuadZpos = 6.8 / 2.;

const int kNQuad = 4;

// Thickness
const float kAnodeCathodeGap = 0.25;       // 2.5 mm => gap between anode & cathode plane
const float kEffCopperWidth = 0.005;       // Eff. cu in cathode pcb (zCu)
const float kCathodeWidth = 0.04;          // Cathode pcb (zCbb)
const float kRohaWidth = 2.5;              // Rohacell (zRoha)
const float kMechExitBoardWidth = 0.04;    // Mech. exit board //0.08 (zMeb)
const float kEffElecReadBoardWidth = 0.04; // Effective electronic readout board //0.02 (zEeb)

// Segments
const float kHalfSegment0Length = 95.5 / 2.;
const float kHalfSegment0Height = 1.2 / 2.;

const float kHalfSegment2Length = 1 / 2.;
const float kHalfSegment2Height = 95.5 / 2.;

// Frames
const float kHalfFrameWidth = 5 / 2.;
const float kFrameZShift = 0.95 / 2.;
const float kHalfRibWidth = kHalfFrameWidth / 2. + kFrameZShift;

// Polygon shape parameters
const float kStartAngle = 0.;
const float kStopAngle = 90.;
const int kNPlanes = 2;
const int kNEdges = 5;
const float kInnerRadius = 23.1;
const float kOuterRadius = 117.6;

//______________________________________________________________________________
TGeoVolume* createQuadrant()
{

  // create a quadrant, a volume assembly containing all the different elements, identical for each chamber
  auto quadrant = new TGeoVolumeAssembly("Station 2 quadrant");

  // useful variables
  float length, height, width; // box volume parameters
  float x, y, z;               // volume positions
  double par[10];              // for polygon volumes

  // materials
  const auto kGasMed = assertMedium(Medium::Gas);
  const auto kCopperMed = assertMedium(Medium::Copper);
  const auto kFR4Med = assertMedium(Medium::FR4);
  const auto kRohacellMed = assertMedium(Medium::Rohacell);
  const auto kEpoxyMed = assertMedium(Medium::Epoxy);
  const auto kAirMed = assertMedium(Medium::Air);

  // create and place the different layers in the quadrant

  /// Plane = Gas + Cathode PCB + Copper sheet + Rohacell + mech exit board + eff. electronic exit board

  // Segment 0 - Horizontal box

  auto seg0 = new TGeoVolumeAssembly("Segment 0");

  length = kHalfSegment0Length;
  height = kHalfSegment0Height;

  x = length + kInnerRadius;
  y = -height;
  z = 0.;

  // gas
  width = kAnodeCathodeGap;
  seg0->AddNode(gGeoManager->MakeBox("Segment 0 gas", kGasMed, length, height, width), 1,
                new TGeoTranslation(x, y, z));
  z += width;

  // copper sheet
  width = kEffCopperWidth / 2.;
  z += width;
  auto copper0 = gGeoManager->MakeBox("SCU0L", kCopperMed, length, height, width);
  seg0->AddNode(copper0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(copper0, 2, new TGeoTranslation(x, y, -z));

  // cathode
  width = kCathodeWidth / 2.;
  z += width;
  auto cathode0 = gGeoManager->MakeBox("SCB0L", kFR4Med, length, height, width);
  seg0->AddNode(cathode0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(cathode0, 2, new TGeoTranslation(x, y, -z));

  // rohacell
  width = kRohaWidth / 2.;
  z += width;
  auto rohacell0 = gGeoManager->MakeBox("SRH0L", kRohacellMed, length, height, width);
  seg0->AddNode(rohacell0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(rohacell0, 2, new TGeoTranslation(x, y, -z));

  // mechanical exit board
  width = kMechExitBoardWidth / 2.;
  z += width;
  auto mech0 = gGeoManager->MakeBox("SMB0L", kFR4Med, length, height, width);
  seg0->AddNode(mech0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(mech0, 2, new TGeoTranslation(x, y, -z));

  // effective electronic exit board
  width = kEffElecReadBoardWidth / 2.;
  z += width;
  auto elec0 = gGeoManager->MakeBox("SEB0L", kCopperMed, length, height, width);
  seg0->AddNode(elec0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(elec0, 2, new TGeoTranslation(x, y, -z));

  // place the segment 0 in the quadrant
  quadrant->AddNode(seg0, 0);

  // Segment 1 - Polygon

  auto seg1 = new TGeoVolumeAssembly("Segment 1");

  x = 0.;
  y = 0.;
  z = 0.;

  // create the shape
  par[0] = kStartAngle; // initial angle
  par[1] = kStopAngle;  // increment in angle starting from initial angle
  par[2] = kNEdges;     // number of side
  par[3] = kNPlanes;    // number of plane

  par[5] = kInnerRadius; // inner radius first plane
  par[6] = kOuterRadius; // outer radious first plane

  par[8] = par[5]; // inner radius of second plane
  par[9] = par[6]; // outer radious of second plane

  // gas
  width = kAnodeCathodeGap;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  seg1->AddNode(new TGeoVolume("Segment 1 gas", new TGeoPgon(par), kGasMed), 1,
                new TGeoTranslation(x, y, z));
  z += width;

  // copper sheet
  width = kEffCopperWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto copper1 = new TGeoVolume("SCU1L", new TGeoPgon(par), kCopperMed);
  seg1->AddNode(copper1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(copper1, 2, new TGeoTranslation(x, y, -z));

  // cathode
  width = kCathodeWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto cathode1 = new TGeoVolume("SCB1L", new TGeoPgon(par), kFR4Med);
  seg1->AddNode(cathode1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(cathode1, 2, new TGeoTranslation(x, y, -z));

  // rohacell
  width = kRohaWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto rohacell1 = new TGeoVolume("SRH1L", new TGeoPgon(par), kRohacellMed);
  seg1->AddNode(rohacell1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(rohacell1, 2, new TGeoTranslation(x, y, -z));

  // mechanical exit board
  width = kMechExitBoardWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto mech1 = new TGeoVolume("SMB1L", new TGeoPgon(par), kFR4Med);
  seg1->AddNode(mech1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(mech1, 2, new TGeoTranslation(x, y, -z));

  // effective electronic exit board
  width = kEffElecReadBoardWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto elec1 = new TGeoVolume("SEB1L", new TGeoPgon(par), kCopperMed);
  seg1->AddNode(elec1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(elec1, 2, new TGeoTranslation(x, y, -z));

  // place the segment 1 volume in the quadrant
  quadrant->AddNode(seg1, 1);

  // Segment 2 - Vertical box
  auto seg2 = new TGeoVolumeAssembly("Segment 2");

  length = kHalfSegment2Length;
  height = kHalfSegment2Height;

  x = -length;
  y = height + kInnerRadius;
  z = 0.;

  // gas
  width = kAnodeCathodeGap;
  seg2->AddNode(gGeoManager->MakeBox("Segment 2 gas", kGasMed, length, height, width), 1,
                new TGeoTranslation(x, y, z));
  z += width;

  // copper sheet
  width = kEffCopperWidth / 2.;
  z += width;
  auto copper2 = gGeoManager->MakeBox("SCU2L", kCopperMed, length, height, width);
  seg2->AddNode(copper2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(copper2, 2, new TGeoTranslation(x, y, -z));

  // cathode
  width = kCathodeWidth / 2.;
  z += width;
  auto cathode2 = gGeoManager->MakeBox("SCB2L", kFR4Med, length, height, width);
  seg2->AddNode(cathode2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(cathode2, 2, new TGeoTranslation(x, y, -z));

  // rohacell
  width = kRohaWidth / 2.;
  z += width;
  auto rohacell2 = gGeoManager->MakeBox("SRH2L", kRohacellMed, length, height, width);
  seg2->AddNode(rohacell2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(rohacell2, 2, new TGeoTranslation(x, y, -z));

  // mechanical exit board
  width = kMechExitBoardWidth / 2.;
  z += width;
  auto mech2 = gGeoManager->MakeBox("SMB2L", kFR4Med, length, height, width);
  seg2->AddNode(mech2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(mech2, 2, new TGeoTranslation(x, y, -z));

  // effective electronic exit board
  width = kEffElecReadBoardWidth / 2.;
  z += width;
  auto elec2 = gGeoManager->MakeBox("SEB2L", kCopperMed, length, height, width);
  seg2->AddNode(elec2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(elec2, 2, new TGeoTranslation(x, y, -z));

  // place the segment 2 in the quadrant
  quadrant->AddNode(seg2, 2);

  /// Frames

  // parameters
  float aribLength, aribHeight, rribLength, rribHeight;

  // Frame 1

  length = 101 / 2.;
  height = 2.5 / 2.;
  auto frame1 = gGeoManager->MakeBox("SFRM1", kEpoxyMed, length, height, kHalfFrameWidth);

  aribLength = length;
  aribHeight = 0.9 / 2.;
  auto arib1 = gGeoManager->MakeBox("SFRA1", kAirMed, aribLength, aribHeight, kHalfRibWidth);

  x = 0.;
  y = -height + aribHeight;
  z = kHalfFrameWidth - aribHeight;
  frame1->AddNode(arib1, 1, new TGeoTranslation(x, y, z));
  frame1->AddNode(arib1, 2, new TGeoTranslation(x, y, -z));

  rribLength = length;
  rribHeight = 0.6 / 2.;
  auto rrib1 = gGeoManager->MakeBox("SFRR1", kRohacellMed, rribLength, rribHeight, kHalfRibWidth);

  y = height - rribHeight;
  frame1->AddNode(rrib1, 1, new TGeoTranslation(x, y, z));
  frame1->AddNode(rrib1, 2, new TGeoTranslation(x, y, -z));

  x = length + 20.6;
  y = -3.7 + height;
  z = 0.;
  quadrant->AddNode(frame1, 1, new TGeoTranslation(x, y, z));

  // Frame 2

  length = 4 / 2.;
  height = 1.2 / 2.;
  auto frame2 = gGeoManager->MakeBox("SFRM2", kEpoxyMed, length, height, kHalfFrameWidth);

  rribLength = length - 1 / 2.;
  rribHeight = 0.6 / 2.;
  auto rrib2 = gGeoManager->MakeBox("SFRR2", kRohacellMed, rribLength, rribHeight, kHalfRibWidth);

  x = -1 / 2.;
  y = 0.;
  z = kHalfFrameWidth - kHalfRibWidth;

  frame2->AddNode(rrib2, 1, new TGeoTranslation(x, y, z));
  frame2->AddNode(rrib2, 2, new TGeoTranslation(x, y, -z));

  x = length + kOuterRadius;
  y = -height;
  z = 0.;
  quadrant->AddNode(frame2, 1, new TGeoTranslation(x, y, z));

  // Frame 3

  par[4] = 0.;
  par[5] = kOuterRadius;
  par[6] = 121.6;
  par[7] = par[4] + 5.;
  par[8] = par[5];
  par[9] = par[6];
  auto frame3 = new TGeoVolume("SFRM3", new TGeoPgon(par), kEpoxyMed);

  par[6] = 120.6;
  par[7] = par[4] + 1.55;
  par[8] = par[5];
  par[9] = par[6];
  auto rrib3 = new TGeoVolume("SFRR3", new TGeoPgon(par), kRohacellMed);

  x = 0.;
  y = 0.;
  z = 0.;
  frame3->AddNode(rrib3, 1, new TGeoTranslation(x, y, z));

  z = 3.45;
  frame3->AddNode(rrib3, 2, new TGeoTranslation(x, y, z));

  z = -kHalfFrameWidth;
  quadrant->AddNode(frame3, 1, new TGeoTranslation(x, y, z));

  // Frame 4

  length = 1 / 2.;
  height = 4 / 2.;
  auto frame4 = gGeoManager->MakeBox("SFRM4", kEpoxyMed, length, height, kHalfFrameWidth);

  rribLength = length;
  rribHeight = height - 1 / 2.;
  auto rrib4 = gGeoManager->MakeBox("SFRR4", kRohacellMed, rribLength, rribHeight, kHalfRibWidth);

  x = 0.;
  y = -1 / 2.;
  z = kHalfFrameWidth - kHalfRibWidth;
  frame4->AddNode(rrib4, 1, new TGeoTranslation(x, y, z));
  frame4->AddNode(rrib4, 2, new TGeoTranslation(x, y, -z));

  x = -length;
  y = -height + kOuterRadius;
  z = 0.;
  quadrant->AddNode(frame4, 1, new TGeoTranslation(x, y, z));

  // Frame 5

  length = 2.7 / 2.;
  height = 101 / 2.;
  auto frame5 = gGeoManager->MakeBox("SFRM5", kEpoxyMed, length, height, kHalfFrameWidth);

  aribLength = 0.9 / 2.;
  aribHeight = height;
  auto arib5 = gGeoManager->MakeBox("SFRA5", kAirMed, aribLength, aribHeight, kHalfRibWidth);

  x = -length + aribLength;
  y = 0.;
  z = kHalfFrameWidth - kHalfRibWidth;
  frame5->AddNode(arib5, 1, new TGeoTranslation(x, y, z));
  frame5->AddNode(arib5, 2, new TGeoTranslation(x, y, -z));

  rribLength = 0.8 / 2.;
  rribHeight = height;
  auto rrib5 = gGeoManager->MakeBox("SFRR5", kRohacellMed, rribLength, rribHeight, kHalfRibWidth);

  x = length - rribLength;
  y = 0.;
  z = kHalfFrameWidth - kHalfRibWidth;
  frame5->AddNode(rrib5, 1, new TGeoTranslation(x, y, z));
  frame5->AddNode(rrib5, 2, new TGeoTranslation(x, y, -z));

  x = -3.7 + length;
  y = height + 20.6;
  z = 0.;
  quadrant->AddNode(frame5, 1, new TGeoTranslation(x, y, z));

  // Frame 6

  length = 1 / 2.;
  height = 2.5 / 2.;
  auto frame6 = gGeoManager->MakeBox("SFRM6", kEpoxyMed, length, height, kHalfFrameWidth);

  rribLength = length;
  rribHeight = 1.5 / 2.;
  auto rrib6 = gGeoManager->MakeBox("SFRR6", kRohacellMed, rribLength, rribHeight, kHalfRibWidth);

  x = 0.;
  y = 1 / 2.;
  z = kHalfFrameWidth - kHalfRibWidth;
  frame6->AddNode(rrib6, 1, new TGeoTranslation(x, y, z));
  frame6->AddNode(rrib6, 2, new TGeoTranslation(x, y, -z));

  x = -length;
  y = height + 20.6;
  z = 0.;
  quadrant->AddNode(frame6, 1, new TGeoTranslation(x, y, z));

  // Frame 7

  par[5] = 20.6;
  par[6] = kInnerRadius;
  par[7] = par[4] + 5.0;
  par[8] = par[5];
  par[9] = par[6];
  auto frame7 = new TGeoVolume("SFRM7", new TGeoPgon(par), kEpoxyMed);

  par[5] = 21.6;
  par[7] = par[4] + 1.55;
  auto rrib7 = new TGeoVolume("SFRR7", new TGeoPgon(par), kRohacellMed);

  x = 0.;
  y = 0.;
  z = 0.;
  frame7->AddNode(rrib7, 1, new TGeoTranslation(x, y, z));

  z = 3.45;
  frame7->AddNode(rrib7, 2, new TGeoTranslation(x, y, z));

  z = -kHalfFrameWidth;
  quadrant->AddNode(frame7, 1, new TGeoTranslation(x, y, z));

  //Frame - 8

  length = 2.5 / 2.;
  height = 1.2 / 2.;
  auto frame8 = gGeoManager->MakeBox("SFRM8", kEpoxyMed, length, height, kHalfFrameWidth);

  rribLength = length - 1 / 2.;
  rribHeight = height;
  auto rrib8 = gGeoManager->MakeBox("SFRR8", kRohacellMed, rribLength, rribHeight, kHalfRibWidth);

  x = -1 / 2.;
  y = 0.;
  z = kHalfFrameWidth - kHalfRibWidth;
  frame8->AddNode(rrib8, 1, new TGeoTranslation(x, y, z));
  frame8->AddNode(rrib8, 2, new TGeoTranslation(x, y, -z));

  x = length + 20.6;
  y = -height;
  z = 0.;
  quadrant->AddNode(frame6, 1, new TGeoTranslation(x, y, z));

  return quadrant;
}

//______________________________________________________________________________
void createStation2Geometry(TGeoVolume& topVolume)
{
  /// Create the geometry of the station 2

  // create a quadrant
  auto quadrant = createQuadrant();

  // rotation matrices to place the quadrants in the half-chambers
  auto rot0 = new TGeoRotation();
  auto rot1 = new TGeoRotation("reflXZ", 90., 180., 90., 90., 180., 0.);
  auto rot2 = new TGeoRotation("reflXY", 90., 180., 90., 270., 0., 0.);
  auto rot3 = new TGeoRotation("reflYZ", 90., 0., 90., -90., 180., 0.);
  std::array<TGeoRotation*, kNQuad> rot = { rot0, rot1, rot2, rot3 };

  // Build the two chambers
  float z = kQuadZpos;
  int detElemID = 0;
  const int kFirstChamberNumber = 3;

  for (int ich = kFirstChamberNumber; ich < kFirstChamberNumber + 2; ich++) {

    // create two half-chambers (new compared to AliRoot !)
    auto in = new TGeoVolumeAssembly(Form("SC0%dI", ich));
    auto out = new TGeoVolumeAssembly(Form("SC0%dO", ich));

    // Place the quadrant in the half-chambers
    for (int i = 0; i < kNQuad; i++) {
      // alternate the z position
      z *= -1.;

      // compute the detection element ID
      detElemID = 100 * ich + i;

      if (i == 0 || i == 3) {
        in->AddNode(quadrant, detElemID, new TGeoCombiTrans(0., 0., z, rot[i]));
      } else
        out->AddNode(quadrant, detElemID, new TGeoCombiTrans(0., 0., z, rot[i]));
    }

    // place the half-chambers in the top volume
    topVolume.AddNode(in, 2 * (ich - 1), new TGeoTranslation(0., 0., kChamberZpos[ich - kFirstChamberNumber]));
    topVolume.AddNode(out, 2 * ich - 1, new TGeoTranslation(0., 0., kChamberZpos[ich - kFirstChamberNumber]));

  } // end of the chamber loop
}

//______________________________________________________________________________
std::vector<TGeoVolume*> getStation2SensitiveVolumes()
{
  /// Create a vector containing the sensitive volume's name of the quadrants for the Detector class

  std::vector<TGeoVolume*> sensitiveVolumeNames;
  const int nSegment = 3;
  for (int i = 0; i < nSegment; i++) {

    auto vol = gGeoManager->GetVolume(Form("Segment %d gas", i));

    if (!vol) {
      throw std::runtime_error(Form("could not get expected volume : Segment %d gas", i));
    } else {
      sensitiveVolumeNames.push_back(vol);
    }
  }

  return sensitiveVolumeNames;
}

} // namespace mch
} // namespace o2
