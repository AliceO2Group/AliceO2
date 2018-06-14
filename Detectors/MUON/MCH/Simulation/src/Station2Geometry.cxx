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
/// \date   23 may 2018

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

using namespace std;

namespace o2
{
namespace mch
{

/// Constants (to be checked !!!!)

// Chamber z positions (from AliMUONConstants)
const float kChamberZpos[2] = { -676.4, -695.4 };

// Quadrant z position w.r.t the chamber center
const float kQuadZpos = 6.8 / 2.;

// Thickness
const float kGasWidth = 2 * 0.25;
const float kCathodeWidth = 0.005;                  // Effective copper width in the PCB (to be checked !)
const float kPCBWidth = 0.04;                       // Total width of the PCB (checked)
const float kInsuWidth = kPCBWidth - kCathodeWidth; // PCB insulator width, computed like this to ensure the total PCB width
const float kRohaWidth = 2.5;                       // Rohacell (checked)
const float kMEBWidth = 0.04;                       // Mechanical exit board (checked)
const float kEERBWidth = 0.04;                      // Effective electronic readout board (seems to be a very big value, should be the same than for station 1, to be investigated)

// Polygon shape parameters
const float kStartAngle = 0.;
const float kStopAngle = 90.;
const int kNPlanes = 2;
const int kNEdges = 5;
const float kFirstInnerRadius = 20.6;
const float kSecondInnerRadius = 23.1;
const float kOuterRadius = 117.6;

/// Segments
const int kNofSeg = 3;
const float kSeg0HalfLength = 94.5 / 2.;
const float kSeg0HalfHeight = 1.2 / 2.;

const float kSeg2HalfLength = 1 / 2.;
const float kSeg2HalfHeight = 95.5 / 2.;

const float kSegHalfLength[kNofSeg] = { kSeg0HalfLength, 0., kSeg2HalfLength };
const float kSegHalfHeight[kNofSeg] = { kSeg0HalfHeight, 0., kSeg2HalfHeight };

/// Frames
const float kFrameWidth = 5.;
const float kFrameZShift = 0.95 / 2.;
const float kHalfRibWidth = kFrameWidth / 2. + kFrameZShift;

//______________________________________________________________________________
TGeoVolume* createSegment(int iSeg)
{

  /// Function creating a segment for the quadrant volume
  /// A segment is a pile-up of layers defining the detection area :
  /// gas + PCB (cathode + insulator) + rohacell + mech exit board + eff. electronic exit board

  // materials
  const auto kGasMed = assertMedium(Medium::Gas);
  const auto kCathodeMed = assertMedium(Medium::Copper);
  const auto kInsuMed = assertMedium(Medium::FR4);
  const auto kRohacellMed = assertMedium(Medium::Rohacell);
  const auto kMEBMed = assertMedium(Medium::Epoxy); // changed w.r.t AliMUONSt2GeometryBuilderv2 after investigation
  const auto kEERBMed = assertMedium(Medium::Copper);

  const int kNofLayers = 6;

  const string kLayerName[kNofLayers] = { "gas", "cathode", "insulator", "rohacell", "MEB", "EERB" };
  const array<const TGeoMedium*, kNofLayers> kLayerMedium = { kGasMed, kCathodeMed, kInsuMed, kRohacellMed, kMEBMed, kEERBMed };
  const float kLayerWidth[kNofLayers] = { kGasWidth, kCathodeWidth, kInsuWidth, kRohaWidth, kMEBWidth, kEERBWidth };

  auto segment = new TGeoVolumeAssembly(Form("Segment %d", iSeg));

  // volume dimensions
  float halfLength = kSegHalfLength[iSeg];
  float halfHeight = kSegHalfHeight[iSeg];
  float halfWidth = 0.001; //random value just to initialize the variable

  float z = 0.; // increment this variable when adding a new layer

  switch (iSeg) {
    case 1: // polygon

      // parameters
      double par[10];
      par[0] = kStartAngle;        // initial angle
      par[1] = kStopAngle;         // increment in angle starting from initial angle
      par[2] = kNEdges;            // number of sides
      par[3] = kNPlanes;           // number of planes
      par[4] = -halfWidth;         // z-position of the first plane
      par[5] = kSecondInnerRadius; // inner radius first plane
      par[6] = kOuterRadius;       // outer radious first plane
      par[7] = halfWidth;          // z-position of the second plane
      par[8] = par[5];             // inner radius of second plane
      par[9] = par[6];             // outer radious of second plane

      // create and place the layers in the segment

      // start with gas
      halfWidth = kLayerWidth[0] / 2.;

      par[4] = -halfWidth;
      par[7] = halfWidth;
      segment->AddNode(new TGeoVolume(Form("Segment %d %s", iSeg, kLayerName[0].data()), new TGeoPgon(par), kLayerMedium[0]), 1);

      z = halfWidth;

      for (int j = 1; j < kNofLayers; j++) {

        halfWidth = kLayerWidth[j] / 2.;

        par[4] = -halfWidth;
        par[7] = halfWidth;
        auto layer = new TGeoVolume(Form("Segment %d %s", iSeg, kLayerName[j].data()), new TGeoPgon(par), kLayerMedium[j]);

        z += halfWidth;
        segment->AddNode(layer, 1, new TGeoTranslation(0., 0., z));
        segment->AddNode(layer, 2, new TGeoTranslation(0., 0., -z));
        z += halfWidth;

      } // end of the layer loop
      break;

    default: //box
      // create and place the layers in the segment

      // start with gas
      halfWidth = kLayerWidth[0] / 2.;

      segment->AddNode(new TGeoVolume(Form("Segment %d %s", iSeg, kLayerName[0].data()),
                                      new TGeoBBox(halfLength, halfHeight, halfWidth), kLayerMedium[0]),
                       1);

      z = halfWidth;

      for (int j = 1; j < kNofLayers; j++) {

        halfWidth = kLayerWidth[j] / 2.;

        auto layer = new TGeoVolume(Form("Segment %d %s", iSeg, kLayerName[j].data()),
                                    new TGeoBBox(halfLength, halfHeight, halfWidth), kLayerMedium[j]);

        z += halfWidth;
        segment->AddNode(layer, 1, new TGeoTranslation(0., 0., z));
        segment->AddNode(layer, 2, new TGeoTranslation(0., 0., -z));
        z += halfWidth;

      } // end of the layer loop

      break;
  } // end of the switch

  return segment;
} // namespace mch

//______________________________________________________________________________
TGeoVolume* createQuadrant()
{

  // create a quadrant, a volume assembly containing all the different elements, identical for each chamber
  auto quadrant = new TGeoVolumeAssembly("Station 2 quadrant");

  // useful variables
  float length = 0., height = 0., // box volume parameters
    width = kGasWidth / 2.;       //just to initialize the variable
  float x = 0., y = 0., z = 0.;   // volume positions
  double par[10];                 // for polygon volumes
  par[0] = kStartAngle;           // initial angle
  par[1] = kStopAngle;            // increment in angle starting from initial angle
  par[2] = kNEdges;               // number of sides
  par[3] = kNPlanes;              // number of planes
  par[4] = -width;                // z-position of the first plane
  par[5] = kSecondInnerRadius;    // inner radius first plane
  par[6] = kOuterRadius;          // outer radious first plane
  par[7] = width;                 // z-position of the second plane
  par[8] = par[5];                // inner radius of second plane
  par[9] = par[6];                // outer radious of second plane

  // materials
  const auto kRibMed = assertMedium(Medium::Rohacell); // to be changed ?
  const auto kFrameMed = assertMedium(Medium::Epoxy);  // to be changed ?
  const auto kAirMed = assertMedium(Medium::Air);      // normally useless

  // create and place the segments in the quadrant volume

  const float kSegXPos[kNofSeg] = { kSegHalfLength[0] + kSecondInnerRadius, 0., -kSegHalfLength[2] };
  const float kSegYPos[kNofSeg] = { -kSegHalfHeight[0], 0., kSegHalfHeight[2] + kSecondInnerRadius };

  for (int i = 0; i < kNofSeg; i++)
    quadrant->AddNode(createSegment(i), 0, new TGeoTranslation(kSegXPos[i], kSegYPos[i], 0.));

  // parameters
  float aribLength, aribHeight, rribLength, rribHeight;

  // Frame 1

  length = 101 / 2.;
  height = 2.5 / 2.;
  auto frame1 = gGeoManager->MakeBox("SFRM1", kFrameMed, length, height, kFrameWidth / 2.);

  aribLength = length;
  aribHeight = 0.9 / 2.;
  auto arib1 = gGeoManager->MakeBox("SFRA1", kAirMed, aribLength, aribHeight, kHalfRibWidth);

  x = 0.;
  y = -height + aribHeight;
  z = kFrameWidth / 2. - aribHeight;
  frame1->AddNode(arib1, 1, new TGeoTranslation(x, y, z));
  frame1->AddNode(arib1, 2, new TGeoTranslation(x, y, -z));

  rribLength = length;
  rribHeight = 0.6 / 2.;
  auto rrib1 = gGeoManager->MakeBox("SFRR1", kRibMed, rribLength, rribHeight, kHalfRibWidth);

  y = height - rribHeight;
  frame1->AddNode(rrib1, 1, new TGeoTranslation(x, y, z));
  frame1->AddNode(rrib1, 2, new TGeoTranslation(x, y, -z));

  x = length + kFirstInnerRadius;
  y = -3.7 + height;
  z = 0.;
  quadrant->AddNode(frame1, 1, new TGeoTranslation(x, y, z));

  // Frame 2

  length = 4 / 2.;
  height = 1.2 / 2.;
  auto frame2 = gGeoManager->MakeBox("SFRM2", kFrameMed, length, height, kFrameWidth / 2.);

  rribLength = length - 1 / 2.;
  rribHeight = 0.6 / 2.;
  auto rrib2 = gGeoManager->MakeBox("SFRR2", kRibMed, rribLength, rribHeight, kHalfRibWidth);

  x = -1 / 2.;
  y = 0.;
  z = kFrameWidth / 2. - kHalfRibWidth;

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
  auto frame3 = new TGeoVolume("SFRM3", new TGeoPgon(par), kFrameMed);

  par[6] = 120.6;
  par[7] = par[4] + 1.55;
  par[8] = par[5];
  par[9] = par[6];
  auto rrib3 = new TGeoVolume("SFRR3", new TGeoPgon(par), kRibMed);

  x = 0.;
  y = 0.;
  z = 0.;
  frame3->AddNode(rrib3, 1, new TGeoTranslation(x, y, z));

  z = 3.45;
  frame3->AddNode(rrib3, 2, new TGeoTranslation(x, y, z));

  z = -kFrameWidth / 2.;
  quadrant->AddNode(frame3, 1, new TGeoTranslation(x, y, z));

  // Frame 4

  length = 1 / 2.;
  height = 4 / 2.;
  auto frame4 = gGeoManager->MakeBox("SFRM4", kFrameMed, length, height, kFrameWidth / 2.);

  rribLength = length;
  rribHeight = height - 1 / 2.;
  auto rrib4 = gGeoManager->MakeBox("SFRR4", kRibMed, rribLength, rribHeight, kHalfRibWidth);

  x = 0.;
  y = -1 / 2.;
  z = kFrameWidth / 2. - kHalfRibWidth;
  frame4->AddNode(rrib4, 1, new TGeoTranslation(x, y, z));
  frame4->AddNode(rrib4, 2, new TGeoTranslation(x, y, -z));

  x = -length;
  y = -height + kOuterRadius;
  z = 0.;
  quadrant->AddNode(frame4, 1, new TGeoTranslation(x, y, z));

  // Frame 5

  length = 2.7 / 2.;
  height = 101 / 2.;
  auto frame5 = gGeoManager->MakeBox("SFRM5", kFrameMed, length, height, kFrameWidth / 2.);

  aribLength = 0.9 / 2.;
  aribHeight = height;
  auto arib5 = gGeoManager->MakeBox("SFRA5", kAirMed, aribLength, aribHeight, kHalfRibWidth);

  x = -length + aribLength;
  y = 0.;
  z = kFrameWidth / 2. - kHalfRibWidth;
  frame5->AddNode(arib5, 1, new TGeoTranslation(x, y, z));
  frame5->AddNode(arib5, 2, new TGeoTranslation(x, y, -z));

  rribLength = 0.8 / 2.;
  rribHeight = height;
  auto rrib5 = gGeoManager->MakeBox("SFRR5", kRibMed, rribLength, rribHeight, kHalfRibWidth);

  x = length - rribLength;
  y = 0.;
  z = kFrameWidth / 2. - kHalfRibWidth;
  frame5->AddNode(rrib5, 1, new TGeoTranslation(x, y, z));
  frame5->AddNode(rrib5, 2, new TGeoTranslation(x, y, -z));

  x = -3.7 + length;
  y = height + kFirstInnerRadius;
  z = 0.;
  quadrant->AddNode(frame5, 1, new TGeoTranslation(x, y, z));

  // Frame 6

  length = 1 / 2.;
  height = 2.5 / 2.;
  auto frame6 = gGeoManager->MakeBox("SFRM6", kFrameMed, length, height, kFrameWidth / 2.);

  rribLength = length;
  rribHeight = 1.5 / 2.;
  auto rrib6 = gGeoManager->MakeBox("SFRR6", kRibMed, rribLength, rribHeight, kHalfRibWidth);

  x = 0.;
  y = 1 / 2.;
  z = kFrameWidth / 2. - kHalfRibWidth;
  frame6->AddNode(rrib6, 1, new TGeoTranslation(x, y, z));
  frame6->AddNode(rrib6, 2, new TGeoTranslation(x, y, -z));

  x = -length;
  y = height + kFirstInnerRadius;
  z = 0.;
  quadrant->AddNode(frame6, 1, new TGeoTranslation(x, y, z));

  // Frame 7

  par[5] = kFirstInnerRadius;
  par[6] = kSecondInnerRadius;
  par[7] = par[4] + 5.0;
  par[8] = par[5];
  par[9] = par[6];
  auto frame7 = new TGeoVolume("SFRM7", new TGeoPgon(par), kFrameMed);

  par[5] = 21.6;
  par[7] = par[4] + 1.55;
  auto rrib7 = new TGeoVolume("SFRR7", new TGeoPgon(par), kRibMed);

  x = 0.;
  y = 0.;
  z = 0.;
  frame7->AddNode(rrib7, 1, new TGeoTranslation(x, y, z));

  z = 3.45;
  frame7->AddNode(rrib7, 2, new TGeoTranslation(x, y, z));

  z = -kFrameWidth / 2.;
  quadrant->AddNode(frame7, 1, new TGeoTranslation(x, y, z));

  //Frame - 8

  length = 2.5 / 2.;
  height = 1.2 / 2.;
  auto frame8 = gGeoManager->MakeBox("SFRM8", kFrameMed, length, height, kFrameWidth / 2.);

  rribLength = length - 1 / 2.;
  rribHeight = height;
  auto rrib8 = gGeoManager->MakeBox("SFRR8", kRibMed, rribLength, rribHeight, kHalfRibWidth);

  x = -1 / 2.;
  y = 0.;
  z = kFrameWidth / 2. - kHalfRibWidth;
  frame8->AddNode(rrib8, 1, new TGeoTranslation(x, y, z));
  frame8->AddNode(rrib8, 2, new TGeoTranslation(x, y, -z));

  x = length + kFirstInnerRadius;
  y = -height;
  z = 0.;
  quadrant->AddNode(frame8, 1, new TGeoTranslation(x, y, z));

  return quadrant;
}

//______________________________________________________________________________
void createStation2Geometry(TGeoVolume& topVolume)
{
  /// Create the geometry of the station 2

  // create a quadrant
  auto quadrant = createQuadrant();

  const int kNQuad = 4;

  // rotation matrices to place the quadrants in the half-chambers
  auto rot0 = new TGeoRotation();
  auto rot1 = new TGeoRotation("reflXZ", 90., 180., 90., 90., 180., 0.);
  auto rot2 = new TGeoRotation("reflXY", 90., 180., 90., 270., 0., 0.);
  auto rot3 = new TGeoRotation("reflYZ", 90., 0., 90., -90., 180., 0.);
  array<TGeoRotation*, kNQuad> rot = { rot0, rot1, rot2, rot3 };

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

  vector<TGeoVolume*> sensitiveVolumeNames;
  const int nSegment = 3;
  for (int i = 0; i < nSegment; i++) {

    auto vol = gGeoManager->GetVolume(Form("Segment %d gas", i));

    if (!vol) {
      throw runtime_error(Form("could not get expected volume : Segment %d gas", i));
    } else {
      sensitiveVolumeNames.push_back(vol);
    }
  }

  return sensitiveVolumeNames;
}

} // namespace mch
} // namespace o2
