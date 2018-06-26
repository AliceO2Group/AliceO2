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

#include "Materials.h"
#include "Station2Geometry.h"

#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoShape.h>
#include <TGeoPgon.h>
#include <TGeoVolume.h>

#include <iostream>
#include <string>
#include <array>

namespace o2
{
namespace mch
{

/// Constants (to be checked !!!!)

// Chamber z positions (from AliMUONConstants)
const double kChamberZpos[2] = { -676.4, -695.4 };

// Quadrant z position w.r.t the chamber center
const double kQuadZpos = 6.8 / 2.;

// Thickness
const double kGasWidth = 2 * 0.25;
const double kCathodeWidth = 0.005;                  // Effective copper width in the PCB (to be checked !)
const double kPCBWidth = 0.04;                       // Total width of the PCB (checked)
const double kInsuWidth = kPCBWidth - kCathodeWidth; // PCB insulator width, computed like this to ensure the total PCB width
const double kRohaWidth = 2.5;                       // Rohacell (checked)
const double kMEBWidth = 0.04;                       // Mechanical exit board (checked)
const double kEERBWidth = 0.04;                      // Effective electronic readout board (seems to be a very big value, should be the same than for station 1, to be investigated)

// Polygon shape parameters
const double kStartAngle = 0.;
const double kStopAngle = 90.;
const int kNofPlanes = 2;
const int kNofEdges = 5;
const double kFirstInnerRadius = 20.6;
const double kSecondInnerRadius = 23.1;
const double kOuterRadius = 117.6;
const double kSecondOuterRadius = 120.6;

/// Segments
const int kNofSeg = 3;
const double kSeg0HalfLength = 94.5 / 2.;
const double kSeg0HalfHeight = 1.2 / 2.;

const double kSeg2HalfLength = 1 / 2.;
const double kSeg2HalfHeight = 95.5 / 2.;

const double kSegHalfLength[kNofSeg] = { kSeg0HalfLength, 0., kSeg2HalfLength };
const double kSegHalfHeight[kNofSeg] = { kSeg0HalfHeight, 0., kSeg2HalfHeight };

/// Frames
const int kNofFrames = 8;
const double kFrameLength[kNofFrames] = {
  101.,
  4.,
  0.,
  1.,
  2.7,
  1.,
  0.,
  2.5
};
const double kFrameHeight[kNofFrames] = {
  2.5,
  1.2,
  0.,
  4.,
  101.,
  2.5,
  0.,
  1.2
};

const double kRibLength[kNofFrames] = {
  kFrameLength[0],
  kFrameLength[1] - 0.5,
  0.,
  kFrameLength[3],
  0.8,
  kFrameLength[5],
  0.,
  kFrameLength[7] - 0.5
};
const double kRibHeight[kNofFrames] = {
  0.6,
  0.6,
  0.,
  kFrameHeight[3] - 0.5,
  kFrameHeight[4],
  1.5,
  0.,
  kFrameHeight[7]
};

const double kFrame3OuterRadius = 121.6;
const double kRib3OuterRadius = 120.6;
const double kRib7InnerRadius = 21.6;

const double kFrameWidth = 5.;  // checked
const double kGlueWidth = 0.02; // new ! this value is temporary (= to the one in Station345Geometry)
const double kRibWidth = 1.;    // changed w.r.t AliMUONSt2GeometryBuilderv2

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

  const std::string kLayerName[kNofLayers] = { "gas", "cathode", "insulator", "rohacell", "MEB", "EERB" };
  const std::array<const TGeoMedium*, kNofLayers> kLayerMedium = { kGasMed, kCathodeMed, kInsuMed, kRohacellMed, kMEBMed, kEERBMed };
  const double kLayerWidth[kNofLayers] = { kGasWidth, kCathodeWidth, kInsuWidth, kRohaWidth, kMEBWidth, kEERBWidth };

  auto segment = new TGeoVolumeAssembly(Form("Segment %d", iSeg));

  // volume dimensions
  double halfLength = kSegHalfLength[iSeg];
  double halfHeight = kSegHalfHeight[iSeg];
  double halfWidth = 0.001; //random value just to initialize the variable

  double z = 0.; // increment this variable when adding a new layer

  switch (iSeg) {
    case 1: // polygon

      // parameters
      double par[10];
      par[0] = kStartAngle;        // initial angle
      par[1] = kStopAngle;         // increment in angle starting from initial angle
      par[2] = kNofEdges;          // number of sides
      par[3] = kNofPlanes;         // number of planes
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
}

//______________________________________________________________________________
void createFrames()
{

  // materials
  const auto kFrameMed = assertMedium(Medium::Epoxy); // to be changed ? PEEK GF-30 in CERN-THESIS-2008-170
  const auto kGlueMed = assertMedium(Medium::Glue);
  const auto kRibMed = assertMedium(Medium::Rohacell); // to be changed ? PEEK GF-30 in CERN-THESIS-2008-170

  // useful variables
  double halfLength = 0., halfHeight = 0., halfWidth = 0., x = 0., y = 0., z = 0.;

  // position of the rib w.r.t the frame
  const double kRibXPos[kNofFrames] = { 0., -0.5, 0., 0., (kFrameLength[4] - kRibLength[4]) / 2., 0., 0., -0.5 };
  const double kRibYPos[kNofFrames] = { (kFrameHeight[0] - kRibHeight[0]) / 2., 0., 0., -0.5, 0., 0.5, 0., 0. };

  for (int i = 1; i <= kNofFrames; i++) {

    // in this loop, we only create box frames
    if (i == 3 || i == 7)
      continue; // go to next frame

    // create the frame
    halfWidth = kFrameWidth / 2.;
    auto frame = gGeoManager->MakeBox(Form("Frame %d", i), kFrameMed, kFrameLength[i - 1] / 2., kFrameHeight[i - 1] / 2., halfWidth);
    z = halfWidth;

    halfLength = kRibLength[i - 1] / 2.;
    halfHeight = kRibHeight[i - 1] / 2.;

    // create a glue layer between the frame and the rib (new !)
    halfWidth = kGlueWidth / 2.;
    auto glue = gGeoManager->MakeBox(Form("Glue %d", i), kGlueMed, halfLength, halfHeight, halfWidth);

    x = kRibXPos[i - 1];
    y = kRibYPos[i - 1];

    z += halfWidth;
    frame->AddNode(glue, 1, new TGeoTranslation(x, y, z));
    frame->AddNode(glue, 2, new TGeoTranslation(x, y, -z));
    z += halfWidth;

    // create the rib
    halfWidth = kRibWidth / 2.;
    auto rib = gGeoManager->MakeBox(Form("Rib %d", i), kRibMed, halfLength, halfHeight, halfWidth);

    z += halfWidth;
    frame->AddNode(rib, 1, new TGeoTranslation(x, y, z));
    frame->AddNode(rib, 2, new TGeoTranslation(x, y, -z));
  }

  /// Polygon shape frames

  // parameters

  double par[10];
  par[0] = kStartAngle;
  par[1] = kStopAngle;
  par[2] = kNofEdges;
  par[3] = kNofPlanes;

  // Frame 3

  // frame layer
  halfWidth = kFrameWidth / 2.;
  z = 0.;

  par[4] = -halfWidth;
  par[5] = kOuterRadius;
  par[6] = kFrame3OuterRadius;
  par[7] = halfWidth;
  par[8] = par[5];
  par[9] = par[6];
  auto frame3 = new TGeoVolume("Frame 3", new TGeoPgon(par), kFrameMed);

  z = halfWidth;

  // glue layer
  halfWidth = kGlueWidth / 2.;
  par[4] = -halfWidth;
  par[6] = kRib3OuterRadius;
  par[7] = halfWidth;
  par[9] = par[6];

  auto glue3 = new TGeoVolume("Glue 3", new TGeoPgon(par), kGlueMed);

  z += halfWidth;
  frame3->AddNode(glue3, 1, new TGeoTranslation(0., 0., z));
  frame3->AddNode(glue3, 2, new TGeoTranslation(0., 0., -z));
  z += halfWidth;

  // rib
  halfWidth = kRibWidth / 2.;
  par[4] = -halfWidth;
  par[7] = halfWidth;

  auto rib3 = new TGeoVolume("Rib 3", new TGeoPgon(par), kRibMed);

  z += halfWidth;
  frame3->AddNode(rib3, 1, new TGeoTranslation(0., 0., z));
  frame3->AddNode(rib3, 2, new TGeoTranslation(0., 0., -z));

  // Frame 7

  // frame layer
  halfWidth = kFrameWidth / 2.;
  z = 0.;

  par[4] = -halfWidth;
  par[5] = kFirstInnerRadius;
  par[6] = kSecondInnerRadius;
  par[7] = halfWidth;
  par[8] = par[5];
  par[9] = par[6];
  auto frame7 = new TGeoVolume("Frame 7", new TGeoPgon(par), kFrameMed);

  z = halfWidth;

  // glue layer
  halfWidth = kGlueWidth / 2.;
  par[4] = -halfWidth;
  par[5] = kRib7InnerRadius;
  par[7] = halfWidth;
  par[8] = par[5];

  auto glue7 = new TGeoVolume("Glue 7", new TGeoPgon(par), kGlueMed);

  z += halfWidth;
  frame7->AddNode(glue7, 1, new TGeoTranslation(0., 0., z));
  frame7->AddNode(glue7, 2, new TGeoTranslation(0., 0., -z));
  z += halfWidth;

  // rib
  halfWidth = kRibWidth / 2.;
  par[4] = -halfWidth;
  par[7] = halfWidth;

  auto rib7 = new TGeoVolume("Rib 7", new TGeoPgon(par), kRibMed);

  z += halfWidth;
  frame7->AddNode(rib7, 1, new TGeoTranslation(0., 0., z));
  frame7->AddNode(rib7, 2, new TGeoTranslation(0., 0., -z));
}

//______________________________________________________________________________
TGeoVolume* createQuadrant()
{

  /// Create a quadrant, a volume assembly containing all the different elements, identical for each chamber
  auto quadrant = new TGeoVolumeAssembly("Station 2 quadrant");

  /// Create and place the segments in the quadrant volume
  const double kSegXPos[kNofSeg] = { kSegHalfLength[0] + kSecondInnerRadius, 0., -kSegHalfLength[2] };
  const double kSegYPos[kNofSeg] = { -kSegHalfHeight[0], 0., kSegHalfHeight[2] + kSecondInnerRadius };

  for (int i = 0; i < kNofSeg; i++)
    quadrant->AddNode(createSegment(i), 0, new TGeoTranslation(kSegXPos[i], kSegYPos[i], 0.));

  /// Create and place the frames in the quadrant
  createFrames();

  // positions
  const double kFrameShift = 3.7;
  const double kFrameXPos[kNofFrames] = { kFrameLength[0] / 2. + kFirstInnerRadius,
                                          kFrameLength[1] / 2. + kOuterRadius,
                                          0.,
                                          -kFrameLength[3] / 2.,
                                          kFrameLength[4] / 2. - kFrameShift,
                                          -kFrameLength[5] / 2.,
                                          0.,
                                          kFrameLength[7] / 2. + kFirstInnerRadius };

  const double kFrameYPos[kNofFrames] = { kFrameHeight[0] / 2. - kFrameShift,
                                          -kFrameHeight[1] / 2.,
                                          0.,
                                          kFrameHeight[3] / 2. + kOuterRadius,
                                          kFrameHeight[4] / 2. + kFirstInnerRadius,
                                          kFrameHeight[5] / 2. + kFirstInnerRadius,
                                          0.,
                                          -kFrameHeight[7] / 2. };

  for (int i = 1; i <= kNofFrames; i++)
    quadrant->AddNode(gGeoManager->GetVolume(Form("Frame %d", i)), 1,
                      new TGeoTranslation(kFrameXPos[i - 1], kFrameYPos[i - 1], 0.));

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
  std::array<TGeoRotation*, kNQuad> rot = { rot0, rot1, rot2, rot3 };

  // Build the two chambers
  double z = kQuadZpos;
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
  for (int i = 0; i < kNofSeg; i++) {

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
