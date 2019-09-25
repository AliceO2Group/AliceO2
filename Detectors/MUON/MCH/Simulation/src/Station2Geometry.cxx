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
/// \brief  Implementation of the station 2 geometry
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

/// Constants

// chamber z position
const double kChamberZPos[2] = {-676.4, -695.4};

// quadrant z position w.r.t the chamber center
const double kQuadrantZPos = 3.9;

// thickness
const double kGasHalfThickness = 0.25;
const double kCathodeHalfThickness = 0.005 / 2; // effective copper thickness in the PCB (to be checked !)
const double kPCBHalfThickness = 0.04 / 2;
const double kInsuHalfThickness = kPCBHalfThickness - kCathodeHalfThickness;
const double kRohaHalfThickness = 2.5 / 2;
const double kMEBHalfThickness = 0.04 / 2;
const double kEERBHalfThickness = 0.04 / 2; // effective electronic readout board (seems to be a very big value, should be the same than for station 1, to be investigated)

// polygon shape parameters
const double kStartAngle = 0.;
const double kStopAngle = 90.;
const int kNPlanes = 2;
const int kNEdges = 5;

/// Segments
const int kNSegments = 3;

const double kSegmentRadius[2] = {23.1, 117.6};

// box segment dimensions
const double kBoxSegHalfLength[kNSegments] = {95.5 / 2, 0., 0.5};
const double kBoxSegHalfHeight[kNSegments] = {0.6, 0., 95.5 / 2};

/// Frames
const int kNFrames = 8;
const double kFrameHalfLength[kNFrames] = {101. / 2, 2., 0., 0.5, 2.7 / 2, 0.5, 0., 2.5 / 2};
const double kFrameHalfHeight[kNFrames] = {2.5 / 2, 0.6, 0., 2., 101. / 2, 2.5 / 2, 0., 0.6};

const double kRibHalfLength[kNFrames] = {
  kFrameHalfLength[0],
  kFrameHalfLength[1] - 0.5,
  0.,
  kFrameHalfLength[3],
  0.8 / 2,
  kFrameHalfLength[5],
  0.,
  kFrameHalfLength[7] - 0.5};
const double kRibHalfHeight[kNFrames] = {
  0.3,
  kFrameHalfHeight[1],
  0.,
  kFrameHalfHeight[3] - 0.5,
  kFrameHalfHeight[4],
  1.5 / 2,
  0.,
  kFrameHalfHeight[7]};

const double kFrame3Radius[2] = {kSegmentRadius[1], 121.6};
const double kRib3Radius[2] = {kSegmentRadius[1], 120.6};

const double kFrame7Radius[2] = {20.6, kSegmentRadius[0]};
const double kRib7Radius[2] = {21.6, kSegmentRadius[0]};

const double kEpoxyHalfThickness = 2.;
const double kRibHalfThickness = 0.5;

//______________________________________________________________________________
TGeoVolume* createSegment(int i)
{

  /// Function creating a segment for the quadrant volume
  /// A segment is a pile-up of layers defining the detection area:
  /// gas + PCB (cathode + insulator) + rohacell + mech. exit board + eff. electronic exit board

  const char* segmentName = Form("Segment %d", i);
  auto segment = new TGeoVolumeAssembly(segmentName);

  const int kNLayers = 6;

  const std::string kLayerName[kNLayers] = {"gas", "cathode", "insulator", "rohacell", "MEB", "EERB"};
  const std::array<TGeoMedium*, kNLayers> kLayerMedium = {assertMedium(Medium::Gas), assertMedium(Medium::Copper), assertMedium(Medium::FR4), assertMedium(Medium::Rohacell), assertMedium(Medium::Epoxy), assertMedium(Medium::Copper)};
  const double kLayerHalfThickness[kNLayers] = {kGasHalfThickness, kCathodeHalfThickness, kInsuHalfThickness, kRohaHalfThickness, kMEBHalfThickness, kEERBHalfThickness};

  // volume dimensions
  double halfLength = kBoxSegHalfLength[i], halfHeight = kBoxSegHalfHeight[i], halfThickness = kLayerHalfThickness[0];

  double z = 0.; // increment this variable when adding a new layer

  switch (i) {
    case 1: // polygon

      // parameters
      double par[10];
      par[0] = kStartAngle;       // initial angle
      par[1] = kStopAngle;        // increment in angle starting from initial angle
      par[2] = kNEdges;           // number of edges
      par[3] = kNPlanes;          // number of planes
      par[4] = -halfThickness;    // z-position of the first plane
      par[5] = kSegmentRadius[0]; // inner radius first plane
      par[6] = kSegmentRadius[1]; // outer radius first plane
      par[7] = halfThickness;     // z-position of the second plane
      par[8] = par[5];            // inner radius of second plane
      par[9] = par[6];            // outer radius of second plane

      // create and place the layers in the segment

      segment->AddNode(new TGeoVolume(Form("%s %s", segmentName, kLayerName[0].data()), new TGeoPgon(par), kLayerMedium[0]), 1);

      z = halfThickness;

      for (int j = 1; j < kNLayers; j++) {

        halfThickness = kLayerHalfThickness[j];

        par[4] = -halfThickness;
        par[7] = halfThickness;
        auto layer = new TGeoVolume(Form("Segment %d %s", i, kLayerName[j].data()), new TGeoPgon(par), kLayerMedium[j]);

        z += halfThickness;
        segment->AddNode(layer, 1, new TGeoTranslation(0., 0., z));
        segment->AddNode(layer, 2, new TGeoTranslation(0., 0., -z));
        z += halfThickness;

      } // end of the layer loop
      break;

    default: //box
      // create and place the layers in the segment

      // start with gas
      halfThickness = kLayerHalfThickness[0];

      segment->AddNode(gGeoManager->MakeBox(Form("%s %s", segmentName, kLayerName[0].data()), kLayerMedium[0], halfLength, halfHeight, halfThickness), 1);

      z = halfThickness;

      for (int j = 1; j < kNLayers; j++) {

        halfThickness = kLayerHalfThickness[j];

        auto layer = gGeoManager->MakeBox(Form("%s %s", segmentName, kLayerName[j].data()), kLayerMedium[j], halfLength, halfHeight, halfThickness);

        z += halfThickness;
        segment->AddNode(layer, 1, new TGeoTranslation(0., 0., z));
        segment->AddNode(layer, 2, new TGeoTranslation(0., 0., -z));
        z += halfThickness;

      } // end of the layer loop

      break;
  } // end of the switch

  return segment;
}

//______________________________________________________________________________
void createFrames()
{

  // materials
  const auto kFrameMed = assertMedium(Medium::Epoxy);  // to be changed ? PEEK GF-30 in CERN-THESIS-2008-170
  const auto kRibMed = assertMedium(Medium::Rohacell); // to be changed ? PEEK GF-30 in CERN-THESIS-2008-170

  // rib position on the frame
  const double kRibXPos[kNFrames] = {
    0.,
    -kFrameHalfLength[1] + kRibHalfLength[1], // left edge
    0.,
    0.,
    kFrameHalfLength[4] - kRibHalfLength[4], // right edge
    0.,
    0.,
    kFrameHalfLength[7] - kRibHalfLength[7] // right edge
  };

  const double kRibYPos[kNFrames] = {
    kFrameHalfHeight[0] - kRibHalfHeight[0], // upper edge
    0.,
    0.,
    -kFrameHalfHeight[3] + kRibHalfHeight[3], // lower edge
    0.,
    kFrameHalfHeight[5] - kRibHalfHeight[5], // upper edge
    0.,
    0.};

  // useful variables
  double halfThickness = 0., z = 0.;

  for (int i = 1; i <= kNFrames; i++) {

    // in this loop, we only create box frames
    if (i == 3 || i == 7)
      continue; // go to next frame

    // create the frame
    auto frame = new TGeoVolumeAssembly(Form("Frame %d", i));

    // create the epoxy
    halfThickness = kEpoxyHalfThickness;
    frame->AddNode(gGeoManager->MakeBox(Form("Epoxy %d", i), kFrameMed, kFrameHalfLength[i - 1], kFrameHalfHeight[i - 1], halfThickness), 1);
    z = halfThickness;

    // create the rib
    halfThickness = kRibHalfThickness;
    auto rib = gGeoManager->MakeBox(Form("Rib %d", i), kRibMed, kRibHalfLength[i - 1], kRibHalfHeight[i - 1], halfThickness);

    z += halfThickness;
    frame->AddNode(rib, 1, new TGeoTranslation(kRibXPos[i - 1], kRibYPos[i - 1], z));
    frame->AddNode(rib, 2, new TGeoTranslation(kRibXPos[i - 1], kRibYPos[i - 1], -z));
  }

  /// Polygon shape frames

  // parameters
  double par[10];
  par[0] = kStartAngle;
  par[1] = kStopAngle;
  par[2] = kNEdges;
  par[3] = kNPlanes;

  // Frame 3
  auto frame3 = new TGeoVolumeAssembly("Frame 3");

  // epoxy layer
  halfThickness = kEpoxyHalfThickness;
  z = 0.;

  par[4] = -halfThickness;
  par[5] = kFrame3Radius[0];
  par[6] = kFrame3Radius[1];
  par[7] = halfThickness;
  par[8] = par[5];
  par[9] = par[6];

  frame3->AddNode(new TGeoVolume("Epoxy 3", new TGeoPgon(par), kFrameMed), 1);
  z += halfThickness;

  // rib
  halfThickness = kRibHalfThickness;
  par[4] = -halfThickness;
  par[5] = kRib3Radius[0];
  par[6] = kRib3Radius[1];
  par[7] = halfThickness;
  par[8] = par[5];
  par[9] = par[6];

  auto rib3 = new TGeoVolume("Rib 3", new TGeoPgon(par), kRibMed);

  z += halfThickness;
  frame3->AddNode(rib3, 1, new TGeoTranslation(0., 0., z));
  frame3->AddNode(rib3, 2, new TGeoTranslation(0., 0., -z));

  // Frame 7
  auto frame7 = new TGeoVolumeAssembly("Frame 7");

  // epoxy layer
  halfThickness = kEpoxyHalfThickness;
  z = 0.;

  par[4] = -halfThickness;
  par[5] = kFrame7Radius[0];
  par[6] = kFrame7Radius[1];
  par[7] = halfThickness;
  par[8] = par[5];
  par[9] = par[6];
  frame3->AddNode(new TGeoVolume("Epoxy 7", new TGeoPgon(par), kFrameMed), 1);
  z += halfThickness;

  // rib
  halfThickness = kRibHalfThickness;
  par[4] = -halfThickness;
  par[5] = kRib7Radius[0];
  par[6] = kRib7Radius[1];
  par[7] = halfThickness;
  par[8] = par[5];
  par[9] = par[6];

  auto rib7 = new TGeoVolume("Rib 7", new TGeoPgon(par), kRibMed);

  z += halfThickness;
  frame7->AddNode(rib7, 1, new TGeoTranslation(0., 0., z));
  frame7->AddNode(rib7, 2, new TGeoTranslation(0., 0., -z));
}

//______________________________________________________________________________
TGeoVolume* createQuadrant()
{

  /// Create a quadrant, a volume assembly containing all the different elements, identical for each chamber
  auto quadrant = new TGeoVolumeAssembly("Station 2 quadrant");

  // create and place the segments in the quadrant
  const double kSegXPos[kNSegments] = {kSegmentRadius[0] + kBoxSegHalfLength[0], 0., -kBoxSegHalfLength[2]};
  const double kSegYPos[kNSegments] = {-kBoxSegHalfHeight[0], 0., kSegmentRadius[0] + kBoxSegHalfHeight[2]};

  for (int i = 0; i < kNSegments; i++)
    quadrant->AddNode(createSegment(i), 0, new TGeoTranslation(kSegXPos[i], kSegYPos[i], 0.));

  // create and place the frames in the quadrant
  createFrames();

  // positions
  const double kFrameXPos[kNFrames] = {
    kSegXPos[0],                                                  // frame n°1 aligned with the segment 0
    kSegXPos[0] + kBoxSegHalfLength[0] + kFrameHalfLength[2 - 1], // frame n°2 at the right of the segment 0
    0.,
    kSegXPos[2],                                                  // frame n°4 aligned with the segment 2
    kSegXPos[2] - kBoxSegHalfLength[2] - kFrameHalfLength[5 - 1], // frame n°5 at the left of the segment 2
    kSegXPos[2],                                                  // frame n°6 aligned with the segment 2
    0.,
    kSegXPos[0] - kBoxSegHalfLength[0] - kFrameHalfLength[8 - 1], // frame n°8 at the left of the segment 0
  };

  const double kFrameYPos[kNFrames] = {
    kSegYPos[0] - kBoxSegHalfHeight[0] - kFrameHalfHeight[1 - 1], // frame n°1 below the segment 0
    kSegYPos[0],                                                  // frame n°2 aligned with the segment 0
    0.,
    kSegYPos[2] + kBoxSegHalfHeight[2] + kFrameHalfHeight[4 - 1], // frame n°4 above the segment 2
    kSegYPos[2],                                                  // frame n°5 aligned with the segment 2
    kSegYPos[2] - kBoxSegHalfHeight[2] - kFrameHalfHeight[6 - 1], // frame n°6 below the segment 2
    0.,
    kSegYPos[0], // frame n°8 aligned with the segment 0
  };

  for (int i = 1; i <= kNFrames; i++)
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
  std::array<TGeoRotation*, kNQuad> rot = {rot0, rot1, rot2, rot3};

  // build the two chambers
  double z = kQuadrantZPos;
  int detElemID = 0;
  const int kFirstChamberNumber = 3;

  for (int ich = kFirstChamberNumber; ich < kFirstChamberNumber + 2; ich++) {

    // create two half-chambers
    auto in = new TGeoVolumeAssembly(Form("SC0%dI", ich));
    auto out = new TGeoVolumeAssembly(Form("SC0%dO", ich));

    // place the quadrants in the half-chambers
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
    topVolume.AddNode(in, 2 * (ich - 1), new TGeoTranslation(0., 0., kChamberZPos[ich - kFirstChamberNumber]));
    topVolume.AddNode(out, 2 * ich - 1, new TGeoTranslation(0., 0., kChamberZPos[ich - kFirstChamberNumber]));

  } // end of the chamber loop
}

//______________________________________________________________________________
std::vector<TGeoVolume*> getStation2SensitiveVolumes()
{
  /// Create a vector containing the sensitive volume's name of the quadrants for the Detector class

  std::vector<TGeoVolume*> sensitiveVolumeNames;
  for (int i = 0; i < kNSegments; i++) {

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
