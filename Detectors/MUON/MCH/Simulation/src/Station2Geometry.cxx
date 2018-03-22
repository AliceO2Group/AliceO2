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

/// TODO :
/// * finish the createQuadrant function
/// * see if the quadrant placements can be improved

#include "Materials.h"
#include "Station2Geometry.h"

#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoShape.h>
#include <TGeoPgon.h>
#include <TGeoVolume.h>
#include <TMath.h>

#include <iostream>
#include <array>

namespace o2
{
namespace mch
{

/// Constants

// Chamber z positions (from AliMUONConstants)
const float kChamberZpos[2] = { -676.4, -695.4 };

// Quadrant z position w.r.t the chamber center
const float kQuadZpos = 6.8 / 2;

// Thickness
const float kAnodeCathodeGap = 0.25;       // 2.5 mm => gap between anode & cathode plane
const float kEffCopperWidth = 0.005;       // Eff. cu in cathode pcb (zCu)
const float kCathodeWidth = 0.04;          // Cathode pcb (zCbb)
const float kRohaWidth = 2.5;              // Rohacell (zRoha)
const float kMechExitBoardWidth = 0.04;    // Mech. exit board //0.08 (zMeb)
const float kEffElecReadBoardWidth = 0.04; // Effective electronic readout board //0.02 (zEeb)

// Segments
const float kHalfSegment0Length = 95.5 / 2;
const float kHalfSegment0Height = 1.2 / 2;

const float kHalfSegment2Length = 1. / 2;
const float kHalfSegment2Height = 95.5 / 2;

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
  auto gas = assertMedium(Medium::Gas);
  auto copper = assertMedium(Medium::Copper);
  auto FR4 = assertMedium(Medium::FR4);
  auto rohacell = assertMedium(Medium::Rohacell);
  assertMedium(Medium::Epoxy);

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
  seg0->AddNode(gGeoManager->MakeBox("Segment 0 gas", gas, length, height, width), 1, new TGeoTranslation(x, y, z));
  z += width;

  // copper sheet
  width = kEffCopperWidth / 2.;
  z += width;
  auto copper0 = gGeoManager->MakeBox("SCU0L", copper, length, height, width);
  seg0->AddNode(copper0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(copper0, 2, new TGeoTranslation(x, y, -z));

  // cathode
  width = kCathodeWidth / 2.;
  z += width;
  auto cathode0 = gGeoManager->MakeBox("SCB0L", FR4, length, height, width);
  seg0->AddNode(cathode0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(cathode0, 2, new TGeoTranslation(x, y, -z));

  // rohacell
  width = kRohaWidth / 2.;
  z += width;
  auto rohacell0 = gGeoManager->MakeBox("SRH0L", rohacell, length, height, width);
  seg0->AddNode(rohacell0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(rohacell0, 2, new TGeoTranslation(x, y, -z));

  // mechanical exit board
  width = kMechExitBoardWidth / 2.;
  z += width;
  auto mech0 = gGeoManager->MakeBox("SMB0L", FR4, length, height, width);
  seg0->AddNode(mech0, 1, new TGeoTranslation(x, y, z));
  seg0->AddNode(mech0, 2, new TGeoTranslation(x, y, -z));

  // effective electronic exit board
  width = kEffElecReadBoardWidth / 2.;
  z += width;
  auto elec0 = gGeoManager->MakeBox("SEB0L", copper, length, height, width);
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
  seg1->AddNode(new TGeoVolume("Segment 1 gas", new TGeoPgon(par), gas), 1, new TGeoTranslation(x, y, z));
  z += width;

  // copper sheet
  width = kEffCopperWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto copper1 = new TGeoVolume("SCU1L", new TGeoPgon(par), copper);
  seg1->AddNode(copper1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(copper1, 2, new TGeoTranslation(x, y, -z));

  // cathode
  width = kCathodeWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto cathode1 = new TGeoVolume("SCB1L", new TGeoPgon(par), FR4);
  seg1->AddNode(cathode1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(cathode1, 2, new TGeoTranslation(x, y, -z));

  // rohacell
  width = kRohaWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto rohacell1 = new TGeoVolume("SRH1L", new TGeoPgon(par), rohacell);
  seg1->AddNode(rohacell1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(rohacell1, 2, new TGeoTranslation(x, y, -z));

  // mechanical exit board
  width = kMechExitBoardWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto mech1 = new TGeoVolume("SMB1L", new TGeoPgon(par), FR4);
  seg1->AddNode(mech1, 1, new TGeoTranslation(x, y, z));
  seg1->AddNode(mech1, 2, new TGeoTranslation(x, y, -z));

  // effective electronic exit board
  width = kEffElecReadBoardWidth / 2.;
  par[4] = -width; // z-position of the first plane
  par[7] = width;  // z-position of the second plane
  z += width;
  auto elec1 = new TGeoVolume("SEB1L", new TGeoPgon(par), copper);
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
  seg2->AddNode(gGeoManager->MakeBox("Segment 2 gas", gas, length, height, width), 1, new TGeoTranslation(x, y, z));
  z += width;

  // copper sheet
  width = kEffCopperWidth / 2.;
  z += width;
  auto copper2 = gGeoManager->MakeBox("SCU2L", copper, length, height, width);
  seg2->AddNode(copper2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(copper2, 2, new TGeoTranslation(x, y, -z));

  // cathode
  width = kCathodeWidth / 2.;
  z += width;
  auto cathode2 = gGeoManager->MakeBox("SCB2L", FR4, length, height, width);
  seg2->AddNode(cathode2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(cathode2, 2, new TGeoTranslation(x, y, -z));

  // rohacell
  width = kRohaWidth / 2.;
  z += width;
  auto rohacell2 = gGeoManager->MakeBox("SRH2L", rohacell, length, height, width);
  seg2->AddNode(rohacell2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(rohacell2, 2, new TGeoTranslation(x, y, -z));

  // mechanical exit board
  width = kMechExitBoardWidth / 2.;
  z += width;
  auto mech2 = gGeoManager->MakeBox("SMB2L", FR4, length, height, width);
  seg2->AddNode(mech2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(mech2, 2, new TGeoTranslation(x, y, -z));

  // effective electronic exit board
  width = kEffElecReadBoardWidth / 2.;
  z += width;
  auto elec2 = gGeoManager->MakeBox("SEB2L", copper, length, height, width);
  seg2->AddNode(elec2, 1, new TGeoTranslation(x, y, z));
  seg2->AddNode(elec2, 2, new TGeoTranslation(x, y, -z));

  // place the segment 2 in the quadrant
  quadrant->AddNode(seg2, 2);

  /* TODO
  /// Frames

  //Frame-1

  float frame1[3];
  frame1[0] = 101.0 / 2.; //100.6 = 94.5 + 2.5 + 3.6
  frame1[1] = 2.5 / 2.;
  frame1[2] = 5.0 / 2.;

  TVirtualMC::GetMC()->Gsvolu("SFRM1", "BOX", idPGF30, frame1, 3); //Frame - 1 // fill with pkk GF30

  float arib1[3];
  arib1[0] = frame1[0];
  arib1[1] = 0.9 / 2.;
  arib1[2] = (frame1[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRA1", "BOX", idAir, arib1, 3); // fill with air

  float xposarib1 = 0;
  float yposarib1 = -frame1[1] + arib1[1];
  float zposarib1 = frame1[2] - arib1[2];

  TVirtualMC::GetMC()->Gspos("SFRA1", 1, "SFRM1", xposarib1, yposarib1, zposarib1, 0, "ONLY");  //replace pkk GF30 with air(b)
  TVirtualMC::GetMC()->Gspos("SFRA1", 2, "SFRM1", xposarib1, yposarib1, -zposarib1, 0, "ONLY"); //replace pkk GF30 with air(nb)

  float rrib1[3];
  rrib1[0] = frame1[0];
  rrib1[1] = 0.6 / 2.;
  rrib1[2] = (frame1[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRR1", "BOX", idRoha, rrib1, 3); // fill with rohacell

  float xposrrib1 = 0.0;
  float yposrrib1 = frame1[1] - rrib1[1];
  float zposrrib1 = frame1[2] - rrib1[2];

  TVirtualMC::GetMC()->Gspos("SFRR1", 1, "SFRM1", xposrrib1, yposrrib1, zposrrib1, 0, "ONLY");  //replace pkk GF30 with rohacell
  TVirtualMC::GetMC()->Gspos("SFRR1", 2, "SFRM1", xposrrib1, yposrrib1, -zposrrib1, 0, "ONLY"); //replace pkk GF30 with rohacell

  float xposFr1 = frame1[0] + 20.6;
  float yposFr1 = -3.7 + frame1[1];
  float zposFr1 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRM1", 1, "SQM3", xposFr1, yposFr1, zposFr1, 0, "ONLY"); // frame -1
  TVirtualMC::GetMC()->Gspos("SFRM1", 2, "SQM4", xposFr1, yposFr1, zposFr1, 0, "ONLY"); // frame -1

  //......................................................................................
  //Frame-2

  float frame2[3];
  frame2[0] = 4.0 / 2.;
  frame2[1] = 1.2 / 2.;
  frame2[2] = 5.0 / 2;

  TVirtualMC::GetMC()->Gsvolu("SFRM2", "BOX", idPGF30, frame2, 3); //Frame - 2

  float rrib2[3];
  rrib2[0] = frame2[0] - 1.0 / 2.0;
  rrib2[1] = frame2[1];
  rrib2[2] = (frame2[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRR2", "BOX", idRoha, rrib2, 3);

  float xposrrib2 = -1.0 / 2.0;
  float yposrrib2 = 0.0;
  float zposrrib2 = frame2[2] - rrib2[2];

  TVirtualMC::GetMC()->Gspos("SFRR2", 1, "SFRM2", xposrrib2, yposrrib2, zposrrib2, 0, "ONLY");  //replace pkk GF30 with rohacell
  TVirtualMC::GetMC()->Gspos("SFRR2", 2, "SFRM2", xposrrib2, yposrrib2, -zposrrib2, 0, "ONLY"); //replace pkk GF30 with roha

  float xposFr2 = frame2[0] + kOuterRadius;
  float yposFr2 = -frame2[1];
  float zposFr2 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRM2", 1, "SQM3", xposFr2, yposFr2, zposFr2, 0, "MANY"); //global positing of frame in SQM3
  TVirtualMC::GetMC()->Gspos("SFRM2", 2, "SQM4", xposFr2, yposFr2, zposFr2, 0, "MANY"); //global positing of frame in SQM4

  //......................................................................................

  //Frame-3

  float pgparFr3[10];
  pgparFr3[0] = kStartAngle;
  pgparFr3[1] = kStopAngle;
  pgparFr3[2] = kNEdges;
  pgparFr3[3] = kNPlanes;
  pgparFr3[4] = 0.;
  pgparFr3[5] = kOuterRadius;
  pgparFr3[6] = 121.6;
  pgparFr3[7] = pgparFr3[4] + 5.0;
  pgparFr3[8] = pgparFr3[5];
  pgparFr3[9] = pgparFr3[6];

  TVirtualMC::GetMC()->Gsvolu("SFRM3", "PGON", idPGF30, pgparFr3, 10);

  float pgparRrib3[10];
  pgparRrib3[0] = kStartAngle;
  pgparRrib3[1] = kStopAngle;
  pgparRrib3[2] = kNEdges;
  pgparRrib3[3] = kNPlanes;
  pgparRrib3[4] = 0.;
  pgparRrib3[5] = kOuterRadius;
  pgparRrib3[6] = 120.6;
  pgparRrib3[7] = pgparRrib3[4] + 1.55;
  pgparRrib3[8] = pgparRrib3[5];
  pgparRrib3[9] = pgparRrib3[6];

  TVirtualMC::GetMC()->Gsvolu("SFRR3", "PGON", idRoha, pgparRrib3, 10);

  float xposrrib3 = 0.0;
  float yposrrib3 = 0.0;
  float zposrrib3 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRR3", 1, "SFRM3", xposrrib3, yposrrib3, zposrrib3, 0, "ONLY");

  zposrrib3 = 3.45;

  TVirtualMC::GetMC()->Gspos("SFRR3", 2, "SFRM3", xposrrib3, yposrrib3, zposrrib3, 0, "ONLY");

  float xposFr3 = 0.0;
  float yposFr3 = 0.0;
  float zposFr3 = -frame1[2];

  TVirtualMC::GetMC()->Gspos("SFRM3", 1, "SQM3", xposFr3, yposFr3, zposFr3, 0, "ONLY"); // frame -1
  TVirtualMC::GetMC()->Gspos("SFRM3", 2, "SQM4", xposFr3, yposFr3, zposFr3, 0, "ONLY"); // frame -1

  //......................................................................................
  //Frame-4

  float frame4[3];
  frame4[0] = 1.0 / 2.;
  frame4[1] = 4.0 / 2.;
  frame4[2] = frame1[2];

  TVirtualMC::GetMC()->Gsvolu("SFRM4", "BOX", idPGF30, frame4, 3);

  float rrib4[3];
  rrib4[0] = frame4[0];
  rrib4[1] = frame4[1] - 1.0 / 2;
  rrib4[2] = (frame4[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRR4", "BOX", idRoha, rrib4, 3);

  float xposrrib4 = 0.0;
  float yposrrib4 = -1.0 / 2;
  float zposrrib4 = frame4[2] - rrib4[2];

  TVirtualMC::GetMC()->Gspos("SFRR4", 1, "SFRM4", xposrrib4, yposrrib4, zposrrib4, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRR4", 2, "SFRM4", xposrrib4, yposrrib4, -zposrrib4, 0, "ONLY");

  float xposFr4 = -frame4[0];
  float yposFr4 = -frame4[1] + kOuterRadius;
  float zposFr4 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRM4", 1, "SQM3", xposFr4, yposFr4, zposFr4, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("SFRM4", 2, "SQM4", xposFr4, yposFr4, zposFr4, 0, "MANY");

  //......................................................................................
  //Frame-5
  float frame5[3];
  frame5[0] = 2.7 / 2.;
  frame5[1] = 101.0 / 2.;
  frame5[2] = 5.0 / 2.;

  TVirtualMC::GetMC()->Gsvolu("SFRM5", "BOX", idPGF30, frame5, 3); //Frame - 1

  float arib5[3];
  arib5[0] = 0.9 / 2.0;
  arib5[1] = frame5[1];
  arib5[2] = (frame5[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRA5", "BOX", idAir, arib5, 3);

  float xposarib5 = -frame5[0] + arib5[0];
  float yposarib5 = 0.0;
  float zposarib5 = frame5[2] - arib5[2];

  TVirtualMC::GetMC()->Gspos("SFRA5", 1, "SFRM5", xposarib5, yposarib5, zposarib5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRA5", 2, "SFRM5", xposarib5, yposarib5, -zposarib5, 0, "ONLY");

  float rrib5[3];
  rrib5[0] = 0.8 / 2.0;
  rrib5[1] = frame5[1];
  rrib5[2] = (frame5[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRR5", "BOX", idRoha, rrib5, 3);

  float xposrrib5 = frame5[0] - rrib5[0];
  float yposrrib5 = 0.0;
  float zposrrib5 = frame5[2] - rrib5[2];

  TVirtualMC::GetMC()->Gspos("SFRR5", 1, "SFRM5", xposrrib5, yposrrib5, zposrrib5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRR5", 2, "SFRM5", xposrrib5, yposrrib5, -zposrrib5, 0, "ONLY");

  float xposFr5 = -3.7 + frame5[0];
  float yposFr5 = frame5[1] + 20.6;
  float zposFr5 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRM5", 1, "SQM3", xposFr5, yposFr5, zposFr5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRM5", 2, "SQM4", xposFr5, yposFr5, zposFr5, 0, "ONLY");

  //......................................................................................
  //Frame -6

  float frame6[3];
  frame6[0] = 1.0 / 2.;
  frame6[1] = 2.5 / 2.;
  frame6[2] = frame1[2];

  TVirtualMC::GetMC()->Gsvolu("SFRM6", "BOX", idPGF30, frame6, 3);

  float rrib6[3];
  rrib6[0] = frame6[0];
  rrib6[1] = 1.5 / 2.;
  rrib6[2] = (frame2[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRR6", "BOX", idRoha, rrib6, 3);

  float xposrrib6 = 0.0;
  float yposrrib6 = 1.0 / 2.0;
  float zposrrib6 = frame6[2] - rrib6[2];

  TVirtualMC::GetMC()->Gspos("SFRR6", 1, "SFRM6", xposrrib6, yposrrib6, zposrrib6, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRR6", 2, "SFRM6", xposrrib6, yposrrib6, -zposrrib6, 0, "ONLY");

  float xposFr6 = -frame6[0];
  float yposFr6 = frame6[1] + 20.6;
  float zposFr6 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRM6", 1, "SQM3", xposFr6, yposFr6, zposFr6, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRM6", 2, "SQM4", xposFr6, yposFr6, zposFr6, 0, "ONLY");

  //......................................................................................
  //Frame - 7 inner pgon

  float pgparFr7[10];
  pgparFr7[0] = kStartAngle;
  pgparFr7[1] = kStopAngle;
  pgparFr7[2] = kNEdges;
  pgparFr7[3] = kNPlanes;
  pgparFr7[4] = 0.;
  pgparFr7[5] = 20.6;
  pgparFr7[6] = kInnerRadius;
  pgparFr7[7] = pgparFr7[4] + 5.0;
  pgparFr7[8] = pgparFr7[5];
  pgparFr7[9] = pgparFr7[6];

  TVirtualMC::GetMC()->Gsvolu("SFRM7", "PGON", idPGF30, pgparFr7, 10);

  float pgparRrib7[10];
  pgparRrib7[0] = kStartAngle;
  pgparRrib7[1] = kStopAngle;
  pgparRrib7[2] = kNEdges;
  pgparRrib7[3] = kNPlanes;
  pgparRrib7[4] = 0.;
  pgparRrib7[5] = 21.6;
  pgparRrib7[6] = kInnerRadius;
  pgparRrib7[7] = pgparRrib7[4] + 1.55;
  pgparRrib7[8] = pgparRrib7[5];
  pgparRrib7[9] = pgparRrib7[6];

  TVirtualMC::GetMC()->Gsvolu("SFRR7", "PGON", idRoha, pgparRrib7, 10);

  float xposrrib7 = 0.0;
  float yposrrib7 = 0.0;
  float zposrrib7 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRR7", 1, "SFRM7", xposrrib7, yposrrib7, zposrrib7, 0, "ONLY");

  zposrrib7 = 3.45;

  TVirtualMC::GetMC()->Gspos("SFRR7", 2, "SFRM7", xposrrib7, yposrrib7, zposrrib7, 0, "ONLY");

  float xposFr7 = 0.0;
  float yposFr7 = 0.0;
  float zposFr7 = -frame1[2];

  TVirtualMC::GetMC()->Gspos("SFRM7", 1, "SQM3", xposFr7, yposFr7, zposFr7, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRM7", 2, "SQM4", xposFr7, yposFr7, zposFr7, 0, "ONLY");

  //......................................................................................
  //Frame - 8

  float frame8[3];
  frame8[0] = 2.5 / 2.0;
  frame8[1] = 1.2 / 2.0;
  frame8[2] = frame1[2];

  TVirtualMC::GetMC()->Gsvolu("SFRM8", "BOX", idPGF30, frame8, 3); //Frame - 2

  float rrib8[3];
  rrib8[0] = frame8[0] - 1.0 / 2;
  rrib8[1] = frame8[1];
  rrib8[2] = (frame8[2] - 0.95) / 2.0;

  TVirtualMC::GetMC()->Gsvolu("SFRR8", "BOX", idRoha, rrib8, 3);

  float xposrrib8 = -1.0 / 2;
  float yposrrib8 = 0.0;
  float zposrrib8 = frame8[2] - rrib8[2];

  TVirtualMC::GetMC()->Gspos("SFRR8", 1, "SFRM8", xposrrib8, yposrrib8, zposrrib8, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRR8", 2, "SFRM8", xposrrib8, yposrrib8, -zposrrib8, 0, "ONLY");

  float xposFr8 = frame8[0] + 20.6;
  float yposFr8 = -frame8[1];
  float zposFr8 = 0.0;

  TVirtualMC::GetMC()->Gspos("SFRM8", 1, "SQM3", xposFr8, yposFr8, zposFr8, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("SFRM8", 2, "SQM4", xposFr8, yposFr8, zposFr8, 0, "ONLY");
  */

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
  std::array<TGeoRotation*, 4> rot = { rot0, rot1, rot2, rot3 };

  int detElemId[4];
  detElemId[0] = 1; // quadrant I
  detElemId[1] = 0; // quadrant II
  detElemId[2] = 3; // quadrant III
  detElemId[3] = 2; // quadrant IV

  // Build the two chambers
  float posz = 0.;

  for (int ich = 3; ich < 5; ich++) {

    // create two half-chambers (new compared to AliRoot !)
    auto in = new TGeoVolumeAssembly(Form("SC0%dI", ich));
    auto out = new TGeoVolumeAssembly(Form("SC0%dO", ich));

    // Place the quadrant in the half-chambers
    for (int i = 0; i < 4; i++) {
      posz = kQuadZpos * TMath::Power(-1, detElemId[i]);

      if (detElemId[i] == 1 || detElemId[i] == 2) {
        in->AddNode(quadrant, detElemId[i] + ich * 100, new TGeoCombiTrans(0., 0., posz, rot[i]));
      } else
        out->AddNode(quadrant, detElemId[i] + ich * 100, new TGeoCombiTrans(0., 0., posz, rot[i]));
    }

    // place the half-chambers in the top volume
    topVolume.AddNode(in, 2 * (ich - 1), new TGeoTranslation(0., 0., kChamberZpos[ich - 3]));
    topVolume.AddNode(out, 2 * ich - 1, new TGeoTranslation(0., 0., kChamberZpos[ich - 3]));

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
