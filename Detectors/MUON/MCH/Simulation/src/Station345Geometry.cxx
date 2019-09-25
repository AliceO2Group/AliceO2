// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Station345Geometry.cxx
/// \brief  Implementation of the slat-stations geometry
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   22 march 2018

#include "Materials.h"
#include "Station345Geometry.h"

#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoShape.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TMath.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <iostream>
#include <string>
#include <array>

using namespace rapidjson;
using namespace std;

namespace o2
{
namespace mch
{

///  Constants

// gas
const float kGasLength = 40.;
const float kGasHalfHeight = 40. / 2;
const float kGasHalfThickness = 0.25;

// PCB = gas + 2*(cathode + insulator)
const float kPCBLength = kGasLength;
const float kShortPCBLength = 35.;
const float kRoundedPCBLength = 42.5;
const float kR1PCBLength = 19.25;
const float kPCBHalfHeight = 58. / 2;

const float kCathodeHalfThickness = 0.002 / 2;
const float kInsuHalfThickness = 0.04 / 2;

const float kPCBHalfThickness = kGasHalfThickness + 2 * (kCathodeHalfThickness + kInsuHalfThickness);

// slat panel = honeycomb nomex + 2 carbon fiber skins
const float kSlatPanelHalfHeight = 42.5 / 2;

const float kGlueHalfThickness = 0.004 / 2;
const float kNomexBulkHalfThickness = 0.025 / 2;
const float kCarbonHalfThickness = 0.02 / 2;
const float kNomexHalfThickness = 0.8 / 2;

// spacers (noryl)
const float kSpacerHalfThickness = kGasHalfThickness;
const float kHoriSpacerHalfHeight = 1.95 / 2;
const float kVertSpacerHalfLength = 2.5 / 2;
const float kRoundedSpacerHalfLength = 1.;

// border (rohacell)
const float kBorderHalfHeight = 2.5;
const float kBorderHalfThickness = kGasHalfThickness;

// DualSampas (parameters from the PRR)
const float kDualSampaHalfLength = 3.2 / 2;
const float kDualSampaHalfHeight = 2.5;
const float kDualSampaHalfThickness = 0.027 / 2; // to be confirmed
const float kDualSampaYPos = 0.45 + kSlatPanelHalfHeight + kDualSampaHalfHeight;

/// Cables (copper)
// Low voltage (values from AliRoot)
const float kLVCableHalfHeight = 2.6 / 2;
const float kLVCableHalfThickness = 0.026 / 2;

/// Support panels
const float kCarbonSupportHalfThickness = 0.03 / 2;
const float kGlueSupportHalfThickness = 0.02 / 2;
const float kNomexSupportHalfThickness = 1.5 / 2;
const float kSt3SupportHalfHeight = 361. / 2;
const float kSt4SupportHalfHeight = 530. / 2;
const float kSt5SupportHalfHeight = 570. / 2;
const float kCh5SupportHalfLength = 162. / 2;
const float kCh6SupportHalfLength = 167. / 2;
const float kSt45SupportHalfLength = 260. / 2;

// Inner radii
const float kSt3Radius = 29.5;
const float kSt45Radius = 37.5;

// Y position of the rounded slats
const float kSt3RoundedSlatYPos = 37.8;
const float kSt45RoundedSlatYPos = 38.2;

// PCB types {name, number of DualSampa array = (nUP bending, nDOWN bending, nUP non-bending, nDOWN non-bending)}
const map<string, array<int, 4>> kPcbTypes = {{"B1N1", {10, 10, 7, 7}}, {"B2N2-", {5, 5, 4, 3}}, {"B2N2+", {5, 5, 3, 4}}, {"B3-N3", {3, 2, 2, 2}}, {"B3+N3", {2, 3, 2, 2}}, {"R1", {3, 4, 2, 3}}, {"R2", {13, 4, 9, 3}}, {"R3", {13, 1, 10, 0}}, {"S2-", {4, 5, 3, 3}}, {"S2+", {5, 4, 3, 3}}};

// Slat types
const map<string, vector<string>> kSlatTypes = {{"122000SR1", {"R1", "B1N1", "B2N2+", "S2-"}},
                                                {"112200SR2", {"R2", "B1N1", "B2N2+", "S2-"}},
                                                {"122200S", {"B1N1", "B2N2-", "B2N2-", "S2+"}},
                                                {"222000N", {"B2N2-", "B2N2-", "B2N2-"}},
                                                {"220000N", {"B2N2-", "B2N2-"}},
                                                {"122000NR1", {"R1", "B1N1", "B2N2+", "B2N2+"}},
                                                {"112200NR2", {"R2", "B1N1", "B2N2+", "B2N2+"}},
                                                {"122200N", {"B1N1", "B2N2-", "B2N2-", "B2N2-"}},
                                                {"122330N", {"B1N1", "B2N2+", "B2N2-", "B3-N3", "B3-N3"}},
                                                {"112233NR3", {"R3", "B1N1", "B2N2-", "B2N2+", "B3+N3", "B3+N3"}},
                                                {"112230N", {"B1N1", "B1N1", "B2N2-", "B2N2-", "B3-N3"}},
                                                {"222330N", {"B2N2+", "B2N2+", "B2N2-", "B3-N3", "B3-N3"}},
                                                {"223300N", {"B2N2+", "B2N2-", "B3-N3", "B3-N3"}},
                                                {"333000N", {"B3-N3", "B3-N3", "B3-N3"}},
                                                {"330000N", {"B3-N3", "B3-N3"}},
                                                {"112233N", {"B1N1", "B1N1", "B2N2+", "B2N2-", "B3-N3", "B3-N3"}},
                                                {"222333N",
                                                 {"B2N2+", "B2N2+", "B2N2+", "B3-N3", "B3-N3", "B3-N3"}},
                                                {"223330N", {"B2N2+", "B2N2+", "B3-N3", "B3-N3", "B3-N3"}},
                                                {"333300N", {"B3+N3", "B3-N3", "B3-N3", "B3-N3"}}};

extern const string jsonSlatDescription;

TGeoVolume* getDualSampa()
{
  return gGeoManager->MakeBox("DualSampa345", assertMedium(Medium::Copper), kDualSampaHalfLength, kDualSampaHalfHeight, kDualSampaHalfThickness);
}

bool isRounded(string name)
{
  return name.find('R') < name.size();
}

bool isShort(string name)
{
  return name.find('S') < name.size();
}

TGeoVolume* getRoundedVolume(const char* name, int mediumID, float halfLength, float halfHeight, float halfThickness, float xPos, float yPos, float radius)
{
  /// Function creating a volume with a rounded shape by creating a hole in a box

  // create the box
  const char* boxName = Form("%sBox", name);
  new TGeoBBox(boxName, halfLength, halfHeight, halfThickness);

  // create the position where the hole will be created
  const char* shiftName = Form("%sX%.1fY%.1fShift", name, TMath::Abs(xPos), TMath::Abs(yPos));
  auto shift = new TGeoTranslation(shiftName, xPos, yPos, 0.);
  shift->RegisterYourself();

  // create the tube that create the hole
  const char* tubeName = Form("%sR%.1fHole", name, radius);
  new TGeoTube(tubeName, 0., radius, halfThickness);

  const char* shapeName = Form("%sX%.1fY%.1fR%.1fShape", name, TMath::Abs(xPos), TMath::Abs(yPos), radius);

  // create the hole in the box and return the volume built
  return new TGeoVolume(name, new TGeoCompositeShape(shapeName, Form("%s-%s:%s", boxName, tubeName, shiftName)), assertMedium(mediumID));
}

//______________________________________________________________________________
void createCommonVolumes()
{
  /// Build the identical volumes (constant shapes, dimensions, ...) shared by many elements

  const auto kSpacerMed = assertMedium(Medium::Noryl);

  // the right vertical spacer (identical to any slat)
  gGeoManager->MakeBox("Right spacer", kSpacerMed, kVertSpacerHalfLength, kSlatPanelHalfHeight, kSpacerHalfThickness);

  // the top spacers and borders : 4 lengths possible according to the PCB shape
  for (const auto length : {kShortPCBLength, kPCBLength, kR1PCBLength, kRoundedPCBLength}) {
    // top spacer
    gGeoManager->MakeBox(Form("Top spacer %.2f long", length), kSpacerMed, length / 2, kHoriSpacerHalfHeight, kSpacerHalfThickness);

    // top border
    gGeoManager->MakeBox(Form("Top border %.2f long", length), assertMedium(Medium::Rohacell), length / 2, kBorderHalfHeight, kBorderHalfThickness);
  }
}

//______________________________________________________________________________
void createPCBs()
{
  /// Build the different PCB types

  /// A PCB is a pile-up of several material layers, from in to out : sensitive gas, cathode and insulator
  /// There are two types of cathodes : a "bending" and a "non-bending" one. We build the PCB volume such that the
  /// bending side faces the IP (z>0 in the local frame). When placing the slat on the half-chambers, the builder grabs the rotation to
  /// apply from the JSON. By doing so, we make sure that we match the mapping convention

  // Define some necessary variables
  string bendName, nonbendName;
  float y = 0., shift = 0., length = 0.; // useful parameters for dimensions and positions
  int nDualSampas = 0;

  // get the DualSampa volume
  auto* dualSampaVol = getDualSampa();

  for (const auto& [pcbName, dualSampas] : kPcbTypes) { // loop over the PCB types of the array

    auto name = (const char*)pcbName.data();

    auto pcb = new TGeoVolumeAssembly(name);

    // Reset shift variables
    float gasShift = 0., pcbShift = 0.;

    float gasLength = kGasLength;
    float pcbLength = kPCBLength;

    int numb = pcbName[1] - '0'; // char -> int conversion

    // change the variables according to the PCB shape if necessary
    switch (pcbName.front()) {
      case 'R': // rounded
        numb = pcbName.back() - '0';
        gasLength = (numb == 1) ? kR1PCBLength : kGasLength;
        pcbLength = (numb == 1) ? kR1PCBLength : kRoundedPCBLength;
        gasShift = -(gasLength - kGasLength);
        pcbShift = -(pcbLength - kPCBLength);

        bendName = Form("%sB", name);
        nonbendName = Form("%sN", name);
        break;
      case 'S': // shortened
        gasLength = kShortPCBLength;
        pcbLength = kShortPCBLength;
        bendName = Form("S2B%c", pcbName.back());
        nonbendName = Form("S2N%c", pcbName.back());
        break;
      default: // normal
        bendName = (numb == 3) ? pcbName.substr(0, 3) : pcbName.substr(0, 2);
        nonbendName = (numb == 3) ? pcbName.substr(3) : pcbName.substr(2);
    }

    float borderLength = pcbLength;

    // create the volume of each material (a box by default)
    // sensitive gas
    auto gas = gGeoManager->MakeBox(Form("%s gas", name), assertMedium(Medium::Gas), gasLength / 2., kGasHalfHeight, kGasHalfThickness);

    float x = pcbLength / 2;
    float halfHeight = kPCBHalfHeight;
    // bending cathode
    auto bend = gGeoManager->MakeBox(bendName.data(), assertMedium(Medium::Copper), x, halfHeight, kCathodeHalfThickness);

    // non-bending cathode
    auto nonbend = gGeoManager->MakeBox(nonbendName.data(), assertMedium(Medium::Copper), x, halfHeight, kCathodeHalfThickness);

    // insulating material
    auto insu = gGeoManager->MakeBox(Form("%s insulator", name), assertMedium(Medium::FR4), x, halfHeight, kInsuHalfThickness);

    // bottom spacer (noryl)
    auto spacer = gGeoManager->MakeBox(Form("%s bottom spacer", name), assertMedium(Medium::Noryl), x, kHoriSpacerHalfHeight, kSpacerHalfThickness);

    // change the volume shape if we are creating a rounded PCB
    if (isRounded(pcbName)) {
      // LHC beam pipe radius ("R3" -> it is a slat of a station 4 or 5)
      float radius = (numb == 3) ? kSt45Radius : kSt3Radius;
      // y position of the PCB center w.r.t the beam pipe shape
      switch (numb) {
        case 1:
          y = 0.; // central for "R1"
          break;
        case 2:
          y = kSt3RoundedSlatYPos; // "R2" -> station 3
          break;
        default:
          y = kSt45RoundedSlatYPos; // "R3" -> station 4 or 5
          break;
      }
      // compute the radius of curvature of the PCB we want to create
      float curvRad = radius + 2 * kRoundedSpacerHalfLength;

      x = -kRoundedPCBLength + gasLength / 2 + kVertSpacerHalfLength;
      gas = getRoundedVolume(Form("%sGas", name), Medium::Gas, gasLength / 2, kGasHalfHeight, kGasHalfThickness, x, -y, curvRad);

      x = -kRoundedPCBLength + pcbLength / 2 + kVertSpacerHalfLength;

      bend = getRoundedVolume(Form("%sBending", name), Medium::Copper, pcbLength / 2, halfHeight, kCathodeHalfThickness, x, -y, curvRad);

      nonbend = getRoundedVolume(Form("%sNonBending", name), Medium::Copper, pcbLength / 2, halfHeight, kCathodeHalfThickness, x, -y, curvRad);

      insu = getRoundedVolume(Form("%sInsulator", name), Medium::FR4, pcbLength / 2, halfHeight, kInsuHalfThickness, x, -y, curvRad);

      if (pcbName.back() != '1') { // change the bottom spacer and border shape for "R2" and "R3" PCBs

        spacer = getRoundedVolume(Form("%sBottomSpacer", name), Medium::Noryl, pcbLength / 2, kHoriSpacerHalfHeight, kSpacerHalfThickness, x, -y + kGasHalfHeight + kHoriSpacerHalfHeight, radius);

        borderLength -= (pcbName.back() == '3') ? curvRad : curvRad + kRoundedSpacerHalfLength;
      }
    }

    /// place all the layers in the PCB
    float halfThickness = kGasHalfThickness;
    pcb->AddNode(gas, 1, new TGeoTranslation(gasShift / 2, 0., 0.));
    float z = halfThickness;

    halfThickness = kCathodeHalfThickness;
    x = pcbShift / 2;
    z += halfThickness;
    pcb->AddNode(bend, 1, new TGeoTranslation(x, 0., z));
    pcb->AddNode(nonbend, 2, new TGeoTranslation(x, 0., -z));
    z += halfThickness;

    halfThickness = kInsuHalfThickness;
    z += halfThickness;
    pcb->AddNode(insu, 1, new TGeoTranslation(x, 0., z));
    pcb->AddNode(insu, 2, new TGeoTranslation(x, 0., -z));
    z += halfThickness;

    // the horizontal spacers
    y = kGasHalfHeight + kHoriSpacerHalfHeight;
    pcb->AddNode(gGeoManager->GetVolume(Form("Top spacer %.2f long", pcbLength)), 1, new TGeoTranslation(x, y, 0.));
    pcb->AddNode(spacer, 1, new TGeoTranslation(x, -y, 0.));

    // the borders
    y = kPCBHalfHeight - kBorderHalfHeight;
    pcb->AddNode(gGeoManager->GetVolume(Form("Top border %.2f long", pcbLength)), 1, new TGeoTranslation(x, y, 0.));
    x = (pcbShift + pcbLength - borderLength) / 2;
    pcb->AddNode(gGeoManager->MakeBox(Form("%s bottom border", name), assertMedium(Medium::Rohacell),
                                      borderLength / 2, kBorderHalfHeight, kBorderHalfThickness),
                 1, new TGeoTranslation(x, -y, 0.));

    // the DualSampa read-out cards
    halfThickness = kDualSampaHalfThickness;
    z += halfThickness;
    for (int i = 0; i < dualSampas.size(); i++) {

      nDualSampas = dualSampas[i];
      length = (i % 2) ? borderLength : pcbLength;
      shift = (i % 2) ? pcbLength - borderLength : 0.;
      y = TMath::Power(-1, i % 2) * kDualSampaYPos;
      z = TMath::Power(-1, i / 2) * TMath::Abs(z);

      for (int j = 0; j < nDualSampas; j++) {
        pcb->AddNode(dualSampaVol, 100 * i + j,
                     new TGeoTranslation((j - nDualSampas / 2) * (length / nDualSampas) - (nDualSampas % 2 - 1) * (length / (2 * nDualSampas)) +
                                           (pcbShift + shift) / 2,
                                         y, z));
      }
    } // end of the MANUs loop
  }   // end of the PCBs loop
}

//______________________________________________________________________________
void createSlats()
{
  /// Slat building function
  /// The different PCB types must have been built before calling this function !!!

  const auto kSpacerMed = assertMedium(Medium::Noryl);
  auto rightSpacer = gGeoManager->GetVolume("Right spacer");

  // Mirror rotation for the slat panel on the non-bending side
  auto mirror = new TGeoRotation();
  mirror->ReflectZ(true);

  for (const auto& [typeName, pcbVector] : kSlatTypes) {

    auto name = (const char*)typeName.data(); // slat name (easier to name volumes)

    // create the slat volume assembly
    auto slat = new TGeoVolumeAssembly(name);

    // Reset slat variables
    float length = 2 * 2 * kVertSpacerHalfLength; // vertical spacers
    float center = (pcbVector.size() - 1) * kGasLength / 2;
    float panelShift = 0.;
    int iVol = 0;

    // loop over the number of PCBs in the current slat
    for (const auto& pcb : pcbVector) {

      float gasLength = kGasLength;
      float pcbLength = kPCBLength;

      switch (pcb.front()) {
        case 'R':
          pcbLength = (pcb.back() == '1') ? kR1PCBLength : kRoundedPCBLength;
          panelShift -= pcbLength - kRoundedPCBLength;
          break;
        case 'S':
          pcbLength = kShortPCBLength;
          gasLength = kShortPCBLength;
          panelShift += pcbLength - kPCBLength;
          break;
      }

      length += pcbLength;

      // place the corresponding PCB volume in the slat and correct the origin of the slat
      slat->AddNode(gGeoManager->GetVolume(pcb.data()), iVol + 1,
                    new TGeoTranslation(iVol * kPCBLength - (kPCBLength - gasLength) / 2 - center, 0, 0));
      iVol++;

    } // end of the PCBs loop
    panelShift /= 2;

    // compute the LV cable length
    float cableHalfLength = (typeName.find('3') < typeName.size()) ? kSt45SupportHalfLength : kCh5SupportHalfLength;
    if (typeName == "122200N")
      cableHalfLength = kCh6SupportHalfLength;
    cableHalfLength -= length / 2;
    if (typeName == "122330N")
      cableHalfLength -= kGasLength / 2;

    float leftSpacerHalfHeight = kSlatPanelHalfHeight;
    if (isRounded(typeName)) {
      length -= 2 * kVertSpacerHalfLength;   // don't count the vertical spacer length twice in the case of rounded slat
      leftSpacerHalfHeight = kGasHalfHeight; // to avoid overlaps with the horizontal spacers
    }

    // left vertical spacer
    auto leftSpacer = gGeoManager->MakeBox(Form("%s left spacer", name), kSpacerMed, kVertSpacerHalfLength, leftSpacerHalfHeight, kSpacerHalfThickness);

    // glue a slat panel on each side of the PCBs
    auto panel = new TGeoVolumeAssembly(Form("%s panel", name));

    float x = length / 2;
    float halfHeight = kSlatPanelHalfHeight;

    // glue
    auto glue = gGeoManager->MakeBox(Form("%s panel glue", name), assertMedium(Medium::Glue), x, halfHeight, kGlueHalfThickness);

    // nomex (bulk)
    auto nomexBulk = gGeoManager->MakeBox(Form("%s panel nomex (bulk)", name), assertMedium(Medium::BulkNomex), x, halfHeight, kNomexBulkHalfThickness);

    // carbon fiber
    auto carbon = gGeoManager->MakeBox(Form("%s panel carbon fiber", name), assertMedium(Medium::Carbon), x, halfHeight, kCarbonHalfThickness);

    // nomex (honeycomb)
    auto nomex = gGeoManager->MakeBox(Form("%s panel nomex (honeycomb)", name), assertMedium(Medium::HoneyNomex), x, halfHeight, kNomexHalfThickness);

    // change the volume shape if we are creating a rounded slat
    if (isRounded(typeName)) {

      // LHC beam pipe radius ("NR3" -> it is a slat of a station 4 or 5)
      float radius = (typeName.back() == '3') ? kSt45Radius : kSt3Radius;

      // extreme angle values for the rounded spacer
      float angMin = 0., angMax = 90.;

      // position of the slat center w.r.t the beam pipe center
      x = length / 2 - kVertSpacerHalfLength;
      float y = 0.;
      float xRoundedPos = x - panelShift / 2;

      // change the LV cable length for st.3 slats
      cableHalfLength = (isShort(typeName)) ? kCh5SupportHalfLength : kCh6SupportHalfLength;

      // change the above values if necessary
      switch (typeName.back()) { // get the last character
        case '1':                // central for "S(N)R1"
          y = 0.;
          x = 2 * kPCBLength + panelShift + kVertSpacerHalfLength;
          xRoundedPos = kRoundedPCBLength + kPCBLength - kRoundedSpacerHalfLength;
          angMin =
            -TMath::RadToDeg() * TMath::ACos((kRoundedPCBLength - kR1PCBLength - kRoundedSpacerHalfLength) / radius);
          angMax = -angMin;
          break;
        case '2': // "S(N)R2" -> station 3
          y = kSt3RoundedSlatYPos;
          angMin = TMath::RadToDeg() * TMath::ASin((y - kSlatPanelHalfHeight) / (radius + kRoundedSpacerHalfLength));
          break;
        default: // "NR3" -> station 4 or 5
          y = kSt45RoundedSlatYPos;
          angMin = TMath::RadToDeg() * TMath::ASin((y - kSlatPanelHalfHeight) / (radius + kRoundedSpacerHalfLength));
          cableHalfLength = kSt45SupportHalfLength;
          break;
      }

      cableHalfLength -= (x + length / 2) / 2;

      // create and place the rounded spacer
      slat->AddNode(gGeoManager->MakeTubs(Form("%s rounded spacer", name), kSpacerMed, radius,
                                          radius + 2 * kRoundedSpacerHalfLength, kSpacerHalfThickness, angMin, angMax),
                    1, new TGeoTranslation(-xRoundedPos, -y, 0.));

      glue = getRoundedVolume(Form("%sGlue", name), Medium::Glue, length / 2, halfHeight, kGlueHalfThickness, -x, -y, radius);

      nomexBulk = getRoundedVolume(Form("%sNomexBulk", name), Medium::BulkNomex, length / 2, halfHeight, kNomexBulkHalfThickness, -x, -y, radius);

      carbon = getRoundedVolume(Form("%sCarbon", name), Medium::Carbon, length / 2, halfHeight, kCarbonHalfThickness, -x, -y, radius);

      nomex = getRoundedVolume(Form("%sNomex", name), Medium::HoneyNomex, length / 2, halfHeight, kNomexHalfThickness, -x, -y, radius);

      leftSpacer = getRoundedVolume(Form("%sLeftSpacer", name), Medium::Noryl, kVertSpacerHalfLength, leftSpacerHalfHeight, kSpacerHalfThickness, 0., -y, radius);

    } // end of the "rounded" condition

    // place all the layers in the slat panel volume assembly
    // be careful : the panel origin is on the glue edge !

    float halfThickness = kGlueHalfThickness;
    float z = halfThickness;
    panel->AddNode(glue, 1, new TGeoTranslation(0., 0., z));
    z += halfThickness;

    halfThickness = kNomexBulkHalfThickness;
    z += halfThickness;
    panel->AddNode(nomexBulk, 1, new TGeoTranslation(0., 0., z));
    z += halfThickness;

    halfThickness = kGlueHalfThickness;
    z += halfThickness;
    panel->AddNode(glue, 2, new TGeoTranslation(0., 0., z));
    z += halfThickness;

    halfThickness = kCarbonHalfThickness;
    z += halfThickness;
    panel->AddNode(carbon, 1, new TGeoTranslation(0., 0., z));
    z += halfThickness;

    halfThickness = kNomexHalfThickness;
    z += halfThickness;
    panel->AddNode(nomex, 1, new TGeoTranslation(0., 0., z));
    z += halfThickness;

    halfThickness = kCarbonHalfThickness;
    z += halfThickness;
    panel->AddNode(carbon, 2, new TGeoTranslation(0., 0., z));

    // place the panel volume on each side of the slat volume assembly
    x = panelShift;
    z = kPCBHalfThickness;
    slat->AddNode(panel, 1, new TGeoTranslation(x, 0., z));
    slat->AddNode(panel, 2, new TGeoCombiTrans(x, 0., -z, mirror));

    // place the vertical spacers
    x = length / 2 - kVertSpacerHalfLength;
    slat->AddNode(rightSpacer, 1, new TGeoTranslation(x + panelShift, 0., 0.));
    // don't place a left spacer for S(N)R1 slat
    if (typeName.back() != '1')
      slat->AddNode(leftSpacer, 1, new TGeoTranslation(-x + panelShift, 0., 0.));

    // place the LV cables (top and bottom)
    cableHalfLength += kVertSpacerHalfLength;
    auto LVcable = gGeoManager->MakeBox(Form("%s LV cable", name), assertMedium(Medium::Copper), cableHalfLength,
                                        kLVCableHalfHeight, kLVCableHalfThickness);
    x = -2 * kVertSpacerHalfLength + panelShift + cableHalfLength + length / 2;
    float y = kDualSampaYPos + kDualSampaHalfHeight + kLVCableHalfHeight;
    z = -kPCBHalfThickness - kLVCableHalfThickness;
    slat->AddNode(LVcable, 1, new TGeoTranslation(x, y, z));
    slat->AddNode(LVcable, 2, new TGeoTranslation(x, -y, z));

  } // end of the slat loop
}

//______________________________________________________________________________
void createSupportPanels()
{
  /// Function building the half-chamber support panels (one different per chamber)

  // dimensions
  float halfLength = 0., halfHeight = 0.;

  for (int i = 5; i <= 10; i++) {

    // define the support panel volume
    auto support = new TGeoVolumeAssembly(Form("Chamber %d support panel", i));

    if (i <= 6) { // station 3 half-chambers
      halfHeight = kSt3SupportHalfHeight;
      halfLength = (i == 5) ? kCh5SupportHalfLength : kCh6SupportHalfLength;
    } else { // station 4 or 5
      halfLength = kSt45SupportHalfLength;
      halfHeight = (i <= 8) ? kSt4SupportHalfHeight : kSt5SupportHalfHeight;
    }

    // LHC beam pipe radius at the given chamber z position
    float radius = (i <= 6) ? kSt3Radius : kSt45Radius;

    float x = -halfLength + kVertSpacerHalfLength;

    // create the nomex volume, change its shape by extracting the pipe shape and place it in the support panel
    float halfThickness = kNomexSupportHalfThickness;
    float z = 0.;

    support->AddNode(getRoundedVolume(Form("NomexSupportPanelCh%d", i), Medium::HoneyNomex, halfLength, halfHeight, halfThickness, x, 0., radius), i, new TGeoTranslation(halfLength, 0., z));

    z += halfThickness; // increment this value when adding a new layer

    // create the glue volume and change its shape by extracting the pipe shape
    halfThickness = kGlueSupportHalfThickness;

    auto glue = getRoundedVolume(Form("GlueSupportPanelCh%d", i), Medium::Glue, halfLength, halfHeight, halfThickness, x, 0., radius);

    // place it on each side of the nomex volume
    z += halfThickness;
    support->AddNode(glue, 1, new TGeoTranslation(halfLength, 0., z));
    support->AddNode(glue, 2, new TGeoTranslation(halfLength, 0., -z));
    z += halfThickness;

    // create the carbon volume and change its shape by extracting the pipe shape
    halfThickness = kCarbonSupportHalfThickness;

    auto carbon = getRoundedVolume(Form("CarbonSupportPanelCh%d", i), Medium::Carbon, halfLength, halfHeight, halfThickness, x, 0., radius);

    // place it on each side of the glue volume
    z += halfThickness;
    support->AddNode(carbon, 1, new TGeoTranslation(halfLength, 0., z));
    support->AddNode(carbon, 2, new TGeoTranslation(halfLength, 0., -z));

  } // end of the chamber loop
}

//______________________________________________________________________________
void buildHalfChambers(TGeoVolume& topVolume)
{
  /// Build the slat half-chambers
  /// The different slat types must have been built before calling this function !!!

  // read the json containing all the necessary parameters to place the slat volumes in the half-chambers
  StringStream is(jsonSlatDescription.c_str());

  Document doc;
  doc.ParseStream(is);

  // get the "half-chambers" array
  Value& hChs = doc["HalfChambers"];
  assert(hChs.IsArray());

  // loop over the objects (half-chambers) of the array
  for (const auto& halfCh : hChs.GetArray()) {
    // check that "halfCh" is an object
    if (!halfCh.IsObject())
      throw runtime_error("Can't create the half-chambers : wrong Value input");

    int moduleID = halfCh["moduleID"].GetInt();
    const string name = halfCh["name"].GetString();
    // get the chamber number (if the chamber name has a '0' at the 3rd digit, take the number after; otherwise it's the
    // chamber 10)
    int nCh = (name.find('0') == 2) ? name[3] - '0' : 10;

    auto halfChVol = new TGeoVolumeAssembly(name.data());

    // place the support panel corresponding to the chamber number
    auto supRot = new TGeoRotation();
    if (moduleID % 2)
      supRot->RotateY(180.);
    halfChVol->AddNode(gGeoManager->GetVolume(Form("Chamber %d support panel", nCh)), moduleID, supRot);

    // place the slat volumes on the different nodes of the half-chamber
    for (const auto& slat : halfCh["nodes"].GetArray()) {
      // check that "slat" is an object
      if (!slat.IsObject())
        throw runtime_error("Can't create the slat : wrong Value input");

      int detID = slat["detID"].GetInt();

      // place the slat on the half-chamber volume
      halfChVol->AddNode(
        gGeoManager->GetVolume(slat["type"].GetString()), detID,
        new TGeoCombiTrans(slat["position"][0].GetDouble(), slat["position"][1].GetDouble(),
                           slat["position"][2].GetDouble(),
                           new TGeoRotation(Form("Slat%drotation", detID), slat["rotation"][0].GetDouble(),
                                            slat["rotation"][1].GetDouble(), slat["rotation"][2].GetDouble(),
                                            slat["rotation"][3].GetDouble(), slat["rotation"][4].GetDouble(),
                                            slat["rotation"][5].GetDouble())));

    } // end of the node loop

    // place the half-chamber in the top volume
    topVolume.AddNode(
      halfChVol, moduleID,
      new TGeoCombiTrans(halfCh["position"][0].GetDouble(), halfCh["position"][1].GetDouble(), halfCh["position"][2].GetDouble(),
                         new TGeoRotation(Form("%srotation", name.data()), halfCh["rotation"][0].GetDouble(),
                                          halfCh["rotation"][1].GetDouble(), halfCh["rotation"][2].GetDouble(),
                                          halfCh["rotation"][3].GetDouble(), halfCh["rotation"][4].GetDouble(),
                                          halfCh["rotation"][5].GetDouble())));

    // if the dipole is present in the geometry, we place the station 3 half-chambers in it (actually not working)
    if (gGeoManager->GetVolume("Dipole") && (nCh == 5 || nCh == 6))
      topVolume.GetNode(Form("%s_%d", name.data(), moduleID))->SetMotherVolume(gGeoManager->GetVolume("DDIP"));

  } // end of the half-chambers loop
}

//______________________________________________________________________________
vector<TGeoVolume*> getStation345SensitiveVolumes()
{
  /// Create a vector containing the sensitive volume's name of the slats (PCB gas) for the Detector class

  vector<TGeoVolume*> sensitiveVolumeNames;
  for (const auto& [name, nonUsed] : kPcbTypes) {

    auto vol = gGeoManager->GetVolume(Form("%s gas", name.c_str()));

    if (!vol) {
      throw runtime_error(Form("could not get expected volume %s", name.c_str()));
    } else {
      sensitiveVolumeNames.push_back(vol);
    }
  }
  return sensitiveVolumeNames;
}

//______________________________________________________________________________
void createStation345Geometry(TGeoVolume& topVolume)
{
  /// Main function which build and place the slats and the half-chambers volumes
  /// This function must be called by the MCH detector class to build the slat stations geometry.

  // create the identical volumes shared by many elements
  createCommonVolumes();

  // create the different PCB types
  createPCBs();

  // create the support panels
  createSupportPanels();

  // create the different slat types
  createSlats();

  // create and place the half-chambers in the top volume
  buildHalfChambers(topVolume);
}

//______________________________________________________________________________

/// Json string describing all the necessary parameters to place the slats in the half-chambers
const string jsonSlatDescription =
  R"(
{
  "HalfChambers": [
    {
      "name":"SC05I",
      "moduleID":8,
      "position":[0.00, -0.1074, -959.75],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":500,
          "type":"122000SR1",
          "position":[81.25, 0.00, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":501,
          "type":"112200SR2",
          "position":[81.25, 37.80, -4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":502,
          "type":"122200S",
          "position":[81.25, 75.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":503,
          "type":"222000N",
          "position":[61.25, 112.80, -4.00],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":504,
          "type":"220000N",
          "position":[41.25, 146.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":514,
          "type":"220000N",
          "position":[41.25, -146.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":515,
          "type":"222000N",
          "position":[61.25, -112.80, -4.00],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":516,
          "type":"122200S",
          "position":[81.25, -75.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":517,
          "type":"112200SR2",
          "position":[81.25, -37.80, -4.00],
          "rotation":[90, 0, 90, 270, 180, 0]
        }
      ]
    },

    {
      "name":"SC05O",
      "moduleID":9,
      "position":[0.00, 0.1074, -975.25],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":505,
          "type":"220000N",
          "position":[-41.25, 146.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":506,
          "type":"222000N",
          "position":[-61.25, 112.80, 4.00],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":507,
          "type":"122200S",
          "position":[-81.25, 75.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":508,
          "type":"112200SR2",
          "position":[-81.25, 37.80, 4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":509,
          "type":"122000SR1",
          "position":[-81.25, 0.00, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":510,
          "type":"112200SR2",
          "position":[-81.25, -37.80, 4.00],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":511,
          "type":"122200S",
          "position":[-81.25, -75.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":512,
          "type":"222000N",
          "position":[-61.25, -112.80, 4.00],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":513,
          "type":"220000N",
          "position":[-41.25, -146.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        }
      ]
    },

    {
      "name":"SC06I",
      "moduleID":10,
      "position":[0.00, -0.1074, -990.75],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":600,
          "type":"122000NR1",
          "position":[81.25, 0.00, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":601,
          "type":"112200NR2",
          "position":[81.25, 37.80, -4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":602,
          "type":"122200N",
          "position":[81.25, 75.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":603,
          "type":"222000N",
          "position":[61.25, 112.80, -4.00],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":604,
          "type":"220000N",
          "position":[41.25, 146.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":614,
          "type":"220000N",
          "position":[41.25, -146.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":615,
          "type":"222000N",
          "position":[61.25, -112.80, -4.00],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":616,
          "type":"122200N",
          "position":[81.25, -75.50, 4.00],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":617,
          "type":"112200NR2",
          "position":[81.25, -37.80, -4.00],
          "rotation":[90, 0, 90, 270, 180, 0]
        }
      ]
    },

    {
      "name":"SC06O",
      "moduleID":11,
      "position":[0.00, 0.1074, -1006.25],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":605,
          "type":"220000N",
          "position":[-41.25, 146.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":606,
          "type":"222000N",
          "position":[-61.25, 112.80, 4.00],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":607,
          "type":"122200N",
          "position":[-81.25, 75.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":608,
          "type":"112200NR2",
          "position":[-81.25, 37.80, 4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":609,
          "type":"122000NR1",
          "position":[-81.25, 0.00, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":610,
          "type":"112200NR2",
          "position":[-81.25, -37.80, 4.00],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":611,
          "type":"122200N",
          "position":[-81.25, -75.5, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":612,
          "type":"222000N",
          "position":[-61.25, -112.80, 4.00],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":613,
          "type":"220000N",
          "position":[-41.25, -146.50, -4.00],
          "rotation":[90, 180, 90, 90, 180, 0]
        }
      ]
    },

    {
      "name":"SC07I",
      "moduleID":12,
      "position":[0.00, -0.1074, -1259.75],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":700,
          "type":"122330N",
          "position":[140.00, 0.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":701,
          "type":"112233NR3",
          "position":[121.25, 38.20, -4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":702,
          "type":"112230N",
          "position":[101.25, 72.60, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":703,
          "type":"222330N",
          "position":[101.25, 109.20, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":704,
          "type":"223300N",
          "position":[81.25, 138.50, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":705,
          "type":"333000N",
          "position":[61.25, 175.50, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":706,
          "type":"330000N",
          "position":[41.25, 204.50, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":720,
          "type":"330000N",
          "position":[41.25, -204.50, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":721,
          "type":"333000N",
          "position":[61.25, -175.50, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":722,
          "type":"223300N",
          "position":[81.25, -138.50, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":723,
          "type":"222330N",
          "position":[101.25, -109.20, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":724,
          "type":"112230N",
          "position":[101.25, -72.60, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":725,
          "type":"112233NR3",
          "position":[121.25, -38.20, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        }
      ]
    },

    {
      "name":"SC07O",
      "moduleID":13,
      "position":[0.00, -0.1074, -1284.25],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":707,
          "type":"330000N",
          "position":[-41.25, 204.5, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":708,
          "type":"333000N",
          "position":[-61.25, 175.50, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":709,
          "type":"223300N",
          "position":[-81.25, 138.50, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":710,
          "type":"222330N",
          "position":[-101.25, 109.20, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":711,
          "type":"112230N",
          "position":[-101.25, 72.60, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":712,
          "type":"112233NR3",
          "position":[-121.25, 38.20, 4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":713,
          "type":"122330N",
          "position":[-140.00, 0.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":714,
          "type":"112233NR3",
          "position":[-121.25, -38.20, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":715,
          "type":"112230N",
          "position":[-101.25, -72.60, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":716,
          "type":"222330N",
          "position":[-101.25, -109.20, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":717,
          "type":"223300N",
          "position":[-81.25, -138.50, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":718,
          "type":"333000N",
          "position":[-61.25, -175.50, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":719,
          "type":"330000N",
          "position":[-41.25, -204.50, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        }
      ]
    },

    {
      "name":"SC08I",
      "moduleID":14,
      "position":[0.00, -0.1074, -1299.75],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":800,
          "type":"122330N",
          "position":[140.00, 0.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":801,
          "type":"112233NR3",
          "position":[121.25, 38.20, -4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":802,
          "type":"112230N",
          "position":[101.25, 76.05, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":803,
          "type":"222330N",
          "position":[101.25, 113.60, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":804,
          "type":"223300N",
          "position":[81.25, 143.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":805,
          "type":"333000N",
          "position":[61.25, 180.00, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":806,
          "type":"330000N",
          "position":[41.25, 208.60, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":820,
          "type":"330000N",
          "position":[41.25, -208.60, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":821,
          "type":"333000N",
          "position":[61.25, -180.00, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":822,
          "type":"223300N",
          "position":[81.25, -143.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":823,
          "type":"222330N",
          "position":[101.25, -113.60, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":824,
          "type":"112230N",
          "position":[101.25, -76.05, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":825,
          "type":"112233NR3",
          "position":[121.25, -38.20, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        }
      ]
    },

    {
      "name":"SC08O",
      "moduleID":15,
      "position":[0.00, -0.1074, -1315.25],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":807,
          "type":"330000N",
          "position":[-41.25, 208.60, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":808,
          "type":"333000N",
          "position":[-61.25, 180.00, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":809,
          "type":"223300N",
          "position":[-81.25, 143.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":810,
          "type":"222330N",
          "position":[-101.25, 113.60, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":811,
          "type":"112230N",
          "position":[-101.25, 76.05, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":812,
          "type":"112233NR3",
          "position":[-121.25, 38.20, 4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":813,
          "type":"122330N",
          "position":[-140.00, 0.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":814,
          "type":"112233NR3",
          "position":[-121.25, -38.20, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":815,
          "type":"112230N",
          "position":[-101.25, -76.05, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":816,
          "type":"222330N",
          "position":[-101.25, -113.60, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":817,
          "type":"223300N",
          "position":[-81.25, -143.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":818,
          "type":"333000N",
          "position":[-61.25, -180.00, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":819,
          "type":"330000N",
          "position":[-41.25, -208.60, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        }
      ]
    },

    {
      "name":"SC09I",
      "moduleID":16,
      "position":[0.00, -0.1074, -1398.85],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":900,
          "type":"122330N",
          "position":[140.00, 0.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":901,
          "type":"112233NR3",
          "position":[121.25, 38.20, -4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":902,
          "type":"112233N",
          "position":[121.25, 76.10, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":903,
          "type":"222333N",
          "position":[121.25, 113.70, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":904,
          "type":"223330N",
          "position":[101.25, 151.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":905,
          "type":"333300N",
          "position":[81.25, 188.05, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":906,
          "type":"333000N",
          "position":[61.25, 224.80, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":920,
          "type":"333000N",
          "position":[61.25, -224.80, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":921,
          "type":"333300N",
          "position":[81.25, -188.05, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":922,
          "type":"223330N",
          "position":[101.25, -151.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":923,
          "type":"222333N",
          "position":[121.25, -113.70, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":924,
          "type":"112233N",
          "position":[121.25, -76.10, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":925,
          "type":"112233NR3",
          "position":[121.25, -38.20, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        }
      ]
    },

    {
      "name":"SC09O",
      "moduleID":17,
      "position":[0.00, -0.1074, -1414.35],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":907,
          "type":"333000N",
          "position":[-61.25, 224.80, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":908,
          "type":"333300N",
          "position":[-81.25, 188.05, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":909,
          "type":"223330N",
          "position":[-101.25, 151.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":910,
          "type":"222333N",
          "position":[-121.25, 113.70, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":911,
          "type":"112233N",
          "position":[-121.25, 76.10, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":912,
          "type":"112233NR3",
          "position":[-121.25, 38.20, 4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":913,
          "type":"122330N",
          "position":[-140.00, 0.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":914,
          "type":"112233NR3",
          "position":[-121.25, -38.20, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":915,
          "type":"112233N",
          "position":[-121.25, -76.10, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":916,
          "type":"222333N",
          "position":[-121.25, -113.70, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":917,
          "type":"223330N",
          "position":[-101.25, -151, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":918,
          "type":"333300N",
          "position":[-81.25, -188.05, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":919,
          "type":"333000N",
          "position":[-61.25, -224.80, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        }
      ]
    },

    {
      "name":"SC10I",
      "moduleID":18,
      "position":[0.00, -0.1074, -1429.85],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":1000,
          "type":"122330N",
          "position":[140.00, 0.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1001,
          "type":"112233NR3",
          "position":[121.25, 38.20, -4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1002,
          "type":"112233N",
          "position":[121.25, 76.10, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1003,
          "type":"222333N",
          "position":[121.25, 113.70, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":1004,
          "type":"223330N",
          "position":[101.25, 151.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1005,
          "type":"333300N",
          "position":[81.25, 188.05, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":1006,
          "type":"333000N",
          "position":[61.25, 224.80, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1020,
          "type":"333000N",
          "position":[61.25, -224.80, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1021,
          "type":"333300N",
          "position":[81.25, -188.05, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":1022,
          "type":"223330N",
          "position":[101.25, -151.00, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1023,
          "type":"222333N",
          "position":[121.25, -113.70, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        },
        {
          "detID":1024,
          "type":"112233N",
          "position":[121.25, -76.10, 4.25],
          "rotation":[90, 0, 90, 90, 0, 0]
        },
        {
          "detID":1025,
          "type":"112233NR3",
          "position":[121.25, -38.20, -4.25],
          "rotation":[90, 0, 90, 270, 180, 0]
        }
      ]
    },

    {
      "name":"SC10O",
      "moduleID":19,
      "position":[0.00, -0.1074, -1445.35],
      "rotation":[90, 0, 90.794, 90, 0.794, 90],
      "nodes":[
        {
          "detID":1007,
          "type":"333000N",
          "position":[-61.25, 224.80, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1008,
          "type":"333300N",
          "position":[-81.25, 188.05, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":1009,
          "type":"223330N",
          "position":[-101.25, 151.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1010,
          "type":"222333N",
          "position":[-121.25, 113.70, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":1011,
          "type":"112233N",
          "position":[-121.25, 76.10, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1012,
          "type":"112233NR3",
          "position":[-121.25, 38.20, 4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1013,
          "type":"122330N",
          "position":[-140.00, 0.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1014,
          "type":"112233NR3",
          "position":[-121.25, -38.20, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":1015,
          "type":"112233N",
          "position":[-121.25, -76.10, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1016,
          "type":"222333N",
          "position":[-121.25, -113.70, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":1017,
          "type":"223330N",
          "position":[-101.25, -151.00, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        },
        {
          "detID":1018,
          "type":"333300N",
          "position":[-81.25, -188.05, 4.25],
          "rotation":[90, 180, 90, 270, 0, 0]
        },
        {
          "detID":1019,
          "type":"333000N",
          "position":[-61.25, -224.80, -4.25],
          "rotation":[90, 180, 90, 90, 180, 0]
        }
      ]
    }
  ]
}

)";
} // namespace mch
} // namespace o2
