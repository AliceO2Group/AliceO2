// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   SlatGeometry.cxx
/// \brief  Implementation of the slat-stations geometry
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   22 mars 2018

#include "Materials.h"
#include "Station345Geometry.h"

#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoShape.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TMath.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

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

/// Gas
const float kGasLength = 40.;
const float kGasHeight = 40.;
const float kGasWidth = 2 * 0.25;

/// PCB
const float kPCBLength = kGasLength;
const float kShortPCBLength = 35.;
const float kRoundedPCBLength = 42.5;
const float kR1PCBLength = 19.25;
const float kPCBHeight = 58.;
const float kPCBWidth = 0.002; // changed w.r.t AliRoot after investigation

/// Insulator (FR4)
const float kInsuWidth = 0.04; // changed w.r.t AliRoot after investigation

/// PCB volume (gas + 2*(pcb plate + insulator))
const float kTotalPCBWidth = kGasWidth + 2 * (kPCBWidth + kInsuWidth);

/// Slat panel = honeycomb nomex + 2 carbon fiber skins
const float kSlatPanelHeight = 42.5; // according to construction plan, its length depends on the number of PCBs

/// Glue
const float kGlueWidth = 0.004; // according to construction plan

/// Bulk Nomex
const float kNomexBulkWidth = 0.025; // according to AliRoot and construction plan

/// Carbon fiber
const float kCarbonWidth = 0.02; // according to AliRoot and construction plan

/// Honeycomb Nomex
const float kNomexWidth = 0.8; // according to AliRoot and construction plan

/// Spacers (noryl)
const float kSpacerWidth = kGasWidth;
// Horizontal
const float kHoriSpacerHeight = 1.95; // according to AliRoot and construction plan
// Vertical
const float kVertSpacerLength = 2.5; // according to AliRoot and construction plan
// Rounded
const float kRoundedSpacerLength = 2.; // according to AliRoot and construction plan

/// Border (Rohacell)
const float kBorderHeight = 5.;       // to be checked !
const float kBorderWidth = kGasWidth; // to be checked

/// MANU (NULOC, equivalent to 44 mum width of copper according to AliRoot)
const float kMANULength = 2.5;      // according to AliRoot
const float kMANUHeight = 5 - 0.35; // according to construction plan
const float kMANUWidth = 0.0044;    // according to AliRoot
const float kMANUypos = 0.45 + (kSlatPanelHeight + kMANUHeight) / 2.;

/// Cables (copper)
// Low voltage (values from AliRoot)
const float kLVcableHeight = 2.6;
const float kLVcableWidth = 0.026;

/// Support panels (to be checked !!!)
const float kCarbonSupportWidth = 0.03;
const float kGlueSupportWidth = 0.02; // added w.r.t AliRoot to match the construction plans
const float kNomexSupportWidth = 1.5;
const float kSupportWidth = 2 * (kCarbonSupportWidth + kGlueSupportWidth) + kNomexSupportWidth;
const float kSupportHeightSt3 = 361.;
const float kSupportHeightSt4 = 530.;
const float kSupportHeightSt5 = 570.;
const float kSupportLengthCh5 = 162.;
const float kSupportLengthCh6 = 167.;
const float kSupportLengthSt45 = 260.;

// Inner radii
const float kRadSt3 = 29.5;
const float kRadSt45 = 37.5;

// Y position of the rounded slats
const float kRoundedSlatYposSt3 = 37.8;
const float kRoundedSlatYposSt45 = 38.2;

// PCB types {name, number of MANUs array}
const map<string, array<int, 4>> kPcbTypes = { { "B1N1", { 10, 10, 7, 7 } }, { "B2N2-", { 5, 5, 4, 3 } }, { "B2N2+", { 5, 5, 3, 4 } }, { "B3-N3", { 3, 2, 2, 2 } }, { "B3+N3", { 2, 3, 2, 2 } }, { "R1", { 3, 4, 2, 3 } }, { "R2", { 13, 4, 9, 3 } }, { "R3", { 13, 1, 10, 0 } }, { "S2-", { 4, 5, 3, 3 } }, { "S2+", { 5, 4, 3, 3 } } };

// Slat types
const map<string, vector<string>> kSlatTypes = { { "122000SR1", { "R1", "B1N1", "B2N2+", "S2-" } },
                                                 { "112200SR2", { "R2", "B1N1", "B2N2+", "S2-" } },
                                                 { "122200S", { "B1N1", "B2N2-", "B2N2-", "S2+" } },
                                                 { "222000N", { "B2N2-", "B2N2-", "B2N2-" } },
                                                 { "220000N", { "B2N2-", "B2N2-" } },
                                                 { "122000NR1", { "R1", "B1N1", "B2N2+", "B2N2+" } },
                                                 { "112200NR2", { "R2", "B1N1", "B2N2+", "B2N2+" } },
                                                 { "122200N", { "B1N1", "B2N2-", "B2N2-", "B2N2-" } },
                                                 { "122330N", { "B1N1", "B2N2+", "B2N2-", "B3-N3", "B3-N3" } },
                                                 { "112233NR3", { "R3", "B1N1", "B2N2-", "B2N2+", "B3+N3", "B3+N3" } },
                                                 { "112230N", { "B1N1", "B1N1", "B2N2-", "B2N2-", "B3-N3" } },
                                                 { "222330N", { "B2N2+", "B2N2+", "B2N2-", "B3-N3", "B3-N3" } },
                                                 { "223300N", { "B2N2+", "B2N2-", "B3-N3", "B3-N3" } },
                                                 { "333000N", { "B3-N3", "B3-N3", "B3-N3" } },
                                                 { "330000N", { "B3-N3", "B3-N3" } },
                                                 { "112233N", { "B1N1", "B1N1", "B2N2+", "B2N2-", "B3-N3", "B3-N3" } },
                                                 { "222333N",
                                                   { "B2N2+", "B2N2+", "B2N2+", "B3-N3", "B3-N3", "B3-N3" } },
                                                 { "223330N", { "B2N2+", "B2N2+", "B3-N3", "B3-N3", "B3-N3" } },
                                                 { "333300N", { "B3+N3", "B3-N3", "B3-N3", "B3-N3" } } };

extern const string jsonSlatDescription;

//______________________________________________________________________________
void createCommonVolumes()
{
  /// Build the identical volumes (constant shapes, dimensions, ...) shared by many elements

  // the right vertical spacer (identical to any slat)
  new TGeoVolume("Right spacer", new TGeoBBox(kVertSpacerLength / 2., kSlatPanelHeight / 2., kSpacerWidth / 2.),
                 assertMedium(Medium::Noryl));

  // the top spacers and borders : 4 lengths possible according to the PCB shape
  float lengths[4] = { kShortPCBLength, kPCBLength, kR1PCBLength, kRoundedPCBLength };
  for (int i = 0; i < 4; i++) {
    // top spacer
    new TGeoVolume(Form("Top spacer %.2f long", lengths[i]),
                   new TGeoBBox(lengths[i] / 2., kHoriSpacerHeight / 2., kSpacerWidth / 2.),
                   assertMedium(Medium::Noryl));

    // top border
    new TGeoVolume(Form("Top border %.2f long", lengths[i]),
                   new TGeoBBox(lengths[i] / 2., kBorderHeight / 2., kBorderWidth / 2.),
                   assertMedium(Medium::Rohacell));
  }

  // MANU (to be checked !!!)
  new TGeoVolume("MANU", new TGeoBBox(kMANULength / 2., kMANUHeight / 2., kMANUWidth / 2.),
                 assertMedium(Medium::Copper));
}

//______________________________________________________________________________
void createPCBs()
{
  /// Build the different PCB types

  /// A PCB is a pile-up of several material layers, from in to out : sensitive gas, pcb plate and insulator
  /// There are two types of pcb plates : a "bending" and a "non-bending" one. We build the PCB volume such that the
  /// bending side faces the IP (z>0). When placing the slat on the half-chambers, the builder grabs the rotation to
  /// apply from the JSON. By doing so, we make sure that we match the mapping convention

  // Define some necessary variables
  string bendName, nonbendName; // pcb plate names
  float width = 0., gasLength = 0., pcbLength = 0., borderLength = 0., radius = 0., curvRad = 0., pcbShift = 0.,
        gasShift = 0., x = 0., y = 0.,
        z = 0.; // useful parameters for dimensions and positions
  int numb = 1, nMANU = 0;

  for (const auto& pcbType : kPcbTypes) { // loop over the PCB types of the array

    const string pcbName = pcbType.first;
    auto name = (const char*)pcbName.data();

    auto pcb = new TGeoVolumeAssembly(name);

    // Reset shift variables
    gasShift = 0.;
    pcbShift = 0.;

    numb = pcbName[1] - '0'; // char -> int conversion

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
        gasLength = kGasLength;
        pcbLength = kPCBLength;
        bendName = (numb == 3) ? pcbName.substr(0, 3) : pcbName.substr(0, 2);
        nonbendName = (numb == 3) ? pcbName.substr(3) : pcbName.substr(2);
    }

    borderLength = pcbLength;

    // create the volume of each material (a box by default)
    // sensitive gas
    auto gas = new TGeoVolume(Form("%s gas", name),
                              new TGeoBBox(Form("%sGasBox", name), gasLength / 2., kGasHeight / 2., kGasWidth / 2.),
                              assertMedium(Medium::Gas));

    x = pcbLength / 2.;
    y = kPCBHeight / 2.;
    // bending pcb plate
    auto bend = new TGeoVolume(bendName.data(), new TGeoBBox(Form("%sBendBox", name), x, y, kPCBWidth / 2.),
                               assertMedium(Medium::Copper));
    // non-bending pcb plate
    auto nonbend = new TGeoVolume(nonbendName.data(), new TGeoBBox(Form("%sNonBendBox", name), x, y, kPCBWidth / 2.),
                                  assertMedium(Medium::Copper));
    // insulating material
    auto insu = new TGeoVolume(Form("%s insulator", name), new TGeoBBox(Form("%sInsuBox", name), x, y, kInsuWidth / 2.),
                               assertMedium(Medium::FR4));

    // bottom spacer (noryl)
    auto spacer = new TGeoVolume(Form("%s bottom spacer", name),
                                 new TGeoBBox(Form("%sSpacerBox", name), x, kHoriSpacerHeight / 2., kSpacerWidth / 2.),
                                 assertMedium(Medium::Noryl));

    // change the volume shape if we are creating a rounded PCB
    if (pcbName.front() == 'R') {
      // LHC beam pipe radius ("R3" -> it is a slat of a station 4 or 5)
      radius = (numb == 3) ? kRadSt45 : kRadSt3;
      // y position of the PCB center w.r.t the beam pipe shape
      switch (numb) {
        case 1:
          y = 0.; // central for "R1"
          break;
        case 2:
          y = kRoundedSlatYposSt3; // "R2" -> station 3
          break;
        default:
          y = kRoundedSlatYposSt45; // "R3" -> station 4 or 5
          break;
      }
      // compute the radius of curvature of the PCB we want to create
      curvRad = radius + kRoundedSpacerLength;

      // create the relative pipe position
      x = -kRoundedPCBLength + (gasLength + kVertSpacerLength) / 2.;
      auto gasTrans = new TGeoTranslation(Form("GasHoleY%.1fShift", y), x, -y, 0.);
      gasTrans->RegisterYourself();

      x = -kRoundedPCBLength + (pcbLength + kVertSpacerLength) / 2.;
      auto pcbTrans = new TGeoTranslation(Form("PCBholeY%.1fShift", y), x, -y, 0.);
      pcbTrans->RegisterYourself();

      auto spacerTrans =
        new TGeoTranslation(Form("SpacerHoleY%.1fShift", y), x, -y + (kGasHeight + kHoriSpacerHeight) / 2., 0.);
      spacerTrans->RegisterYourself();

      // for each volume, create a hole and change the volume shape by extracting the pipe shape
      new TGeoTube(Form("GasHoleR%.1f", curvRad), 0., curvRad, kGasWidth / 2);
      gas->SetShape(new TGeoCompositeShape(Form("R%.1fY%.1fGasShape", curvRad, y),
                                           Form("%sGasBox-GasHoleR%.1f:GasHoleY%.1fShift", name, curvRad, y)));

      new TGeoTube(Form("HoleR%.1f", curvRad), 0., curvRad, kPCBWidth / 2.);
      bend->SetShape(new TGeoCompositeShape(Form("R%.1fY%.1fBendShape", curvRad, y),
                                            Form("%sBendBox-HoleR%.1f:PCBholeY%.1fShift", name, curvRad, y)));

      nonbend->SetShape(new TGeoCompositeShape(Form("R%.1fY%.1fNonBendShape", curvRad, y),
                                               Form("%sNonBendBox-HoleR%.1f:PCBholeY%.1fShift", name, curvRad, y)));

      new TGeoTube(Form("InsuHoleR%.1f", curvRad), 0., curvRad, kInsuWidth / 2.);
      insu->SetShape(new TGeoCompositeShape(Form("R%.1fY%.1fInsuShape", curvRad, y),
                                            Form("%sInsuBox-InsuHoleR%.1f:PCBholeY%.1fShift", name, curvRad, y)));

      if (pcbName.back() != '1') { // change the bottom spacer and border shape for "R2" and "R3" PCBs
        new TGeoTube(Form("SpacerHoleR%.1f", radius), 0., radius, kSpacerWidth / 2);
        spacer->SetShape(
          new TGeoCompositeShape(Form("R%.1fY%.1fSpacerShape", radius, y),
                                 Form("%sSpacerBox-SpacerHoleR%.1f:SpacerHoleY%.1fShift", name, radius, y)));
        borderLength -= (pcbName.back() == '3') ? curvRad : curvRad + kRoundedSpacerLength / 2.;
      }
    }

    /// place all the layers in the pcb volume assembly

    width = kGasWidth; // increment this value when adding a new layer
    pcb->AddNode(gas, 1, new TGeoTranslation(gasShift / 2., 0., 0.));

    width += kPCBWidth;
    x = pcbShift / 2.;
    z = width / 2.;
    pcb->AddNode(bend, 1, new TGeoTranslation(x, 0., z));
    pcb->AddNode(nonbend, 2, new TGeoTranslation(x, 0., -z));

    width += kInsuWidth;
    z = width / 2.;
    pcb->AddNode(insu, 1, new TGeoTranslation(x, 0., z));
    pcb->AddNode(insu, 2, new TGeoTranslation(x, 0., -z));

    // the horizontal spacers
    y = (kGasHeight + kHoriSpacerHeight) / 2.;
    pcb->AddNode(gGeoManager->GetVolume(Form("Top spacer %.2f long", pcbLength)), 1, new TGeoTranslation(x, y, 0.));
    pcb->AddNode(spacer, 1, new TGeoTranslation(x, -y, 0.));

    // the borders
    y = (kPCBHeight - kBorderHeight) / 2.;
    pcb->AddNode(gGeoManager->GetVolume(Form("Top border %.2f long", pcbLength)), 1, new TGeoTranslation(x, y, 0.));
    x = (pcbShift + pcbLength - borderLength) / 2.;
    pcb->AddNode(gGeoManager->MakeBox(Form("%s bottom border", name), assertMedium(Medium::Rohacell),
                                      borderLength / 2., kBorderHeight / 2., kBorderWidth / 2.),
                 1, new TGeoTranslation(x, -y, 0.));

    // the MANUs
    width += kMANUWidth;
    auto manus = pcbType.second;
    float length, shift;
    for (int i = 0; i < manus.size(); i++) {

      nMANU = manus[i];
      length = (i % 2) ? borderLength : pcbLength;
      shift = (i % 2) ? pcbLength - borderLength : 0.;
      y = TMath::Power(-1, i % 2) * kMANUypos;
      z = TMath::Power(-1, i / 2) * width / 2.;

      for (int j = 0; j < nMANU; j++) {
        pcb->AddNode(gGeoManager->GetVolume("MANU"), 100 * i + j,
                     new TGeoTranslation((j - nMANU / 2) * (length / nMANU) - (nMANU % 2 - 1) * (length / (2 * nMANU)) +
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

  // Define some necessary variables
  float length = 0., center = 0., PCBlength = 0., gasLength = 0., x = 0., y = 0., z = 0., xRoundedPos = 0., angMin = 0.,
        angMax = 0., radius = 0., width = 0., cableLength = 0., leftSpacerHeight = 0., panelShift = 0.;
  int ivol = 0;

  // Mirror rotation for the slat panel on the non-bending side
  auto mirror = new TGeoRotation();
  mirror->ReflectZ(true);

  for (const auto& slatType : kSlatTypes) {

    const string typeName = slatType.first;   // slat type name
    auto name = (const char*)typeName.data(); // slat name (easier to name volumes)
    const auto PCBs = slatType.second;        // PCB names vector

    // create the slat volume assembly
    auto slat = new TGeoVolumeAssembly(name);

    // Reset slat variables
    length = 2 * kVertSpacerLength; // vertical spacers
    center = (PCBs.size() - 1) * kGasLength / 2;
    panelShift = 0.;
    ivol = 0;

    // loop over the number of PCBs in the current slat
    for (const auto& pcb : PCBs) {

      gasLength = kGasLength;

      switch (pcb.front()) {
        case 'R':
          PCBlength = (pcb.back() == '1') ? kR1PCBLength : kRoundedPCBLength;
          panelShift -= PCBlength - kRoundedPCBLength;
          break;
        case 'S':
          PCBlength = kShortPCBLength;
          gasLength = kShortPCBLength;
          panelShift += PCBlength - kPCBLength;
          break;
        default:
          PCBlength = kPCBLength;
          break;
      }

      length += PCBlength;

      // place the corresponding PCB volume in the slat and correct the origin of the slat
      slat->AddNode(gGeoManager->GetVolume(pcb.data()), ivol + 1,
                    new TGeoTranslation(ivol * kPCBLength - 0.5 * (kPCBLength - gasLength) - center, 0, 0));
      ivol++;

    } // end of the PCBs loop

    // compute the LV cable length
    cableLength = (typeName.find('3') < typeName.size()) ? kSupportLengthSt45 : kSupportLengthCh5;
    if (typeName == "122200N")
      cableLength = kSupportLengthCh6;
    cableLength -= length;
    if (typeName == "122330N")
      cableLength -= kGasLength;

    leftSpacerHeight = kSlatPanelHeight;
    if (typeName.find('R') < typeName.size()) {
      length -= kVertSpacerLength;   // don't count the vertical spacer length twice in the case of rounded slat
      leftSpacerHeight = kGasHeight; // to avoid overlaps with the horizontal spacers
    }

    // left vertical spacer
    auto left = new TGeoVolume(
      Form("%s left spacer", name),
      new TGeoBBox(Form("%sLeftSpacerBox", name), kVertSpacerLength / 2., leftSpacerHeight / 2., kSpacerWidth / 2.),
      assertMedium(Medium::Noryl));

    // glue a slat panel on each side of the PCBs
    auto panel = new TGeoVolumeAssembly(Form("%s panel", name));

    x = length / 2.;
    y = kSlatPanelHeight / 2.;

    // glue
    auto glue =
      new TGeoVolume(Form("%s panel glue", name), new TGeoBBox(Form("%sGlueBox", name), x, y, kGlueWidth / 2.),
                     assertMedium(Medium::Glue));

    // nomex (bulk)
    auto nomexBulk = new TGeoVolume(
      Form("%s panel nomex (bulk)", name),
      new TGeoBBox(Form("%sNomexBulkBox", name), length / 2., kSlatPanelHeight / 2., kNomexBulkWidth / 2.),
      assertMedium(Medium::BulkNomex));

    // carbon fiber
    auto carbon =
      new TGeoVolume(Form("%s panel carbon fiber", name),
                     new TGeoBBox(Form("%sCarbonBox", name), x, y, kCarbonWidth / 2.), assertMedium(Medium::Carbon));

    // nomex (honeycomb)
    auto nomex =
      new TGeoVolume(Form("%s panel nomex (honeycomb)", name),
                     new TGeoBBox(Form("%sNomexBox", name), length / 2., kSlatPanelHeight / 2., kNomexWidth / 2.),
                     assertMedium(Medium::HoneyNomex));

    // change the volume shape if we are creating a rounded slat
    if (typeName.find('R') < typeName.size()) {

      // LHC beam pipe radius ("NR3" -> it is a slat of a station 4 or 5)
      radius = (typeName.back() == '3') ? kRadSt45 : kRadSt3;

      // extreme angle value for the rounded spacer
      angMax = 90.;

      // position of the slat center w.r.t the beam pipe center
      x = (length - kVertSpacerLength) / 2.;
      xRoundedPos = x - panelShift / 2.;

      // change the LV cable length for st.3 slats
      cableLength = (typeName.find('S') < typeName.size()) ? kSupportLengthCh5 : kSupportLengthCh6;

      // change the above values if necessary
      switch (typeName.back()) { // get the last character
        case '1':                // central for "S(N)R1"
          y = 0.;
          x = 2 * kPCBLength + (panelShift + kVertSpacerLength) / 2.;
          xRoundedPos = kRoundedPCBLength + kPCBLength - kRoundedSpacerLength / 2.;
          angMin =
            -TMath::RadToDeg() * TMath::ACos((kRoundedPCBLength - kR1PCBLength - kRoundedSpacerLength / 2.) / radius);
          angMax = -angMin;
          break;
        case '2': // "S(N)R2" -> station 3
          y = kRoundedSlatYposSt3;
          angMin = TMath::RadToDeg() * TMath::ASin((y - kSlatPanelHeight / 2.) / (radius + kRoundedSpacerLength / 2.));
          break;
        default: // "NR3" -> station 4 or 5
          y = kRoundedSlatYposSt45;
          angMin = TMath::RadToDeg() * TMath::ASin((y - kSlatPanelHeight / 2.) / (radius + kRoundedSpacerLength / 2.));
          cableLength = kSupportLengthSt45;
          break;
      }

      cableLength -= x + length / 2.;

      // create and place the rounded spacer
      slat->AddNode(gGeoManager->MakeTubs(Form("%s rounded spacer", name), assertMedium(Medium::Noryl), radius,
                                          radius + kRoundedSpacerLength, kSpacerWidth, angMin, angMax),
                    1, new TGeoTranslation(-xRoundedPos, -y, 0.));

      // create the pipe position
      auto pipeShift = new TGeoTranslation(Form("%sPanelHoleShift", name), -x, -y, 0.);
      pipeShift->RegisterYourself();

      auto leftSpacerShift = new TGeoTranslation(Form("%sLeftSpacerShift", name), 0., -y, 0.);
      leftSpacerShift->RegisterYourself();

      // for each volume, create a hole and change the volume shape by extracting the pipe shape

      new TGeoTube(Form("%sGlueHole", name), 0., radius, kGlueWidth / 2.);
      glue->SetShape(new TGeoCompositeShape(Form("%sGlueShape", name),
                                            Form("%sGlueBox-%sGlueHole:%sPanelHoleShift", name, name, name)));

      new TGeoTube(Form("%sNomexBulkHole", name), 0., radius, kNomexBulkWidth / 2.);
      nomexBulk->SetShape(new TGeoCompositeShape(
        Form("%sNomexBulkShape", name), Form("%sNomexBulkBox-%sNomexBulkHole:%sPanelHoleShift", name, name, name)));

      new TGeoTube(Form("%sCarbonHole", name), 0., radius, kCarbonWidth / 2.);
      carbon->SetShape(new TGeoCompositeShape(Form("%sCarbonShape", name),
                                              Form("%sCarbonBox-%sCarbonHole:%sPanelHoleShift", name, name, name)));

      new TGeoTube(Form("%sNomexHole", name), 0., radius, kNomexWidth / 2.);
      nomex->SetShape(new TGeoCompositeShape(Form("%sNomexShape", name),
                                             Form("%sNomexBox-%sNomexHole:%sPanelHoleShift", name, name, name)));

      new TGeoTube(Form("%sLeftSpacerHole", name), 0., radius, kSpacerWidth / 2.);
      left->SetShape(new TGeoCompositeShape(
        Form("%sLeftSpacerShape", name), Form("%sLeftSpacerBox-%sLeftSpacerHole:%sLeftSpacerShift", name, name, name)));
    } // end of the "rounded" condition

    // place all the layers in the slat panel volume assembly
    // be careful : the panel origin is on the glue edge !

    width = kGlueWidth; // increment this value when adding a new layer
    panel->AddNode(glue, 1, new TGeoTranslation(0., 0., width / 2.));

    panel->AddNode(nomexBulk, 1, new TGeoTranslation(0., 0., width + kNomexBulkWidth / 2.));
    width += kNomexBulkWidth;

    panel->AddNode(glue, 2, new TGeoTranslation(0., 0., width + kGlueWidth / 2.));
    width += kGlueWidth;

    panel->AddNode(carbon, 1, new TGeoTranslation(0., 0., width + kCarbonWidth / 2.));
    width += kCarbonWidth;

    panel->AddNode(nomex, 1, new TGeoTranslation(0., 0., width + kNomexWidth / 2.));
    width += kNomexWidth;

    panel->AddNode(carbon, 2, new TGeoTranslation(0., 0., width + kCarbonWidth / 2.));

    // place the panel volume on each side of the slat volume assembly
    x = panelShift / 2.;
    z = kTotalPCBWidth / 2.;
    slat->AddNode(panel, 1, new TGeoTranslation(x, 0., z));
    slat->AddNode(panel, 2, new TGeoCombiTrans(x, 0., -z, mirror));

    // place the vertical spacers
    x = (length - kVertSpacerLength) / 2.;
    slat->AddNode(gGeoManager->GetVolume("Right spacer"), 1, new TGeoTranslation(x + panelShift / 2., 0., 0.));
    // don't place a left spacer for S(N)R1 slat
    if (typeName.back() != '1')
      slat->AddNode(left, 1, new TGeoTranslation(-x + panelShift / 2., 0., 0.));

    // place the LV cables (top and bottom)
    cableLength += kVertSpacerLength;
    auto LVcable = gGeoManager->MakeBox(Form("%s LV cable", name), assertMedium(Medium::Copper), cableLength / 2.,
                                        kLVcableHeight / 2., kLVcableWidth / 2.);
    x = -kVertSpacerLength + (length + cableLength + panelShift) / 2.;
    y = kMANUypos + (kMANUHeight + kLVcableHeight) / 2.;
    z = -(kTotalPCBWidth - kLVcableWidth) / 2.;
    slat->AddNode(LVcable, 1, new TGeoTranslation(x, y, z));
    slat->AddNode(LVcable, 2, new TGeoTranslation(x, -y, z));

  } // end of the slat loop
}

//______________________________________________________________________________
void createSupportPanels()
{
  /// Function building the half-chamber support panels (one different per chamber)

  // dimensions
  float halfLength = 0., halfHeight = 0., width = 0., radius = 0., z = 0.;

  for (int i = 5; i <= 10; i++) {

    // define the support panel volume
    auto support = new TGeoVolumeAssembly(Form("Chamber %d support panel", i));

    if (i <= 6) { // station 3 half-chambers
      halfHeight = kSupportHeightSt3 / 2.;
      halfLength = (i == 5) ? kSupportLengthCh5 / 2. : kSupportLengthCh6 / 2.;
    } else { // station 4 or 5
      halfLength = kSupportLengthSt45 / 2.;
      halfHeight = (i <= 8) ? kSupportHeightSt4 / 2. : kSupportHeightSt5 / 2.;
    }

    // LHC beam pipe radius at the given chamber z position
    radius = (i <= 6) ? kRadSt3 : kRadSt45;

    // place this shape on the ALICE x=0 coordinate
    auto holeTrans = new TGeoTranslation(Form("holeCh%dShift", i), -halfLength + kVertSpacerLength / 2., 0., 0.);
    holeTrans->RegisterYourself();

    // create the hole in the nomex volume
    z = kNomexSupportWidth / 2.;
    new TGeoTube(Form("NomexSupportPanelHoleCh%d", i), 0., radius, z);

    // create a box for the nomex volume
    new TGeoBBox(Form("NomexSupportPanelCh%dBox", i), halfLength, halfHeight, z);

    // create the nomex volume, change its shape by extracting the pipe shape and place it in the support panel
    width = kNomexSupportWidth; // increment the width when adding a new layer
    support->AddNode(
      new TGeoVolume(
        Form("NomexSupportPanelCh%d", i),
        new TGeoCompositeShape(Form("NomexSupportPanelCh%dShape", i),
                               Form("NomexSupportPanelCh%dBox-NomexSupportPanelHoleCh%d:holeCh%dShift", i, i, i)),
        assertMedium(Medium::HoneyNomex)),
      i, new TGeoTranslation(halfLength, 0., 0.));

    // create the hole in the glue volume
    z = kGlueSupportWidth / 2.;
    new TGeoTube(Form("GlueSupportPanelHoleCh%d", i), 0., radius, z);

    // create a box for the glue volume
    new TGeoBBox(Form("GlueSupportPanelCh%dBox", i), halfLength, halfHeight, z);

    // create the glue volume and change its shape by extracting the pipe shape
    auto glue = new TGeoVolume(
      Form("GlueSupportPanelCh%d", i),
      new TGeoCompositeShape(Form("GlueSupportPanelCh%dShape", i),
                             Form("GlueSupportPanelCh%dBox-GlueSupportPanelHoleCh%d:holeCh%dShift", i, i, i)),
      assertMedium(Medium::Glue));

    // place it on each side of the nomex volume
    width += kGlueSupportWidth;
    z = width / 2.;
    support->AddNode(glue, 1, new TGeoTranslation(halfLength, 0., z));
    support->AddNode(glue, 2, new TGeoTranslation(halfLength, 0., -z));

    // create the hole in the carbon volume
    z = kCarbonSupportWidth / 2.;
    new TGeoTube(Form("CarbonSupportPanelHoleCh%d", i), 0., radius, z);

    // create a box for the carbon volume
    new TGeoBBox(Form("CarbonSupportPanelCh%dBox", i), halfLength, halfHeight, z);

    // create the carbon volume and change its shape by extracting the pipe shape
    auto carbon = new TGeoVolume(
      Form("CarbonSupportPanelCh%d", i),
      new TGeoCompositeShape(Form("CarbonSupportPanelCh%dShape", i),
                             Form("CarbonSupportPanelCh%dBox-CarbonSupportPanelHoleCh%d:holeCh%dShift", i, i, i)),
      assertMedium(Medium::Carbon));

    // place it on each side of the glue volume
    width += kCarbonSupportWidth;
    z = width / 2.;
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

  int moduleID = 0, nCh = 0, detID = 0;

  // loop over the objects (half-chambers) of the array
  for (const auto& halfCh : hChs.GetArray()) {
    // check that "halfCh" is an object
    if (!halfCh.IsObject())
      throw runtime_error("Can't create the half-chambers : wrong Value input");

    moduleID = halfCh["moduleID"].GetInt();
    const string name = halfCh["name"].GetString();
    // get the chamber number (if the chamber name has a '0' at the 3rd digit, take the number after; otherwise it's the
    // chamber 10)
    nCh = (name.find('0') == 2) ? name[3] - '0' : 10;

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

      detID = slat["detID"].GetInt();

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
      new TGeoCombiTrans(halfCh["position"][0].GetDouble(), halfCh["position"][1].GetDouble(),
                         halfCh["position"][2].GetDouble(),
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
  for (const auto& pcb : kPcbTypes) {

    const auto& volName = pcb.first;
    auto vol = gGeoManager->GetVolume(Form("%s gas", volName.data()));

    if (!vol) {
      throw runtime_error(Form("could not get expected volume %s", volName.c_str()));
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
  /// This function must me called by the MCH detector class to build the slat stations geometry.

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
      "position":[0.00, -0.1663, -959.75],
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
      "position":[0.00, 0.1663, -975.25],
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
      "position":[0.00, -0.1663, -990.75],
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
      "position":[0.00, 0.1663, -1006.25],
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
      "position":[0.00, -0.1663, -1259.75],
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
      "position":[0.00, -0.1663, -1284.25],
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
      "position":[0.00, -0.1663, -1299.75],
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
      "position":[0.00, -0.1663, -1315.25],
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
      "position":[0.00, -0.1663, -1398.85],
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
      "position":[0.00, -0.1663, -1414.35],
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
      "position":[0.00, -0.1663, -1429.85],
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
      "position":[0.00, -0.1663, -1445.35],
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
