// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Geometry.cxx
/// \brief  Implementation of the trigger-stations geometry
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   19 june 2018

#include "Geometry.h"

#include "Materials.h"
#include "MIDBase/Constants.h"

#include <TGeoVolume.h>
#include <TGeoManager.h>
#include <TGeoShape.h>
#include <TGeoCompositeShape.h>

namespace o2
{
namespace mid
{

/// RPC thickness

// Gas gap (gas enclosed by bakelite electrodes + graphite and spacers)
const float kGasHalfThickness = 0.2 / 2.;
const float kSpacerHalfThickness = 0.2 / 2.;
const float kElectrodHalfThickness = 0.2 / 2.;

// Insulating material (PET foil) between the gas gap and the strip plane
const float kInsulatorHalfThickness = 0.01 / 2.;

// Strip plane (styrofoam + mylar/copper foils)
const float kStyrofoamHalfThickness = 0.3 / 2.;
const float kMylarHalfThickness = 0.019 / 2.;
const float kCopperHalfThickness = 0.002 / 2.;

// Stiffener plane (nomex enclosed between aluminium sheets)
const float kNomexHalfThickness = 0.88 / 2.;
const float kAluminiumHalfThickness = 0.06 / 2.;

/// RPC y position in the first chamber
const float kRPCypos[] = { 0.0000, 68.1530, 135.6953, 204.4590, 271.3907, 272.6120, 203.5430, 136.3060, 67.8477 };

TGeoVolume* createRPC(const char* type, int iChamber)
{
  /// Function building a resisitive plate chamber (RPC), the detection element of the MID, of a given type and for the given chamber number.

  const char* name = Form("%sRPC%d", type, iChamber);

  auto rpc = new TGeoVolumeAssembly(name);

  // get the dimensions from MIDBase/Constants
  double halfLength = (!strcmp(type, "short")) ? Constants::sRPCShortHalfLength : Constants::sRPCHalfLength;
  halfLength *= Constants::sScaleFactors[iChamber];
  double halfHeight = Constants::getRPCHalfHeight(iChamber);
  halfHeight *= Constants::sScaleFactors[iChamber];

  /// create the volume of each material (a box by default)

  /// Gas gap
  // trigger gas
  auto gas = new TGeoVolume(Form("Gas %s", name),
                            new TGeoBBox(Form("%sGasBox", name), halfLength, halfHeight, kGasHalfThickness),
                            assertMedium(Medium::Gas));

  // resisitive electrod plate
  auto electrod = new TGeoVolume(Form("Electrod %s", name),
                                 new TGeoBBox(Form("%sElecBox", name), halfLength, halfHeight, kElectrodHalfThickness),
                                 assertMedium(Medium::Bakelite));

  /// Insulator
  auto insu = new TGeoVolume(Form("Insulator %s", name),
                             new TGeoBBox(Form("%sInsuBox", name), halfLength, halfHeight, kInsulatorHalfThickness),
                             assertMedium(Medium::Mylar));

  /// Strip plane
  // cooper foil
  auto copper = new TGeoVolume(Form("Copper %s", name),
                               new TGeoBBox(Form("%sCopperBox", name), halfLength, halfHeight, kCopperHalfThickness),
                               assertMedium(Medium::Copper));

  // mylar foil
  auto mylar = new TGeoVolume(Form("Mylar %s", name),
                              new TGeoBBox(Form("%sMylarBox", name), halfLength, halfHeight, kMylarHalfThickness),
                              assertMedium(Medium::Mylar));

  // styrofoam plane
  auto styro = new TGeoVolume(Form("Styrofoam %s", name),
                              new TGeoBBox(Form("%sStyroBox", name), halfLength, halfHeight, kStyrofoamHalfThickness),
                              assertMedium(Medium::Styrofoam));

  /// Stiffener plane
  // aluminium foil
  auto alu = new TGeoVolume(Form("Aluminium %s", name),
                            new TGeoBBox(Form("%sAluBox", name), halfLength, halfHeight, kAluminiumHalfThickness),
                            assertMedium(Medium::Aluminium));

  // nomex
  auto nomex = new TGeoVolume(Form("Nomex %s", name),
                              new TGeoBBox(Form("%sNomexBox", name), halfLength, halfHeight, kNomexHalfThickness),
                              assertMedium(Medium::Nomex));

  // change the volume shape if we are creating a "cut" RPC
  if (!strcmp(type, "cut")) {
    // dimensions of the cut
    double cutHalfLength = Constants::sLocalBoardWidth * Constants::sScaleFactors[iChamber] / 2.;
    double cutHalfHeight = Constants::sLocalBoardHeight * Constants::sScaleFactors[iChamber] / 2.;

    // position of the cut w.r.t the center of the RPC
    auto cutPos = new TGeoTranslation(Form("%sCutPos", name), cutHalfLength - halfLength, cutHalfHeight - halfHeight, 0.);
    cutPos->RegisterYourself();

    // for each volume, create a box and change the volume shape by extracting the cut shape
    new TGeoBBox(Form("%sGasCut", name), cutHalfLength, cutHalfHeight, kGasHalfThickness);
    gas->SetShape(new TGeoCompositeShape(Form("%sGasShape", name), Form("%sGasBox-%sGasCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sElecCut", name), cutHalfLength, cutHalfHeight, kElectrodHalfThickness);
    electrod->SetShape(new TGeoCompositeShape(Form("%sElecShape", name), Form("%sElecBox-%sElecCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sInsuCut", name), cutHalfLength, cutHalfHeight, kInsulatorHalfThickness);
    insu->SetShape(new TGeoCompositeShape(Form("%sInsuShape", name), Form("%sInsuBox-%sInsuCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sCopperCut", name), cutHalfLength, cutHalfHeight, kCopperHalfThickness);
    copper->SetShape(new TGeoCompositeShape(Form("%sCopperShape", name), Form("%sCopperBox-%sCopperCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sMylarCut", name), cutHalfLength, cutHalfHeight, kMylarHalfThickness);
    mylar->SetShape(new TGeoCompositeShape(Form("%sMylarShape", name), Form("%sMylarBox-%sMylarCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sStyroCut", name), cutHalfLength, cutHalfHeight, kStyrofoamHalfThickness);
    styro->SetShape(new TGeoCompositeShape(Form("%sStyroShape", name), Form("%sStyroBox-%sStyroCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sAluCut", name), cutHalfLength, cutHalfHeight, kAluminiumHalfThickness);
    alu->SetShape(new TGeoCompositeShape(Form("%sAluShape", name), Form("%sAluBox-%sAluCut:%sCutPos", name, name, name)));

    new TGeoBBox(Form("%sNomexCut", name), cutHalfLength, cutHalfHeight, kNomexHalfThickness);
    nomex->SetShape(new TGeoCompositeShape(Form("%sNomexShape", name), Form("%sNomexBox-%sNomexCut:%sCutPos", name, name, name)));
  }

  /// place all the layers in the RPC
  double halfThickness = kGasHalfThickness;
  rpc->AddNode(gas, 1);
  double z = halfThickness; // increment this value when adding a new layer

  halfThickness = kElectrodHalfThickness;
  z += halfThickness;
  rpc->AddNode(electrod, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(electrod, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kInsulatorHalfThickness;
  z += halfThickness;
  rpc->AddNode(insu, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(insu, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kCopperHalfThickness;
  z += halfThickness;
  rpc->AddNode(copper, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(copper, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kMylarHalfThickness;
  z += halfThickness;
  rpc->AddNode(mylar, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(mylar, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kStyrofoamHalfThickness;
  z += halfThickness;
  rpc->AddNode(styro, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(styro, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kMylarHalfThickness;
  z += halfThickness;
  rpc->AddNode(mylar, 3, new TGeoTranslation(0., 0., z));
  rpc->AddNode(mylar, 4, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kCopperHalfThickness;
  z += halfThickness;
  rpc->AddNode(copper, 3, new TGeoTranslation(0., 0., z));
  rpc->AddNode(copper, 4, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kAluminiumHalfThickness;
  z += halfThickness;
  rpc->AddNode(alu, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(alu, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kNomexHalfThickness;
  z += halfThickness;
  rpc->AddNode(nomex, 1, new TGeoTranslation(0., 0., z));
  rpc->AddNode(nomex, 2, new TGeoTranslation(0., 0., -z));
  z += halfThickness;

  halfThickness = kAluminiumHalfThickness;
  z += halfThickness;
  rpc->AddNode(alu, 3, new TGeoTranslation(0., 0., z));
  rpc->AddNode(alu, 4, new TGeoTranslation(0., 0., -z));

  return rpc;
}

TGeoVolume* createChamber(int iChamber)
{
  /// Function a trigger chamber, an assembly of RPCs (and services)

  auto chamber = new TGeoVolumeAssembly(Form("SC1%d", iChamber + 1));

  // create the 3 types of RPC
  auto shortRPC = createRPC("short", iChamber);
  auto longRPC = createRPC("long", iChamber);
  auto cutRPC = createRPC("cut", iChamber);

  // positions
  double x = 0., y = 0., z = Constants::sRPCZShift;

  // rotations
  auto rotX = new TGeoRotation("rotX", 90., 0., 90., -90., 180., 0.);
  auto rotY = new TGeoRotation("rotY", 90., 180., 90., 90., 180., 0.);
  auto rotZ = new TGeoRotation("rotZ", 90., 180., 90., 270., 0., 0.);

  // place them on the chamber
  for (int iRPC = 0; iRPC < Constants::sNRPCLines; iRPC++) {
    x = (iRPC == 0) ? Constants::sRPCShortCenterPos : Constants::sRPCCenterPos;
    x *= Constants::sScaleFactors[iChamber];
    y = kRPCypos[iRPC] * Constants::sScaleFactors[iChamber];

    switch (iRPC) {
      case 0: // short
        chamber->AddNode(shortRPC, 0, new TGeoTranslation(x, y, z));
        chamber->AddNode(shortRPC, 9, new TGeoCombiTrans(-x, -y, -z, rotY));
        break;
      case 1: // cut
        chamber->AddNode(cutRPC, 1, new TGeoTranslation(x, y, z));
        chamber->AddNode(cutRPC, 17, new TGeoCombiTrans(x, -y, z, rotX));
        break;
      case 8: // cut
        chamber->AddNode(cutRPC, 8, new TGeoCombiTrans(-x, y, z, rotY));
        chamber->AddNode(cutRPC, 10, new TGeoCombiTrans(-x, -y, z, rotZ));
        break;
      default: // long
        if (iRPC >= 5)
          x *= -1;
        chamber->AddNode(longRPC, iRPC, new TGeoTranslation(x, y, z));
        chamber->AddNode(longRPC, 2 * Constants::sNRPCLines - iRPC, new TGeoCombiTrans(x, -y, z, rotX));
        break;
    }

    z *= -1; // switch the z side for the next RPC placement
  }

  return chamber;
}

void createGeometry(TGeoVolume& topVolume)
{
  createMaterials();

  auto rot = new TGeoRotation("MID chamber inclination", 90., 0., 90. - Constants::sBeamAngle, 90., -Constants::sBeamAngle, 90.);

  // create and place the trigger chambers
  for (int iCh = 0; iCh < Constants::sNChambers; iCh++) {

    topVolume.AddNode(createChamber(iCh), 1, new TGeoCombiTrans(0., 0., Constants::sDefaultChamberZ[iCh], rot));
  }
}

//______________________________________________________________________________
std::vector<TGeoVolume*> getSensitiveVolumes()
{
  /// Create a vector containing the sensitive volume's name of the RPCs for the Detector class

  std::vector<TGeoVolume*> sensitiveVolumeNames;
  const char* type[3] = { "short", "cut", "long" };
  for (int i = 0; i < Constants::sNChambers; i++) {
    for (int j = 0; j < 3; j++) {
      auto vol = gGeoManager->GetVolume(Form("Gas %sRPC%d", type[j], i));

      if (!vol) {
        throw std::runtime_error(Form("could not get expected volume Gas %sRPC%d", type[j], i));
      } else {
        sensitiveVolumeNames.push_back(vol);
      }
    }
  }
  return sensitiveVolumeNames;
}

} // namespace mid
} // namespace o2
