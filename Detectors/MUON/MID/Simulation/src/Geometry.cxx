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

/// \file   MID/Simulation/src/Geometry.cxx
/// \brief  Implementation of the trigger-stations geometry
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   19 june 2018

#include "MIDSimulation/Geometry.h"

#include <sstream>

#include "Materials.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/GeometryParameters.h"

#include <TGeoVolume.h>
#include <TGeoManager.h>
#include <TGeoShape.h>
#include <TGeoCompositeShape.h>
#include <TGeoTube.h>

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

/// Service parameters
// vertical support
const float kVerticalSupportHalfExtDim[] = {1.5, 311., 1.5};
const float kVerticalSupportHalfIntDim[] = {1.2, 311., 1.2};
const float kVerticalSupportXPos[] = {61.45, 122.45, 192.95, 236.95};

// horizontal support
const float kHorizontalSupportHalfExtDim[] = {96.775, 2., 3.};
const float kHorizontalSupportHalfIntDim[] = {96.775, 1.9, 2.8};
const double kHorizontalSupportPos[] = {geoparams::RPCCenterPos + geoparams::RPCHalfLength - kHorizontalSupportHalfExtDim[0], 17., kVerticalSupportHalfExtDim[2] + kHorizontalSupportHalfExtDim[2]};

TGeoVolume* createVerticalSupport(int iChamber)
{
  /// Function creating a vertical support, an aluminium rod

  auto supp = new TGeoVolume(Form("Vertical support chamber %d", iChamber), new TGeoBBox(Form("VertSuppBox%d", iChamber), kVerticalSupportHalfExtDim[0], kVerticalSupportHalfExtDim[1] * geoparams::ChamberScaleFactors[iChamber], kVerticalSupportHalfExtDim[2]), assertMedium(Medium::Aluminium));

  new TGeoBBox(Form("VertSuppCut%d", iChamber), kVerticalSupportHalfIntDim[0], kVerticalSupportHalfIntDim[1] * geoparams::ChamberScaleFactors[iChamber], kVerticalSupportHalfIntDim[2]);

  supp->SetShape(new TGeoCompositeShape(Form("VertSuppCut%d", iChamber), Form("VertSuppBox%d-VertSuppCut%d", iChamber, iChamber)));

  return supp;
}

TGeoVolume* createHorizontalSupport(int iChamber)
{
  /// Function creating a horizontal support, an aluminium rod

  auto supp = new TGeoVolume(Form("Horizontal support chamber %d", iChamber), new TGeoBBox(Form("HoriSuppBox%d", iChamber), kHorizontalSupportHalfExtDim[0] * geoparams::ChamberScaleFactors[iChamber], kHorizontalSupportHalfExtDim[1], kHorizontalSupportHalfExtDim[2]), assertMedium(Medium::Aluminium));

  new TGeoBBox(Form("HoriSuppCut%d", iChamber), kHorizontalSupportHalfIntDim[0] * geoparams::ChamberScaleFactors[iChamber], kHorizontalSupportHalfIntDim[1], kHorizontalSupportHalfIntDim[2]);

  supp->SetShape(new TGeoCompositeShape(Form("HoriSuppCut%d", iChamber), Form("HoriSuppBox%d-HoriSuppCut%d", iChamber, iChamber)));

  return supp;
}

TGeoVolume* createRPC(geoparams::RPCtype type, int iChamber)
{
  /// Function building a resisitive plate chamber (RPC), the detection element of the MID, of a given type and for the given chamber number.

  auto sname = getRPCVolumeName(type, iChamber);
  auto name = sname.c_str();

  auto rpc = new TGeoVolumeAssembly(name);

  // get the dimensions from MIDBase/Constants
  double halfLength = (type == geoparams::RPCtype::Short) ? geoparams::RPCShortHalfLength : geoparams::RPCHalfLength;
  halfLength *= geoparams::ChamberScaleFactors[iChamber];
  double halfHeight = geoparams::getRPCHalfHeight(iChamber);

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
  if (type == geoparams::RPCtype::TopCut || type == geoparams::RPCtype::BottomCut) {
    // dimensions of the cut
    double cutHalfLength = geoparams::getLocalBoardWidth(iChamber) / 2.;
    double cutHalfHeight = geoparams::getLocalBoardHeight(iChamber) / 2.;

    bool isTopCut = (type == geoparams::RPCtype::TopCut);
    const char* cutName = Form("%sCut%s", (isTopCut) ? "top" : "bottom", name);

    // position of the cut w.r.t the center of the RPC
    auto cutPos = new TGeoTranslation(Form("%sPos", cutName), cutHalfLength - halfLength, (isTopCut) ? halfHeight - cutHalfHeight : cutHalfHeight - halfHeight, 0.);
    cutPos->RegisterYourself();

    // for each volume, create a box and change the volume shape by extracting the cut shape
    new TGeoBBox(Form("%sGasCut", name), cutHalfLength, cutHalfHeight, 2 * kGasHalfThickness);
    gas->SetShape(new TGeoCompositeShape(Form("%sGasShape", name), Form("%sGasBox-%sGasCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sElecCut", name), cutHalfLength, cutHalfHeight, 2 * kElectrodHalfThickness);
    electrod->SetShape(new TGeoCompositeShape(Form("%sElecShape", name), Form("%sElecBox-%sElecCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sInsuCut", name), cutHalfLength, cutHalfHeight, 2 * kInsulatorHalfThickness);
    insu->SetShape(new TGeoCompositeShape(Form("%sInsuShape", name), Form("%sInsuBox-%sInsuCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sCopperCut", name), cutHalfLength, cutHalfHeight, 2 * kCopperHalfThickness);
    copper->SetShape(new TGeoCompositeShape(Form("%sCopperShape", name), Form("%sCopperBox-%sCopperCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sMylarCut", name), cutHalfLength, cutHalfHeight, 2 * kMylarHalfThickness);
    mylar->SetShape(new TGeoCompositeShape(Form("%sMylarShape", name), Form("%sMylarBox-%sMylarCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sStyroCut", name), cutHalfLength, cutHalfHeight, 2 * kStyrofoamHalfThickness);
    styro->SetShape(new TGeoCompositeShape(Form("%sStyroShape", name), Form("%sStyroBox-%sStyroCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sAluCut", name), cutHalfLength, cutHalfHeight, 2 * kAluminiumHalfThickness);
    alu->SetShape(new TGeoCompositeShape(Form("%sAluShape", name), Form("%sAluBox-%sAluCut:%sPos", name, name, cutName)));

    new TGeoBBox(Form("%sNomexCut", name), cutHalfLength, cutHalfHeight, 2 * kNomexHalfThickness);
    nomex->SetShape(new TGeoCompositeShape(Form("%sNomexShape", name), Form("%sNomexBox-%sNomexCut:%sPos", name, name, cutName)));
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

TGeoMatrix* getTransformation(const ROOT::Math::Transform3D& matrix)
{
  /// Converts Transform3D into TGeoMatrix
  double xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz;
  matrix.GetComponents(xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz);
  double vect[3] = {dx, dy, dz};
  double rotMatrix[9] = {xx, xy, xz, yx, yy, yz, zx, zy, zz};
  TGeoHMatrix* geoMatrix = new TGeoHMatrix("Transformation");
  geoMatrix->SetTranslation(vect);
  geoMatrix->SetRotation(rotMatrix);
  return geoMatrix;
}

TGeoVolume* createChamber(int iChamber)
{
  /// Function creating a trigger chamber, an assembly of RPCs (and services)

  auto chamber = new TGeoVolumeAssembly(geoparams::getChamberVolumeName(iChamber).c_str());

  double scale = geoparams::ChamberScaleFactors[iChamber];

  // create the service volumes
  auto vertSupp = createVerticalSupport(iChamber);
  auto horiSupp = createHorizontalSupport(iChamber);

  // create the 4 types of RPC
  auto longRPC = createRPC(geoparams::RPCtype::Long, iChamber);
  auto bottomCutRPC = createRPC(geoparams::RPCtype::BottomCut, iChamber);
  auto topCutRPC = createRPC(geoparams::RPCtype::TopCut, iChamber);
  auto shortRPC = createRPC(geoparams::RPCtype::Short, iChamber);

  // for node counting
  int iHoriSuppNode = 0, iVertSuppNode = 0;

  // place the volumes on both side of the chamber
  for (int iside = 0; iside < 2; iside++) {

    bool isRight = (iside == 0);
    double xSign = (isRight) ? 1. : -1.;

    // place 4 vertical supports per side
    for (int i = 0; i < 4; i++) {
      chamber->AddNode(vertSupp, iVertSuppNode++, new TGeoTranslation(xSign * kVerticalSupportXPos[i] * scale, 0., 0.));
    }

    // place the RPCs
    for (int iRPC = 0; iRPC < detparams::NRPCLines; iRPC++) {

      double x = xSign * geoparams::getRPCCenterPosX(iChamber, iRPC);
      double zSign = (iRPC % 2 == 0) ? 1. : -1.;

      if (!isRight) {
        zSign *= -1.;
      }
      double z = zSign * geoparams::RPCZShift;
      double y = 2 * geoparams::getRPCHalfHeight(iChamber) * (iRPC - 4) / (1 - (z / geoparams::DefaultChamberZ[0]));

      // ID convention (from bottom to top of the chamber) : long, long, long, cut, short, cut, long, long, long
      TGeoVolume* rpc = nullptr;
      switch (iRPC) {
        case 4: // short
          rpc = shortRPC;
          break;
        case 5: // cut (bottom)
          rpc = bottomCutRPC;
          break;
        case 3: // cut (top)
          rpc = topCutRPC;
          break;
        default: // long
          rpc = longRPC;
          break;
      }

      int deId = detparams::getDEId(isRight, iChamber, iRPC);
      chamber->AddNode(rpc, deId, getTransformation(getDefaultRPCTransform(isRight, iChamber, iRPC)));

      // place 3 horizontal supports behind the RPC (and the vertical rods)
      x = xSign * kHorizontalSupportPos[0] * scale;
      z = -zSign * kHorizontalSupportPos[2];
      for (int i = 0; i < 3; i++) {
        chamber->AddNode(horiSupp, iHoriSuppNode++, new TGeoTranslation(x, y + (i - 1) * kHorizontalSupportPos[1] * scale, z));
      }

    } // end of the loop over the number of RPC lines

  } // end of the side loop

  return chamber;
}

/// Magnet geometry variant selector
enum class MagnetVariant {
  AluminiumWalls, ///< 11 cm cryostat, Al inner/outer walls
  SteelWalls      ///< 10 cm cryostat, Fe inner/outer walls
};

/// Creates the MID magnet/cryostat geometry
/// Port of GEANT4 simulation by Ian Perez Garcia (ICN-UNAM)
/// Reference: github.com/IanPG/MID-Geometry-Studies
void createMagnetGeometry(TGeoVolume& topVolume,
                          MagnetVariant variant = MagnetVariant::AluminiumWalls)
{
  const float R_cryostat_inner = 140.0f;
  const float R_cryostat_outer = 200.0f;
  const float R_coil_inner = 160.0f;
  const float magnetHalfLength = 400.0f;

  const float thick_actual_coil = 4.8f;
  const float thick_mli = 0.2f;
  const float thick_coil_support = 2.0f;

  float thick_inner_wall;
  float thick_outer_wall;
  int wallMedium;
  const char* variantTag;

  if (variant == MagnetVariant::AluminiumWalls) {
    thick_inner_wall = 2.5f;
    thick_outer_wall = 1.5f;
    wallMedium = Medium::Aluminium;
    variantTag = "Al";
  } else {
    thick_inner_wall = 1.5f;
    thick_outer_wall = 1.5f;
    wallMedium = Medium::Iron;
    variantTag = "Steel";
  }

  const float R_inner_wall_outer = R_cryostat_inner + thick_inner_wall;
  const float R_coil_outer = R_coil_inner + thick_actual_coil;
  const float R_mli_inner = R_coil_outer;
  const float R_mli_outer = R_mli_inner + thick_mli;
  const float R_coil_support_inner = R_mli_outer;
  const float R_coil_support_outer = R_coil_support_inner + thick_coil_support;
  const float R_outer_wall_inner = R_cryostat_outer - thick_outer_wall;

  // Mother volume (contains all cryostat layers)
  auto magnetMotherShape = new TGeoTube(Form("MIDMagnetMother_%s_S", variantTag),
                                        R_cryostat_inner, R_cryostat_outer, magnetHalfLength);
  auto magnetMotherVol = new TGeoVolume(Form("MIDMagnetMother_%s", variantTag),
                                        magnetMotherShape, assertMedium(Medium::Vacuum));
  magnetMotherVol->SetVisibility(kFALSE);
  topVolume.AddNode(magnetMotherVol, 1, new TGeoTranslation(0., 0., -1155.));

  // Layer 1: Inner wall (Al or Fe)
  auto innerWallVol = new TGeoVolume(Form("MIDInnerWall_%s", variantTag),
                                     new TGeoTube(Form("MIDInnerWall_%s_S", variantTag),
                                                  R_cryostat_inner, R_inner_wall_outer, magnetHalfLength),
                                     assertMedium(wallMedium));
  innerWallVol->SetLineColor((variant == MagnetVariant::AluminiumWalls) ? kCyan + 1 : kRed + 1);
  magnetMotherVol->AddNode(innerWallVol, 1, nullptr);

  // Layer 2: Inner vacuum gap
  auto vacGap1Vol = new TGeoVolume(Form("MIDVacGap1_%s", variantTag),
                                   new TGeoTube(Form("MIDVacGap1_%s_S", variantTag),
                                                R_inner_wall_outer, R_coil_inner, magnetHalfLength),
                                   assertMedium(Medium::Vacuum));
  magnetMotherVol->AddNode(vacGap1Vol, 1, nullptr);

  // Layer 3: Winding Pack (NbTi+Cu+Al, density=2.96 g/cm3)
  auto coilVol = new TGeoVolume(Form("MIDCoil_%s", variantTag),
                                new TGeoTube(Form("MIDCoil_%s_S", variantTag),
                                             R_coil_inner, R_coil_outer, magnetHalfLength),
                                assertMedium(Medium::WindingPack));
  coilVol->SetLineColor(kRed);
  magnetMotherVol->AddNode(coilVol, 1, nullptr);

  // Layer 4: MLI - Multi-Layer Insulation (2mm Al)
  auto mliVol = new TGeoVolume(Form("MIDMLI_%s", variantTag),
                               new TGeoTube(Form("MIDMLI_%s_S", variantTag),
                                            R_mli_inner, R_mli_outer, magnetHalfLength),
                               assertMedium(Medium::Aluminium));
  mliVol->SetLineColor(kYellow);
  magnetMotherVol->AddNode(mliVol, 1, nullptr);

  // Layer 5: A-5083 Support cylinder (20mm Al)
  auto supportVol = new TGeoVolume(Form("MIDCoilSupport_%s", variantTag),
                                   new TGeoTube(Form("MIDCoilSupport_%s_S", variantTag),
                                                R_coil_support_inner, R_coil_support_outer, magnetHalfLength),
                                   assertMedium(Medium::Aluminium));
  supportVol->SetLineColor(kBlue - 7);
  magnetMotherVol->AddNode(supportVol, 1, nullptr);

  // Layer 6: Outer vacuum gap
  auto vacGap2Vol = new TGeoVolume(Form("MIDVacGap2_%s", variantTag),
                                   new TGeoTube(Form("MIDVacGap2_%s_S", variantTag),
                                                R_coil_support_outer, R_outer_wall_inner, magnetHalfLength),
                                   assertMedium(Medium::Vacuum));
  magnetMotherVol->AddNode(vacGap2Vol, 1, nullptr);

  // Layer 7: Outer wall (Al or Fe)
  auto outerWallVol = new TGeoVolume(Form("MIDOuterWall_%s", variantTag),
                                     new TGeoTube(Form("MIDOuterWall_%s_S", variantTag),
                                                  R_outer_wall_inner, R_cryostat_outer, magnetHalfLength),
                                     assertMedium(wallMedium));
  outerWallVol->SetLineColor((variant == MagnetVariant::AluminiumWalls) ? kCyan + 1 : kRed + 1);
  magnetMotherVol->AddNode(outerWallVol, 1, nullptr);
}

void createGeometry(TGeoVolume& topVolume)
{
  createMaterials();

  // Add magnet/cryostat geometry (Francisco Esquivel, June 2026)
  printf("[MID] Calling createMagnetGeometry...\n");
  createMagnetGeometry(topVolume, MagnetVariant::AluminiumWalls);
  printf("[MID] createMagnetGeometry done. Top volume nodes: %d\n", topVolume.GetNdaughters());

  // create and place the trigger chambers
  for (int iCh = 0; iCh < detparams::NChambers; iCh++) {

    topVolume.AddNode(createChamber(iCh), 1, getTransformation(getDefaultChamberTransform(iCh)));
  }
}

//______________________________________________________________________________
std::vector<TGeoVolume*> getSensitiveVolumes()
{
  /// Create a vector containing the sensitive volume's name of the RPCs for the Detector class

  std::vector<TGeoVolume*> sensitiveVolumeNames;
  std::vector<geoparams::RPCtype> types = {geoparams::RPCtype::Long, geoparams::RPCtype::BottomCut, geoparams::RPCtype::TopCut, geoparams::RPCtype::Short};
  for (int ich = 0; ich < detparams::NChambers; ++ich) {
    for (auto& type : types) {

      auto name = Form("Gas %s", getRPCVolumeName(type, ich).c_str());
      auto vol = gGeoManager->GetVolume(name);

      if (!vol) {
        throw std::runtime_error(Form("could not get expected volume %s", name));
      } else {
        sensitiveVolumeNames.push_back(vol);
      }
    }
  }
  return sensitiveVolumeNames;
}

} // namespace mid
} // namespace o2
