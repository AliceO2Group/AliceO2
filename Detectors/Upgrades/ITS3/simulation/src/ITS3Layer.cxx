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

/// \file ITS3Layer.h
/// \brief Definition of the ITS3Layer class
/// \author Fabrizio Grosa <fgrosa@cern.ch>
/// \author felix.schlepper@cern.ch

#include "TGeoTube.h"
#include "TGeoVolume.h"

#include "CommonConstants/MathConstants.h"
#include "ITSBase/Specs.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Simulation/ITS3Layer.h"
#include "fairlogger/Logger.h"

namespace o2::its3
{

void ITS3Layer::init()
{
  // First we start by creating variables which we are reusing a couple of times.
  mR = constants::radii[mNLayer];
  mRmin = mR - constants::thickness / 2.;
  mRmax = mR + constants::thickness / 2.;
}

void ITS3Layer::createLayer(TGeoVolume* motherVolume, int layer)
{
  // Create one layer of ITS3 and attach it to the motherVolume.
  if (layer > 2) {
    LOGP(fatal, "Cannot create more than 3 layers!");
    return;
  }
  mNLayer = layer;

  init();
  createPixelArray();
  createTile();
  createRSU();
  createSegment();
  createChip();
  createCarbonForm();
  createLayerImpl();

  // Add it to motherVolume
  motherVolume->AddNode(mLayer, 0);
}

void ITS3Layer::createPixelArray()
{
  using namespace constants::pixelarray;
  // A pixel array is pure silicon and the sensitive part of our detector.
  // It will be segmented into a 440x144 matrix by the
  // SuperSegmentationAlpide.
  // Pixel Array is just a longer version of the biasing but starts in phi at
  // biasPhi2.
  double pixelArrayPhi1 =
    constants::tile::biasing::length / mRmin * constants::math::Rad2Deg;
  double pixelArrayPhi2 =
    length / mRmin * constants::math::Rad2Deg + pixelArrayPhi1;
  auto pixelArray = new TGeoTubeSeg(mRmin, mRmax, width / 2.,
                                    pixelArrayPhi1, pixelArrayPhi2);
  mPixelArray = new TGeoVolume(
    Form("pixelarray_%d", mNLayer) /* TODO change to correct name */,
    pixelArray,
    gGeoManager->GetMedium(material::MaterialNames[material::Silicon]));
  mPixelArray->SetLineColor(color);
  mPixelArray->RegisterYourself();
}

void ITS3Layer::createTile()
{
  using namespace constants::tile;
  // This functions creates a single Tile, which is the basic building block
  // of the chip. It consists of a pixelArray (sensitive area), biasing, power
  // switches and readout periphery (latter three are insensitive). We start
  // building the tile with the left upper edge of the biasing as center of
  // the tile’s z-coordinate axis.
  mTile = new TGeoVolumeAssembly(Form("tile_%d", mNLayer));
  mTile->VisibleDaughters();

  // Biasing
  auto zMoveBiasing = new TGeoTranslation(0, 0, +biasing::width / 2.);
  double biasPhi1 = 0;
  double biasPhi2 = biasing::length / mRmin * constants::math::Rad2Deg;
  auto biasing =
    new TGeoTubeSeg(mRmin, mRmax, biasing::width / 2, biasPhi1,
                    biasPhi2);
  auto biasingVol = new TGeoVolume(
    Form("biasing_%d", mNLayer), biasing,
    gGeoManager->GetMedium(material::MaterialNames[material::DeadZone]));
  biasingVol->SetLineColor(biasing::color);
  biasingVol->RegisterYourself();
  if (mVerbose) {
    std::cout << "Biasing:" << std::endl;
    biasingVol->InspectShape();
    biasingVol->InspectMaterial();
  }
  mTile->AddNode(biasingVol, 0, zMoveBiasing);

  // Pixel Array is just a longer version of the biasing but starts in phi at
  // biasPhi2.
  mTile->AddNode(mPixelArray, 0, zMoveBiasing);

  // The readout periphery is also on top of the pixel array but extrudes on +z a bit e.g. is wider.
  auto zMoveReadout = new TGeoTranslation(0, 0, +readout::width / 2.);
  double readoutPhi1 =
    constants::pixelarray::length / mRmin * constants::math::Rad2Deg + biasPhi2;
  double readoutPhi2 =
    readout::length / mRmin * constants::math::Rad2Deg + readoutPhi1;
  auto readout = new TGeoTubeSeg(mRmin, mRmax, readout::width / 2,
                                 readoutPhi1, readoutPhi2);
  auto readoutVol = new TGeoVolume(
    Form("readout_%d", mNLayer), readout,
    gGeoManager->GetMedium(material::MaterialNames[material::DeadZone]));
  readoutVol->SetLineColor(readout::color);
  readoutVol->RegisterYourself();
  if (mVerbose) {
    std::cout << "Readout:" << std::endl;
    readoutVol->InspectShape();
    readoutVol->InspectMaterial();
  }
  mTile->AddNode(readoutVol, 0, zMoveReadout);

  // Power Switches are on the side right side of the pixel array and biasing.
  auto zMovePowerSwitches = new TGeoTranslation(0, 0, +powerswitches::width / 2. + biasing::width);
  double powerPhi1 = 0;
  double powerPhi2 = powerswitches::length / mRmin * constants::math::Rad2Deg;
  auto powerSwitches = new TGeoTubeSeg(
    mRmin, mRmax, powerswitches::width / 2, powerPhi1, powerPhi2);
  auto powerSwitchesVol = new TGeoVolume(
    Form("powerswitches_%d", mNLayer), powerSwitches,
    gGeoManager->GetMedium(material::MaterialNames[material::DeadZone]));
  powerSwitchesVol->SetLineColor(powerswitches::color);
  if (mVerbose) {
    std::cout << "PowerSwitches:" << std::endl;
    powerSwitchesVol->InspectShape();
    powerSwitchesVol->InspectMaterial();
  }
  mTile->AddNode(powerSwitchesVol, 0, zMovePowerSwitches);

  if (mSubstrate) {
    // Create the substrate layer at the back of the tile.
    // TODO
  }
}

void ITS3Layer::createRSU()
{
  using namespace constants::rsu;
  // A Repeated Sensor Unit (RSU) is 12 Tiles + 4 Databackbones stichted together.
  mRSU = new TGeoVolumeAssembly(Form("rsu_%d", mNLayer));
  mRSU->VisibleDaughters();
  int nCopyRSU{0}, nCopyDB{0};

  // Create the DatabackBone
  // The Databackbone spans the whole phi of the tile.
  double dataBackbonePhi1 = 0;
  double dataBackbonePhi2 = databackbone::length / mRmin *
                            constants::math::Rad2Deg;
  auto dataBackbone = new TGeoTubeSeg(mRmin, mRmax, databackbone::width / 2.,
                                      dataBackbonePhi1,
                                      dataBackbonePhi2);
  auto dataBackboneVol = new TGeoVolume(
    Form("databackbone_%d", mNLayer), dataBackbone,
    gGeoManager->GetMedium(material::MaterialNames[material::DeadZone]));
  dataBackboneVol->SetLineColor(databackbone::color);
  dataBackboneVol->RegisterYourself();
  if (mVerbose) {
    std::cout << "DataBackbone:" << std::endl;
    dataBackboneVol->InspectShape();
    dataBackboneVol->InspectMaterial();
  }

  // Lower Left
  auto zMoveLL1 = new TGeoTranslation(0, 0, constants::tile::width);
  auto zMoveLL2 = new TGeoTranslation(0, 0, constants::tile::width * 2.);
  auto zMoveLLDB = new TGeoTranslation(0, 0, -databackbone::width / 2.);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, nullptr);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLL1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLL2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveLLDB);

  // Lower Right
  auto zMoveLR0 = new TGeoTranslation(0, 0, +width / 2.);
  auto zMoveLR1 = new TGeoTranslation(0, 0, constants::tile::width + width / 2.);
  auto zMoveLR2 = new TGeoTranslation(0, 0, constants::tile::width * 2. + width / 2.);
  auto zMoveLRDB = new TGeoTranslation(0, 0, -databackbone::width / 2. + width / 2.);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLR0);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLR1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLR2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveLRDB);

  // Rotation for top half
  double phi = length / mRmin * constants::math::Rad2Deg;
  auto rot = new TGeoRotation("", 0, 0, phi / 2.);

  // Upper Left
  auto zMoveUL1 = new TGeoCombiTrans(0, 0, constants::tile::width, rot);
  auto zMoveUL2 = new TGeoCombiTrans(0, 0, constants::tile::width * 2., rot);
  auto zMoveULDB = new TGeoCombiTrans(0, 0, -databackbone::width / 2., rot);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, rot);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUL1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUL2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveULDB);

  // Upper Right
  auto zMoveUR0 = new TGeoCombiTrans(0, 0, +width / 2., rot);
  auto zMoveUR1 = new TGeoCombiTrans(0, 0, constants::tile::width + width / 2., rot);
  auto zMoveUR2 = new TGeoCombiTrans(0, 0, constants::tile::width * 2. + width / 2., rot);
  auto zMoveURDB = new TGeoCombiTrans(0, 0, -databackbone::width / 2. + width / 2., rot);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUR0);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUR1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUR2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveURDB);
}

void ITS3Layer::createSegment()
{
  using namespace constants::segment;
  // A segment is 12 RSUs + left and right end cap. We place the first rsu
  // as z-coordinate center and attach to this. Hence, we will displace the
  // left end-cap to the left and the right to right.
  mSegment = new TGeoVolumeAssembly(Form("segment_%d", mNLayer));
  mSegment->VisibleDaughters();

  for (int i{0}; i < nRSUs; ++i) {
    auto zMove = new TGeoTranslation(0, 0, +i * constants::rsu::width + constants::rsu::databackbone::width);
    mSegment->AddNode(mRSU, i, zMove);
  }

  // LEC
  double lecPhi1 = 0;
  double lecPhi2 = lec::length / mRmin * constants::math::Rad2Deg;
  auto zMoveLEC = new TGeoTranslation(0, 0, -lec::width / 2.);
  auto lec =
    new TGeoTubeSeg(mRmin, mRmax, lec::width / 2., lecPhi1, lecPhi2);
  auto lecVol = new TGeoVolume(
    Form("lec_%d", mNLayer), lec,
    gGeoManager->GetMedium(material::MaterialNames[material::DeadZone]));
  lecVol->SetLineColor(lec::color);
  lecVol->RegisterYourself();
  if (mVerbose) {
    std::cout << "LEC:" << std::endl;
    lecVol->InspectShape();
    lecVol->InspectMaterial();
  }
  mSegment->AddNode(lecVol, 0, zMoveLEC);

  // REC; reuses lecPhi1,2
  auto zMoveREC = new TGeoTranslation(0, 0, nRSUs * constants::rsu::width + rec::width / 2.);
  auto rec =
    new TGeoTubeSeg(mRmin, mRmax, rec::width / 2., lecPhi1, lecPhi2);
  auto recVol = new TGeoVolume(
    Form("rec_%d", mNLayer), rec,
    gGeoManager->GetMedium(material::MaterialNames[material::DeadZone]));
  recVol->SetLineColor(rec::color);
  recVol->RegisterYourself();
  if (mVerbose) {
    std::cout << "REC:" << std::endl;
    recVol->InspectShape();
    recVol->InspectMaterial();
  }
  mSegment->AddNode(recVol, 0, zMoveREC);
}

void ITS3Layer::createChip()
{

  // A HalfLayer is composed out of multiple segment stitched together along
  // rphi.
  mChip = new TGeoVolumeAssembly(Form("chip_%d", mNLayer));
  mChip->VisibleDaughters();

  for (int i{0}; i < constants::nSegments[mNLayer]; ++i) {
    double phiOffset = constants::segment::length / mRmin * constants::math::Rad2Deg;
    auto rot = new TGeoRotation("", 0, 0, phiOffset * i);
    mChip->AddNode(mSegment, i, rot);
  }
}

void ITS3Layer::createCarbonForm()
{
  mCarbonForm = new TGeoVolumeAssembly(Form("carbonform_%d", mNLayer));
  mCarbonForm->VisibleDaughters();
  mCarbonForm->AddNode(mChip, 0);
  // TODO
}

void ITS3Layer::createLayerImpl()
{
  // At long last a single layer... A layer is two HalfLayers (duuhhh) but
  // we have to take care of the equatorial gap. So both half layers will be
  // offset slightly by rotating in phi the upper HalfLayer and negative phi
  // the other one.
  mLayer = new TGeoVolumeAssembly(Form("layer_%d", mNLayer));
  mLayer->VisibleDaughters();

  // The offset is the right angle triangle of the middle radius with the
  // transverse axis.
  double phiOffset = std::asin(constants::equatorialGap / mR) * constants::math::Rad2Deg;
  // double phiOffset = constants::equatorialGap / mRmin / 2.;
  auto rotTop = new TGeoRotation("", 0, 0, +phiOffset);
  auto rotBot = new TGeoRotation("", 0, 0, phiOffset + 180);

  mLayer->AddNode(mCarbonForm, 0, rotTop);
  mLayer->AddNode(mCarbonForm, 1, rotBot);
}
} // namespace o2::its3
