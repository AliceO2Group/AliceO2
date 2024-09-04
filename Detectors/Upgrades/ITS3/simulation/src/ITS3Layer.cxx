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
/// \author felix.schlepper@cern.ch
/// \author chunzheng.wang@cern.ch

#include "TGeoTube.h"
#include "TGeoVolume.h"
#include "TGeoCompositeShape.h"

#include "CommonConstants/MathConstants.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SpecsV2.h"
#include "ITS3Simulation/ITS3Layer.h"
#include "fairlogger/Logger.h"

namespace o2m = o2::constants::math;
namespace its3c = o2::its3::constants;

namespace o2::its3
{
using its3TGeo = o2::its::GeometryTGeo;

void ITS3Layer::init()
{
  mR = its3c::radii[mNLayer];
  mRmin = its3c::radiiInner[mNLayer];
  mRmax = its3c::radiiOuter[mNLayer];
}

void ITS3Layer::getMaterials(bool create)
{
  if (gGeoManager == nullptr) {
    LOGP(fatal, "gGeoManager not initalized!");
  }

  mSilicon = getMaterial("IT3_SI$", create);
  mAir = getMaterial("IT3_AIR$", create);
  mCarbon = getMaterial("IT3_CARBON$", create);
}

TGeoMedium* ITS3Layer::getMaterial(const char* matName, bool create)
{
  auto mat = gGeoManager->GetMedium(matName);
  if (mat == nullptr) {
    if (!create) {
      LOGP(fatal, "Cannot get medium {}", matName);
    } else { // create dummy
      auto matDummy = gGeoManager->GetMaterial("MAT_DUMMY$");
      if (matDummy == nullptr) {
        LOGP(info, "Created Dummy material");
        matDummy = new TGeoMaterial("MAT_DUMMY$", 26.98, 13, 2.7);
      }
      mat = new TGeoMedium(matName, 1, matDummy);
      LOGP(info, "Created medium {}", matName);
    }
  }
  return mat;
}

void ITS3Layer::createLayer(TGeoVolume* motherVolume)
{
  // Create one layer of ITS3 and attach it to the motherVolume.
  getMaterials();
  createLayerImpl();
  mBuilt = true;

  LOGP(info, "ITS3-Layer: Created Layer {} with mR={} (minR={}, maxR={})", mNLayer, mR, mRmin, mRmax);
  if (motherVolume == nullptr) {
    return;
  }
  // Add it to motherVolume
  LOGP(debug, "  `-> Attaching to motherVolume '{}'", motherVolume->GetName());
  auto* trans = new TGeoTranslation(0, 0, -constants::segment::lengthSensitive / 2.);
  motherVolume->AddNode(mLayer, 0, trans);
}

void ITS3Layer::createPixelArray()
{
  if (mPixelArray != nullptr) {
    return;
  }
  // A pixel array is pure silicon and the sensitive part of our detector.
  // It will be segmented into a 442x156 matrix by the
  // SuperSegmentationAlpide.
  // Pixel Array is just a longer version of the biasing but starts in phi at
  // biasPhi2.
  using namespace its3c::pixelarray;
  double pixelArrayPhi1 = constants::tile::readout::width / mR * o2m::Rad2Deg;
  double pixelArrayPhi2 = width / mR * o2m::Rad2Deg + pixelArrayPhi1;
  auto pixelArray = new TGeoTubeSeg(mRmin, mRmax, length / 2.,
                                    pixelArrayPhi1, pixelArrayPhi2);
  mPixelArray = new TGeoVolume(its3TGeo::getITS3PixelArrayPattern(mNLayer), pixelArray, mSilicon);
  mPixelArray->SetLineColor(color);
  mPixelArray->RegisterYourself();
}

void ITS3Layer::createTile()
{
  if (mTile != nullptr) {
    return;
  } else {
    createPixelArray();
  }
  // This functions creates a single Tile, which is the basic building block
  // of the chip. It consists of a pixelArray (sensitive area), biasing, power
  // switches and readout periphery (latter three are insensitive).
  // We construct the Tile such that the PixelArray is in the z-middle
  using namespace constants::tile;
  mTile = new TGeoVolumeAssembly(its3TGeo::getITS3TilePattern(mNLayer));
  mTile->VisibleDaughters();

  // The readout periphery is also on top of the pixel array but extrudes on +z a bit e.g. is wider.
  auto zMoveReadout = new TGeoTranslation(0, 0, +powerswitches::length / 2.);
  double readoutPhi1 = 0;
  double readoutPhi2 = readout::width / mR * o2m::Rad2Deg;
  auto readout = new TGeoTubeSeg(mRmin, mRmax, readout::length / 2, readoutPhi1, readoutPhi2);
  auto readoutVol = new TGeoVolume(Form("readout%d", mNLayer), readout, mSilicon);
  readoutVol->SetLineColor(readout::color);
  readoutVol->RegisterYourself();
  mTile->AddNode(readoutVol, 0, zMoveReadout);

  // Pixel Array is just a longer version of the biasing but starts in phi at
  // biasPhi2.
  mTile->AddNode(mPixelArray, 0);

  // Biasing
  double biasPhi1 = constants::pixelarray::width / mR * o2m::Rad2Deg + readoutPhi2;
  double biasPhi2 = biasing::width / mR * o2m::Rad2Deg + biasPhi1;
  auto biasing = new TGeoTubeSeg(mRmin, mRmax, biasing::length / 2, biasPhi1, biasPhi2);
  auto biasingVol = new TGeoVolume(Form("biasing%d", mNLayer), biasing, mSilicon);
  biasingVol->SetLineColor(biasing::color);
  biasingVol->RegisterYourself();
  mTile->AddNode(biasingVol, 0);

  // Power Switches are on the side right side of the pixel array and biasing.
  auto zMovePowerSwitches = new TGeoTranslation(0, 0, +powerswitches::length / 2. + constants::pixelarray::length / 2.);
  double powerPhi1 = readoutPhi2;
  double powerPhi2 = powerswitches::width / mR * o2m::Rad2Deg + powerPhi1;
  auto powerSwitches = new TGeoTubeSeg(mRmin, mRmax, powerswitches::length / 2, powerPhi1, powerPhi2);
  auto powerSwitchesVol = new TGeoVolume(Form("powerswitches%d", mNLayer), powerSwitches, mSilicon);
  powerSwitchesVol->SetLineColor(powerswitches::color);
  powerSwitchesVol->RegisterYourself();
  mTile->AddNode(powerSwitchesVol, 0, zMovePowerSwitches);
}

void ITS3Layer::createRSU()
{
  if (mRSU != nullptr) {
    return;
  } else {
    createTile();
  }
  // A Repeated Sensor Unit (RSU) is 12 Tiles + 4 Databackbones stichted together.
  using namespace constants::rsu;
  mRSU = new TGeoVolumeAssembly(its3TGeo::getITS3RSUPattern(mNLayer));
  mRSU->VisibleDaughters();
  int nCopyRSU{0}, nCopyDB{0};

  // Create the DatabackBone
  // The Databackbone spans the whole phi of the tile.
  double dataBackbonePhi1 = 0;
  double dataBackbonePhi2 = databackbone::width / mR * o2m::Rad2Deg;
  auto dataBackbone = new TGeoTubeSeg(mRmin, mRmax, databackbone::length / 2., dataBackbonePhi1, dataBackbonePhi2);
  auto dataBackboneVol = new TGeoVolume(Form("databackbone%d", mNLayer), dataBackbone, mSilicon);
  dataBackboneVol->SetLineColor(databackbone::color);
  dataBackboneVol->RegisterYourself();

  // Lower Left
  auto zMoveLL1 = new TGeoTranslation(0, 0, constants::tile::length);
  auto zMoveLL2 = new TGeoTranslation(0, 0, constants::tile::length * 2.);
  auto zMoveLLDB = new TGeoTranslation(0, 0, -databackbone::length / 2. - constants::pixelarray::length / 2.);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, nullptr);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLL1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLL2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveLLDB);

  // Lower Right
  auto zMoveLR0 = new TGeoTranslation(0, 0, +length / 2.);
  auto zMoveLR1 = new TGeoTranslation(0, 0, constants::tile::length + length / 2.);
  auto zMoveLR2 = new TGeoTranslation(0, 0, constants::tile::length * 2. + length / 2.);
  auto zMoveLRDB = new TGeoTranslation(0, 0, -databackbone::length / 2. + length / 2. - constants::pixelarray::length / 2.);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLR0);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLR1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveLR2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveLRDB);

  // Rotation for top half and vertical mirroring
  double phi = width / mR * o2m::Rad2Deg;
  auto rot = new TGeoRotation("", 0, 0, -phi);
  rot->ReflectY(true);

  // Upper Left
  auto zMoveUL1 = new TGeoCombiTrans(0, 0, constants::tile::length, rot);
  auto zMoveUL2 = new TGeoCombiTrans(0, 0, constants::tile::length * 2., rot);
  auto zMoveULDB = new TGeoCombiTrans(0, 0, -databackbone::length / 2. - constants::pixelarray::length / 2., rot);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, rot);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUL1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUL2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveULDB);

  // Upper Right
  auto zMoveUR0 = new TGeoCombiTrans(0, 0, +length / 2., rot);
  auto zMoveUR1 = new TGeoCombiTrans(0, 0, constants::tile::length + length / 2., rot);
  auto zMoveUR2 = new TGeoCombiTrans(0, 0, constants::tile::length * 2. + length / 2., rot);
  auto zMoveURDB = new TGeoCombiTrans(0, 0, -databackbone::length / 2. + length / 2. - constants::pixelarray::length / 2., rot);
  // Lets attach the tiles to the QS.
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUR0);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUR1);
  mRSU->AddNode(mTile, nCopyRSU++, zMoveUR2);
  mRSU->AddNode(dataBackboneVol, nCopyDB++, zMoveURDB);
}

void ITS3Layer::createSegment()
{
  if (mSegment != nullptr) {
    return;
  } else {
    createRSU();
  }
  // A segment is 12 RSUs + left and right end cap. We place the first rsu
  // as z-coordinate center and attach to this. Hence, we will displace the
  // left end-cap to the left and the right to right.
  using namespace constants::segment;
  mSegment = new TGeoVolumeAssembly(its3TGeo::getITS3SegmentPattern(mNLayer));
  mSegment->VisibleDaughters();

  for (size_t i{0}; i < nRSUs; ++i) {
    auto zMove = new TGeoTranslation(0, 0, +i * constants::rsu::length + constants::rsu::databackbone::length + constants::pixelarray::length / 2.);
    mSegment->AddNode(mRSU, i, zMove);
  }

  // LEC
  double lecPhi1 = 0;
  double lecPhi2 = lec::width / mR * o2m::Rad2Deg;
  auto zMoveLEC = new TGeoTranslation(0, 0, -lec::length / 2.);
  auto lec =
    new TGeoTubeSeg(mRmin, mRmax, lec::length / 2., lecPhi1, lecPhi2);
  auto lecVol = new TGeoVolume(Form("lec%d", mNLayer), lec, mSilicon);
  lecVol->SetLineColor(lec::color);
  lecVol->RegisterYourself();
  mSegment->AddNode(lecVol, 0, zMoveLEC);

  // REC; reuses lecPhi1,2
  auto zMoveREC = new TGeoTranslation(0, 0, nRSUs * constants::rsu::length + rec::length / 2.);
  auto rec =
    new TGeoTubeSeg(mRmin, mRmax, rec::length / 2., lecPhi1, lecPhi2);
  auto recVol = new TGeoVolume(Form("rec%d", mNLayer), rec, mSilicon);
  recVol->SetLineColor(rec::color);
  recVol->RegisterYourself();
  mSegment->AddNode(recVol, 0, zMoveREC);
}

void ITS3Layer::createChip()
{
  if (mChip != nullptr) {
    return;
  } else {
    createSegment();
  }
  // A HalfLayer is composed out of multiple segment stitched together along
  // rphi.
  mChip = new TGeoVolumeAssembly(its3TGeo::getITS3ChipPattern(mNLayer));
  mChip->VisibleDaughters();

  for (unsigned int i{0}; i < constants::nSegments[mNLayer]; ++i) {
    double phiOffset = constants::segment::width / mR * o2m::Rad2Deg;
    auto rot = new TGeoRotation("", 0, 0, phiOffset * i);
    mChip->AddNode(mSegment, i, rot);
  }
}

void ITS3Layer::createCarbonForm()
{
  if (mCarbonForm != nullptr) {
    return;
  } else {
    createChip();
  }
  // TODO : Waiting for the further information from WP5(Corrado)
  using namespace constants::carbonfoam;
  mCarbonForm = new TGeoVolumeAssembly(its3TGeo::getITS3CarbonFormPattern(mNLayer));
  mCarbonForm->VisibleDaughters();
  double dRadius = -1;
  if (mNLayer < 2) {
    dRadius = constants::radii[mNLayer + 1] - constants::radii[mNLayer] - constants::thickness;
  } else {
    dRadius = 0.7; // TODO: lack of carbon foam radius for layer 2, use 0.7mm as a temporary value
  }
  double phiSta = edgeBetwChipAndFoam / (0.5 * constants::radii[mNLayer + 1] + constants::radii[mNLayer]) * o2m::Rad2Deg;
  double phiEnd = (constants::nSegments[mNLayer] * constants::segment::width) / constants::radii[mNLayer] * o2m::Rad2Deg - phiSta;
  double phiLongeronsCover = longeronsWidth / (0.5 * constants::radii[mNLayer + 1] + constants::radii[mNLayer]) * o2m::Rad2Deg;

  // H-rings foam
  auto HringC = new TGeoTubeSeg(Form("HringC%d", mNLayer), mRmax, mRmax + dRadius, HringLength / 2., phiSta, phiEnd);
  auto HringA = new TGeoTubeSeg(Form("HringA%d", mNLayer), mRmax, mRmax + dRadius, HringLength / 2., phiSta, phiEnd);
  auto HringCWithHoles = getHringShape(HringC);
  auto HringAWithHoles = getHringShape(HringA);
  auto HringCVol = new TGeoVolume(Form("hringC%d", mNLayer), HringCWithHoles, mCarbon);
  HringCVol->SetLineColor(color);
  auto HringAVol = new TGeoVolume(Form("hringA%d", mNLayer), HringAWithHoles, mCarbon);
  HringAVol->SetLineColor(color);
  auto zMoveHringC = new TGeoTranslation(0, 0, -constants::segment::lec::length + HringLength / 2.);
  auto zMoveHringA = new TGeoTranslation(0, 0, -constants::segment::lec::length + HringLength / 2. + constants::segment::length - HringLength);

  // Longerons are made by same material
  [[maybe_unused]] auto longeronR = new TGeoTubeSeg(Form("longeronR%d", mNLayer), mRmax, mRmax + dRadius, longeronsLength / 2, phiSta, phiSta + phiLongeronsCover);
  [[maybe_unused]] auto longeronL = new TGeoTubeSeg(Form("longeronL%d", mNLayer), mRmax, mRmax + dRadius, longeronsLength / 2, phiEnd - phiLongeronsCover, phiEnd);
  TString nameLongerons = Form("longeronR%d + longeronL%d", mNLayer, mNLayer);
  auto longerons = new TGeoCompositeShape(nameLongerons);
  auto longeronsVol = new TGeoVolume(Form("longerons%d", mNLayer), longerons, mCarbon);
  longeronsVol->SetLineColor(color);
  auto zMoveLongerons = new TGeoTranslation(0, 0, -constants::segment::lec::length + constants::segment::length / 2.);

  mCarbonForm->AddNode(HringCVol, 0, zMoveHringC);
  mCarbonForm->AddNode(HringAVol, 0, zMoveHringA);
  mCarbonForm->AddNode(longeronsVol, 0, zMoveLongerons);
  mCarbonForm->AddNode(mChip, 0);
}

TGeoCompositeShape* ITS3Layer::getHringShape(TGeoTubeSeg* Hring)
{
  // Function to dig holes in H-rings
  using namespace constants::carbonfoam;
  double stepPhiHoles = (Hring->GetPhi2() - Hring->GetPhi1()) / (nHoles[mNLayer]);
  double phiHolesSta = Hring->GetPhi1() + stepPhiHoles / 2.;
  double radiusHring = 0.5 * (Hring->GetRmin() + Hring->GetRmax());
  TGeoCompositeShape* HringWithHoles = nullptr;
  TString nameAllHoles = "";
  for (int iHoles = 0; iHoles < nHoles[mNLayer]; iHoles++) {
    double phiHole = phiHolesSta + stepPhiHoles * iHoles;
    TString nameHole = Form("hole_%d_%d", iHoles, mNLayer);
    [[maybe_unused]] auto hole = new TGeoTube(nameHole, 0, radiusHoles[mNLayer], 3 * Hring->GetDz());
    // move hole to the hring radius
    auto zMoveHole = new TGeoTranslation(Form("zMoveHole_%d_%d", iHoles, mNLayer), radiusHring * cos(phiHole * o2m::Deg2Rad), radiusHring * sin(phiHole * o2m::Deg2Rad), 0);
    zMoveHole->RegisterYourself();
    nameAllHoles += Form("hole_%d_%d:zMoveHole_%d_%d + ", iHoles, mNLayer, iHoles, mNLayer);
  }
  nameAllHoles.Remove(nameAllHoles.Length() - 3, 3);
  TString nameHringWithHoles = Form("%s - (%s)", Hring->GetName(), nameAllHoles.Data());
  HringWithHoles = new TGeoCompositeShape(nameHringWithHoles);
  return HringWithHoles;
}

void ITS3Layer::createLayerImpl()
{
  if (mLayer != nullptr) {
    return;
  } else {
    createCarbonForm();
  }
  // At long last a single layer... A layer is two HalfLayers (duuhhh) but
  // we have to take care of the equatorial gap. So both half layers will be
  // offset slightly by rotating in phi the upper HalfLayer and negative phi
  // the other one.
  mLayer = new TGeoVolumeAssembly(its3TGeo::getITS3LayerPattern(mNLayer));
  mLayer->VisibleDaughters();

  // The offset is the right angle triangle of the middle radius with the
  // transverse axis.
  double phiOffset = std::asin(constants::equatorialGap / 2. / mR) * o2m::Rad2Deg;
  auto rotTop = new TGeoRotation("", 0, 0, +phiOffset);
  auto rotBot = new TGeoRotation("", 0, 0, phiOffset + 180);

  mLayer->AddNode(mCarbonForm, 0, rotTop);
  mLayer->AddNode(mCarbonForm, 1, rotBot);
}

void ITS3Layer::buildPartial(TGeoVolume* motherVolume, TGeoMatrix* mat, BuildLevel level, bool createMaterials)
{
  if (!mBuilt) {
    getMaterials(createMaterials);
  }
  switch (level) {
    case BuildLevel::kPixelArray:
      createPixelArray();
      motherVolume->AddNode(mPixelArray, 0, mat);
      break;
    case BuildLevel::kTile:
      createTile();
      motherVolume->AddNode(mTile, 0, mat);
      break;
    case BuildLevel::kRSU:
      createRSU();
      motherVolume->AddNode(mRSU, 0, mat);
      break;
    case BuildLevel::kSegment:
      createSegment();
      motherVolume->AddNode(mSegment, 0, mat);
      break;
    case BuildLevel::kChip:
      createChip();
      motherVolume->AddNode(mChip, 0, mat);
      break;
    case BuildLevel::kCarbonForm:
      createCarbonForm();
      motherVolume->AddNode(mCarbonForm, 0, mat);
      break;
    case BuildLevel::kLayer:
      [[fallthrough]];
    default:
      createLayerImpl();
      motherVolume->AddNode(mLayer, 0, mat);
  }
  LOGP(info, "Partially built ITS3-{}-{}", mNLayer, getName(level));
}

} // namespace o2::its3
