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

/// \file FT3Layer.cxx
/// \brief Implementation of the FT3Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#include "FT3Simulation/FT3Layer.h"
#include "FT3Base/GeometryTGeo.h"

#include <fairlogger/Logger.h> // for LOG

#include <TGeoManager.h>        // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>         // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoTube.h>           // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>         // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include "TMathBase.h"          // for Abs
#include <TMath.h>              // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <TGeoBBox.h>
#include <string>
#include <cstdio> // for snprintf
#include <cmath>

class TGeoMedium;

using namespace TMath;
using namespace o2::ft3;
using namespace o2::itsmft;

ClassImp(FT3Layer);

FT3Layer::~FT3Layer() = default;

TGeoMaterial* FT3Layer::carbonFiberMat = nullptr;
TGeoMedium* FT3Layer::medCarbonFiber = nullptr;

TGeoMaterial* FT3Layer::kaptonMat = nullptr;
TGeoMedium* FT3Layer::kaptonMed = nullptr;

TGeoMaterial* FT3Layer::waterMat = nullptr;
TGeoMedium* FT3Layer::waterMed = nullptr;

TGeoMaterial* FT3Layer::foamMat = nullptr;
TGeoMedium* FT3Layer::medFoam = nullptr;

FT3Layer::FT3Layer(Int_t layerDirection, Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut, Float_t Layerx2X0, bool partOfMiddleLayers)
{
  // Creates a simple parametrized EndCap layer covering the given
  // pseudorapidity range at the z layer position
  mDirection = layerDirection;
  mLayerNumber = layerNumber;
  mIsMiddleLayer = partOfMiddleLayers;
  mLayerName = layerName;
  mZ = layerDirection ? std::abs(z) : -std::abs(z);
  mx2X0 = Layerx2X0;
  mInnerRadius = rIn;
  mOuterRadius = rOut;
  const double Si_X0 = 9.5;
  mChipThickness = Layerx2X0 * Si_X0;

  // Sanity checks
  if (std::isnan(mZ)) {
    LOG(fatal) << "FT3 Layer " << mLayerNumber << " has z = NaN, which is not a valid number.";
  }
  if (mZ < 0.001 && mZ > -0.001) {
    LOG(fatal) << "FT3 Layer " << mLayerNumber << " has z = " << mZ << " cm, which is very close to 0.";
  }

  LOG(info) << "Creating FT3 Layer " << mLayerNumber << " ; direction " << mDirection;
  LOG(info) << "   Using silicon X0 = " << Si_X0 << " to emulate layer radiation length.";
  LOG(info) << "   Layer z = " << mZ << " ; R_in = " << mInnerRadius << " ; R_out = " << mOuterRadius << " ; x2X0 = " << mx2X0 << " ; ChipThickness = " << mChipThickness;
}

void FT3Layer::initialize_mat()
{

  if (carbonFiberMat) {
    return;
  }

  carbonFiberMat = new TGeoMaterial("CarbonFiber", 12.0, 6.0, 1.6);
  medCarbonFiber = new TGeoMedium("CarbonFiber", 1, carbonFiberMat);

  auto* itsC = new TGeoElement("FT3_C", "Carbon", 6, 12.0107);

  auto* itsFoam = new TGeoMixture("FT3_Foam", 1);
  itsFoam->AddElement(itsC, 1);
  itsFoam->SetDensity(0.17);

  medFoam = new TGeoMedium("FT3_Foam", 1, itsFoam);
  foamMat = medFoam->GetMaterial();

  kaptonMat = new TGeoMaterial("Kapton (cooling pipe)", 13.84, 6.88, 1.346);
  kaptonMed = new TGeoMedium("Kapton (cooling pipe)", 1, kaptonMat);

  waterMat = new TGeoMaterial("Water", 18.01528, 8.0, 1.064);
  waterMed = new TGeoMedium("Water", 2, waterMat);
}

static double y_circle(double x, double radius)
{
  return (x * x < radius * radius) ? std::sqrt(radius * radius - x * x) : 0;
}

void FT3Layer::createSeparationLayer_waterCooling(TGeoVolume* motherVolume, const std::string& separationLayerName)
{

  FT3Layer::initialize_mat();

  const double carbonFiberThickness = 0.01; // cm
  const double foamSpacingThickness = 0.5;  // cm

  TGeoTube* carbonFiberLayer = new TGeoTube(mInnerRadius, mOuterRadius, carbonFiberThickness / 2);

  // volumes
  TGeoVolume* carbonFiberLayerVol1 = new TGeoVolume((separationLayerName + "_CarbonFiber1").c_str(), carbonFiberLayer, medCarbonFiber);
  TGeoVolume* carbonFiberLayerVol2 = new TGeoVolume((separationLayerName + "_CarbonFiber2").c_str(), carbonFiberLayer, medCarbonFiber);

  carbonFiberLayerVol1->SetLineColor(kGray + 2);
  carbonFiberLayerVol2->SetLineColor(kGray + 2);

  const double zSeparation = foamSpacingThickness / 2.0 + carbonFiberThickness / 2.0;

  motherVolume->AddNode(carbonFiberLayerVol1, 1, new TGeoTranslation(0, 0, mZ - zSeparation));
  motherVolume->AddNode(carbonFiberLayerVol2, 1, new TGeoTranslation(0, 0, mZ + zSeparation));

  const double pipeOuterRadius = 0.20;
  const double kaptonThickness = 0.0025;
  const double pipeInnerRadius = pipeOuterRadius - kaptonThickness;
  const double pipeMaxLength = mOuterRadius * 2.0;

  int name_it = 0;

  // positions of the pipes depending on the overlap of the sensors inactive regions: (ALICE 3 dimensions)
  // partial:
  //  std::vector<double> X_pos = {-63.2, -58.4, -53.6, -48.8, -44.0, -39.199999999999996, -34.4, -29.599999999999994, -24.799999999999997, -19.999999999999993, -15.199999999999998, -10.399999999999993, -5.599999999999998, -0.7999999999999936, 4.000000000000002, 8.800000000000006, 13.600000000000001, 18.400000000000006, 23.200000000000003, 28.000000000000007, 32.800000000000004, 37.60000000000001, 42.400000000000006, 47.20000000000001, 52.00000000000001, 56.80000000000001, 61.60000000000001, 66.4};
  // complete:
  // std::vector<double> X_pos = {-63.4, -58.8, -54.199999999999996, -49.599999999999994, -44.99999999999999, -40.39999999999999, -35.79999999999999, -31.199999999999992, -26.59999999999999, -21.999999999999993, -17.39999999999999, -12.799999999999994, -8.199999999999992, -3.5999999999999934, 1.000000000000008, 5.600000000000007, 10.200000000000008, 14.800000000000008, 19.40000000000001, 24.000000000000007, 28.60000000000001, 33.20000000000001, 37.80000000000001, 42.40000000000001, 47.000000000000014, 51.600000000000016, 56.20000000000002, 60.80000000000002, 65.40000000000002};
  std::vector<double> X_pos = {-62.3168, -57.9836, -53.650400000000005, -49.317200000000014, -44.984000000000016, -40.65080000000002, -36.31760000000002, -31.984400000000026, -27.65120000000003, -23.318000000000037, -18.98480000000004, -14.651600000000043, -10.318400000000047, -5.98520000000005, -1.6520000000000519, 2.6811999999999445, 7.014399999999941, 11.347599999999936, 15.680799999999934, 20.01399999999993, 24.347199999999926, 28.68039999999992, 33.013599999999926, 37.34679999999992, 41.980000000000004, 46.613200000000006, 51.246399999999994, 55.87960000000001, 60.5128};

  for (double xPos : X_pos) {

    double pipeLength = pipeMaxLength;
    double yMax = 0.0;

    TGeoRotation* rotation = new TGeoRotation();
    rotation->RotateX(90);

    if (std::abs(xPos) < mInnerRadius) {
      double yInner = std::abs(y_circle(xPos, mInnerRadius));
      double yOuter = std::abs(y_circle(xPos, mOuterRadius));

      yMax = 2 * yOuter;
      pipeLength = yMax;

      double positiveYLength = yOuter - yInner;

      TGeoVolume* kaptonPipePos = new TGeoVolume((separationLayerName + "_KaptonPipePos_" + std::to_string(name_it)).c_str(), new TGeoTube(pipeInnerRadius, pipeOuterRadius, positiveYLength / 2), kaptonMed);
      kaptonPipePos->SetLineColor(kGray);
      TGeoVolume* waterVolumePos = new TGeoVolume((separationLayerName + "_WaterVolumePos_" + std::to_string(name_it)).c_str(), new TGeoTube(0.0, pipeInnerRadius, positiveYLength / 2), waterMed);
      waterVolumePos->SetLineColor(kBlue);

      motherVolume->AddNode(waterVolumePos, 1, new TGeoCombiTrans(xPos, (yInner + yOuter) / 2.0, mZ, rotation));

      TGeoVolume* kaptonPipeNeg = new TGeoVolume((separationLayerName + "_KaptonPipeNeg_" + std::to_string(name_it)).c_str(), new TGeoTube(pipeInnerRadius, pipeOuterRadius, positiveYLength / 2), kaptonMed);
      kaptonPipeNeg->SetLineColor(kGray);
      TGeoVolume* waterVolumeNeg = new TGeoVolume((separationLayerName + "_WaterVolumeNeg_" + std::to_string(name_it)).c_str(), new TGeoTube(0.0, pipeInnerRadius, positiveYLength / 2), waterMed);
      waterVolumeNeg->SetLineColor(kBlue);

      motherVolume->AddNode(waterVolumeNeg, 1, new TGeoCombiTrans(xPos, -(yInner + yOuter) / 2.0, mZ, rotation));

      motherVolume->AddNode(kaptonPipePos, 1, new TGeoCombiTrans(xPos, (yInner + yOuter) / 2.0, mZ, rotation));
      motherVolume->AddNode(kaptonPipeNeg, 1, new TGeoCombiTrans(xPos, -(yInner + yOuter) / 2.0, mZ, rotation));

    } else {

      double yOuter = std::abs(y_circle(xPos, mOuterRadius));
      yMax = 2 * yOuter;
      pipeLength = yMax;

      TGeoVolume* kaptonPipe = new TGeoVolume((separationLayerName + "_KaptonPipe_" + std::to_string(name_it)).c_str(), new TGeoTube(pipeInnerRadius, pipeOuterRadius, pipeLength / 2), kaptonMed);
      kaptonPipe->SetLineColor(kGray);
      TGeoVolume* waterVolume = new TGeoVolume((separationLayerName + "_WaterVolume_" + std::to_string(name_it)).c_str(), new TGeoTube(0.0, pipeInnerRadius, pipeLength / 2), waterMed);
      waterVolume->SetLineColor(kBlue);

      motherVolume->AddNode(waterVolume, 1, new TGeoCombiTrans(xPos, 0, mZ, rotation));
      motherVolume->AddNode(kaptonPipe, 1, new TGeoCombiTrans(xPos, 0, mZ, rotation));
    }

    name_it++;
  }
}

void FT3Layer::createSeparationLayer(TGeoVolume* motherVolume, const std::string& separationLayerName)
{

  FT3Layer::initialize_mat();

  constexpr double carbonFiberThickness = 0.01; // cm
  constexpr double foamSpacingThickness = 1.0;  // cm

  TGeoTube* carbonFiberLayer = new TGeoTube(mInnerRadius, mOuterRadius, carbonFiberThickness / 2);
  TGeoTube* foamLayer = new TGeoTube(mInnerRadius, mOuterRadius, foamSpacingThickness / 2);

  // volumes
  TGeoVolume* carbonFiberLayerVol1 = new TGeoVolume((separationLayerName + "_CarbonFiber1").c_str(), carbonFiberLayer, medCarbonFiber);
  TGeoVolume* foamLayerVol = new TGeoVolume((separationLayerName + "_Foam").c_str(), foamLayer, medFoam);
  TGeoVolume* carbonFiberLayerVol2 = new TGeoVolume((separationLayerName + "_CarbonFiber2").c_str(), carbonFiberLayer, medCarbonFiber);

  carbonFiberLayerVol1->SetLineColor(kGray + 2);
  foamLayerVol->SetLineColor(kBlack);
  foamLayerVol->SetFillColorAlpha(kBlack, 1.0);
  carbonFiberLayerVol2->SetLineColor(kGray + 2);

  const double zSeparation = foamSpacingThickness / 2.0 + carbonFiberThickness / 2.0;

  motherVolume->AddNode(carbonFiberLayerVol1, 1, new TGeoTranslation(0, 0, 0 - zSeparation));
  motherVolume->AddNode(foamLayerVol, 1, new TGeoTranslation(0, 0, 0));
  motherVolume->AddNode(carbonFiberLayerVol2, 1, new TGeoTranslation(0, 0, 0 + zSeparation));
}

void FT3Layer::createLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber < 0) {
    LOG(fatal) << "Invalid layer number " << mLayerNumber << " for FT3 layer.";
  }
  if (mIsMiddleLayer) { // ML disks

    std::string chipName = o2::ft3::GeometryTGeo::getFT3ChipPattern() + std::to_string(mLayerNumber),
                sensName = Form("%s_%d_%d", GeometryTGeo::getFT3SensorPattern(), mDirection, mLayerNumber);
    TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
    TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
    TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);

    TGeoMedium* medSi = gGeoManager->GetMedium("FT3_SILICON$");
    TGeoMedium* medAir = gGeoManager->GetMedium("FT3_AIR$");

    TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
    sensVol->SetLineColor(kYellow);
    TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
    chipVol->SetLineColor(kYellow);
    TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
    layerVol->SetLineColor(kYellow);

    LOG(info) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
    chipVol->AddNode(sensVol, 1, nullptr);

    LOG(info) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
    layerVol->AddNode(chipVol, 1, nullptr);

    // Finally put everything in the mother volume
    auto* FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
    auto* FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

    LOG(info) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
    motherVolume->AddNode(layerVol, 1, FwdDiskCombiTrans);

  } else { // OT disks

    FT3Module module;

    // layer structure
    std::string frontLayerName = o2::ft3::GeometryTGeo::getFT3LayerPattern() + std::to_string(mDirection) + std::to_string(mLayerNumber) + "_Front";
    std::string backLayerName = o2::ft3::GeometryTGeo::getFT3LayerPattern() + std::to_string(mDirection) + std::to_string(mLayerNumber) + "_Back";
    std::string separationLayerName = "FT3SeparationLayer" + std::to_string(mDirection) + std::to_string(mLayerNumber);

    TGeoMedium* medAir = gGeoManager->GetMedium("FT3_AIR$");
    TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, 10 * mChipThickness / 2);
    TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
    layerVol->SetLineColor(kYellow + 2);

    // createSeparationLayer_waterCooling(motherVolume, separationLayerName);
    createSeparationLayer(layerVol, separationLayerName);

    // create disk faces
    module.createModule(0, mLayerNumber, mDirection, mInnerRadius, mOuterRadius, 0., "front", "rectangular", layerVol);
    module.createModule(0, mLayerNumber, mDirection, mInnerRadius, mOuterRadius, 0., "back", "rectangular", layerVol);

    // Finally put everything in the mother volume
    auto* FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
    auto* FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

    LOG(info) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
    motherVolume->AddNode(layerVol, 1, FwdDiskCombiTrans);
  }
}
