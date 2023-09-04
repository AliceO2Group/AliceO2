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

#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Simulation/ITS3Layer.h"

#include <fairlogger/Logger.h> // for LOG

#include <TGeoTube.h> // for TGeoTube, TGeoTubeSeg
#include <TGeoCompositeShape.h>
#include <TGeoVolume.h> // for TGeoVolume, TGeoVolumeAssembly

using namespace o2::its3;

/// \cond CLASSIMP
ClassImp(ITS3Layer);
/// \endcond

ITS3Layer::ITS3Layer(int lay)
  : TObject(),
    mLayerNumber(lay)
{
  SegmentationSuperAlpide seg(lay);
  mChipThickness = seg.mDetectorLayerThickness;
  mSensorThickness = seg.mSensorLayerThickness;
}

ITS3Layer::~ITS3Layer() = default;

void ITS3Layer::createLayer(TGeoVolume* motherVolume, double radiusBetweenLayer)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("IT3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IT3_AIR$");

  double rmin = mRadius;
  double rmax = rmin + mChipThickness;
  double rminSensor = rmax - mSensorThickness;
  radiusBetweenLayer = radiusBetweenLayer - mChipThickness;
  double phiGap = TMath::ASin(mGapPhi / 2.f / mRadius) * TMath::RadToDeg(); // degrees
  double piDeg = TMath::Pi() * TMath::RadToDeg();                           // degrees

  const int nElements = 7;
  std::string names[nElements];

  // do we need to keep the hierarchy?
  names[0] = Form("%s%d", o2::its::GeometryTGeo::getITS3SensorPattern(), mLayerNumber);
  names[1] = Form("%s%d", o2::its::GeometryTGeo::getITS3ChipPattern(), mLayerNumber);
  names[2] = Form("%s%d", o2::its::GeometryTGeo::getITS3ModulePattern(), mLayerNumber);
  names[3] = Form("%s%d", o2::its::GeometryTGeo::getITS3HalfStavePattern(), mLayerNumber);
  names[4] = Form("%s%d", o2::its::GeometryTGeo::getITS3StavePattern(), mLayerNumber);
  names[5] = Form("%s%d", o2::its::GeometryTGeo::getITS3HalfBarrelPattern(), mLayerNumber);
  names[6] = Form("%s%d", o2::its::GeometryTGeo::getITS3LayerPattern(), mLayerNumber);

  TGeoTubeSeg* halfLayer[nElements - 1];
  TGeoVolume* volHalfLayer[nElements - 1];
  for (int iEl{0}; iEl < nElements - 1; ++iEl) {
    TGeoMedium* med = (iEl <= 2) ? medSi : medAir;
    if (iEl == 4) {
      halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax + radiusBetweenLayer, mZLen / 2, phiGap, piDeg - phiGap);
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
      createCarbonFoamStructure(volHalfLayer[iEl], radiusBetweenLayer);
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kGray + 2);
    } else if (iEl < 2) {
      halfLayer[iEl] = new TGeoTubeSeg(rminSensor, rmax, mZLen / 2, phiGap, piDeg - phiGap);
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kRed + 1);
    } else if (iEl == 2) {
      halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax, mZLen / 2, phiGap, piDeg - phiGap);
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kRed + 1);
    } else { // all the others are simply half cylinders filling all the space
      halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax + radiusBetweenLayer, mZLen / 2, phiGap, piDeg - phiGap);
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
    }
    volHalfLayer[iEl]->SetUniqueID(mChipTypeID);

    if (iEl > 0) {
      LOGP(debug, "Inserting {} inside {}", volHalfLayer[iEl - 1]->GetName(), volHalfLayer[iEl]->GetName());
      int id = 0;
      if (iEl == 1) {
        id = 1;
      }
      volHalfLayer[iEl]->AddNode(volHalfLayer[iEl - 1], id, nullptr);
    }
  }

  TGeoTranslation* translationTop = new TGeoTranslation(0., mGapY / 2, 0.);
  TGeoTranslation* translationBottom = new TGeoTranslation(0., -mGapY / 2, 0.);
  TGeoRotation* rotationBottom = new TGeoRotation("", piDeg, 0., 0.);
  TGeoCombiTrans* rotoTranslationBottom =
    new TGeoCombiTrans(*translationBottom, *rotationBottom);

  TGeoVolumeAssembly* volLayer = new TGeoVolumeAssembly(names[nElements - 1].data());
  volLayer->AddNode(volHalfLayer[nElements - 2], 0, translationTop);
  volLayer->AddNode(volHalfLayer[nElements - 2], 1, rotoTranslationBottom);

  // Finally put everything in the mother volume
  LOGP(debug, "Inserting {} inside {}", volLayer->GetName(), motherVolume->GetName());
  motherVolume->AddNode(volLayer, 1, nullptr);
}

void ITS3Layer::createLayerWithDeadZones(TGeoVolume* motherVolume, double radiusBetweenLayer)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("IT3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IT3_AIR$");

  double rmin = mRadius;
  double rmax = rmin + mChipThickness;
  double rminSensor = rmax - mSensorThickness;
  double rmed = (rmax + rmin) / 2;
  // width of sensors of layers is calculated from r and chips' widths
  double widthSensor = (TMath::Pi() * rmed - (mNumSubSensorsHalfLayer - 1) * mMiddleChipWidth - 2 * mFringeChipWidth) / mNumSubSensorsHalfLayer;
  radiusBetweenLayer = radiusBetweenLayer - mChipThickness;
  double phiGap = TMath::ASin(mGapPhi / 2.f / mRadius) * TMath::RadToDeg(); // degrees
  double piDeg = TMath::Pi() * TMath::RadToDeg();                           // degrees

  const int nElements = 7;
  std::string names[nElements];
  int nObjPerElement[nElements] = {mNumSubSensorsHalfLayer, 1, 1, 1, 1, 1, 1}; // mNumSubSensorsHalfLayer chips and sensors per half layer

  names[0] = Form("%s%d", o2::its::GeometryTGeo::getITS3SensorPattern(), mLayerNumber);
  names[1] = Form("%s%d", o2::its::GeometryTGeo::getITS3ChipPattern(), mLayerNumber);
  names[2] = Form("%s%d", o2::its::GeometryTGeo::getITS3ModulePattern(), mLayerNumber);
  names[3] = Form("%s%d", o2::its::GeometryTGeo::getITS3HalfStavePattern(), mLayerNumber);
  names[4] = Form("%s%d", o2::its::GeometryTGeo::getITS3StavePattern(), mLayerNumber);
  names[5] = Form("%s%d", o2::its::GeometryTGeo::getITS3HalfBarrelPattern(), mLayerNumber);
  names[6] = Form("%s%d", o2::its::GeometryTGeo::getITS3LayerPattern(), mLayerNumber);

  std::array<std::vector<TGeoTubeSeg*>, nElements - 1> halfLayer{};
  TGeoVolume* volHalfLayer[nElements - 1];

  for (int iEl{0}; iEl < nElements - 1; ++iEl) {
    TGeoMedium* med = (iEl <= 2) ? medSi : medAir;

    for (int iObj{0}; iObj < nObjPerElement[iEl]; ++iObj) {
      if (iEl == 0) { // subsensors (mNumSubSensorsHalfLayer sectors with dead zones)
        if (iObj == 0) {
          halfLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rminSensor, rmax, mZLen / 2, TMath::RadToDeg() * mFringeChipWidth / rmed + phiGap, TMath::RadToDeg() * (mFringeChipWidth + widthSensor) / rmed));
        } else if (iObj == mNumSubSensorsHalfLayer - 1) {
          halfLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rminSensor, rmax, mZLen / 2, TMath::RadToDeg() * (mFringeChipWidth + iObj * widthSensor + iObj * mMiddleChipWidth) / rmed, piDeg - TMath::RadToDeg() * mFringeChipWidth / rmed - phiGap));
        } else {
          halfLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rminSensor, rmax, mZLen / 2, TMath::RadToDeg() * (mFringeChipWidth + iObj * widthSensor + iObj * mMiddleChipWidth) / rmed, TMath::RadToDeg() * (mFringeChipWidth + (iObj + 1) * widthSensor + iObj * mMiddleChipWidth) / rmed));
        }
      } else if (iEl == 1) {
        halfLayer[iEl].push_back(new TGeoTubeSeg(rminSensor, rmax, mZLen / 2, phiGap, piDeg - phiGap));
      } else if (iEl == 2) {
        halfLayer[iEl].push_back(new TGeoTubeSeg(rmin, rmax, mZLen / 2, phiGap, piDeg - phiGap));
      } else { // all the others are simply half cylinders filling all the space
        halfLayer[iEl].push_back(new TGeoTubeSeg(rmin, rmax + radiusBetweenLayer, mZLen / 2, phiGap, piDeg - phiGap));
      }
    }

    if (iEl == 0) {
      std::string subSensNames = "";
      for (int iObj{0}; iObj < nObjPerElement[iEl] - 1; ++iObj) {
        subSensNames += Form("subsens%dlayer%d+", iObj, mLayerNumber);
      }
      subSensNames += Form("subsens%dlayer%d", nObjPerElement[iEl] - 1, mLayerNumber);
      TGeoCompositeShape* sensor = new TGeoCompositeShape(subSensNames.data());
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), sensor, med);
      volHalfLayer[iEl]->SetUniqueID(mChipTypeID);
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kRed + 1);
    } else {
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl][0], med);
      volHalfLayer[iEl]->SetUniqueID(mChipTypeID);
      if (iEl == 4) {
        createCarbonFoamStructure(volHalfLayer[iEl], radiusBetweenLayer);
        volHalfLayer[iEl]->SetVisibility(true);
        volHalfLayer[iEl]->SetLineColor(kGray + 2);
      } else if (iEl == 1) {
        volHalfLayer[iEl]->SetVisibility(true);
        volHalfLayer[iEl]->SetLineColor(kBlue + 2);
      }

      int id = (iEl == 1) ? 1 : 0;
      LOGP(debug, "Inserting {} id {} inside {}", volHalfLayer[iEl - 1]->GetName(), id, volHalfLayer[iEl]->GetName());
      volHalfLayer[iEl]->AddNode(volHalfLayer[iEl - 1], id, nullptr);
    }
  }

  TGeoTranslation* translationTop = new TGeoTranslation(0., mGapY / 2, 0.);
  TGeoTranslation* translationBottom = new TGeoTranslation(0., -mGapY / 2, 0.);
  TGeoRotation* rotationBottom = new TGeoRotation("", piDeg, 0., 0.);
  TGeoCombiTrans* rotoTranslationBottom =
    new TGeoCombiTrans(*translationBottom, *rotationBottom);

  TGeoVolumeAssembly* volLayer = new TGeoVolumeAssembly(names[nElements - 1].data());
  volLayer->AddNode(volHalfLayer[nElements - 2], 0, translationTop);
  volLayer->AddNode(volHalfLayer[nElements - 2], 1, rotoTranslationBottom);

  // Finally put everything in the mother volume
  LOGP(debug, "Inserting {} inside {}", volLayer->GetName(), motherVolume->GetName());
  motherVolume->AddNode(volLayer, 1, nullptr);
}

void ITS3Layer::create4thLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("IT3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IT3_AIR$");

  double rmin = mRadius;
  double rmax = rmin + mChipThickness;
  double rminSensor = rmax - mSensorThickness;
  double rmed = (rmax + rmin) / 2;
  // width of sensors of layers is calculated from r and chips' widths
  double widthSensor = (0.5 * TMath::Pi() * rmed - (mNumSubSensorsHalfLayer - 1) * mMiddleChipWidth - 2 * mFringeChipWidth) / mNumSubSensorsHalfLayer;
  double phiGap = TMath::ASin(mGapPhi / 2.f / mRadius) * TMath::RadToDeg(); // degrees
  double piDeg = TMath::Pi() * TMath::RadToDeg();                           // degrees

  const int nElements = 7;
  std::string names[nElements];
  int nObjPerElement[nElements] = {mNumSubSensorsHalfLayer, 1, 1, 1, 1, 1, 1}; // mNumSubSensorsHalfLayer chips and sensors per half layer

  names[0] = Form("%s%d", o2::its::GeometryTGeo::getITS3SensorPattern(), mLayerNumber);
  names[1] = Form("%s%d", o2::its::GeometryTGeo::getITS3ChipPattern(), mLayerNumber);
  names[2] = Form("%s%d", o2::its::GeometryTGeo::getITS3ModulePattern(), mLayerNumber);
  names[3] = Form("%s%d", o2::its::GeometryTGeo::getITS3HalfStavePattern(), mLayerNumber);
  names[4] = Form("%s%d", o2::its::GeometryTGeo::getITS3StavePattern(), mLayerNumber);
  names[5] = Form("%s%d", o2::its::GeometryTGeo::getITS3HalfBarrelPattern(), mLayerNumber);
  names[6] = Form("%s%d", o2::its::GeometryTGeo::getITS3LayerPattern(), mLayerNumber);

  std::array<std::vector<TGeoTubeSeg*>, nElements - 1> quarterLayer{};
  TGeoVolume* volQuarterLayer[nElements - 1];

  for (int iEl{0}; iEl < nElements - 2; ++iEl) { // we need only 2 half barrels as usual, so we stop at the stave
    TGeoMedium* med = (iEl <= 2) ? medSi : medAir;

    for (int iObj{0}; iObj < nObjPerElement[iEl]; ++iObj) {
      if (iEl == 0) { // subsensors (mNumSubSensorsHalfLayer sectors with dead zones)
        if (iObj == 0) {
          quarterLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rminSensor, rmax, mZLen / 2, piDeg / 4. + TMath::RadToDeg() * mFringeChipWidth / rmed + phiGap, piDeg / 4. + TMath::RadToDeg() * (mFringeChipWidth + widthSensor) / rmed));
        } else if (iObj == mNumSubSensorsHalfLayer - 1) {
          quarterLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rminSensor, rmax, mZLen / 2, piDeg / 4. + TMath::RadToDeg() * (mFringeChipWidth + iObj * widthSensor + iObj * mMiddleChipWidth) / rmed, 3. / 4. * piDeg - TMath::RadToDeg() * mFringeChipWidth / rmed - phiGap));
        } else {
          quarterLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rminSensor, rmax, mZLen / 2, piDeg / 4. + TMath::RadToDeg() * (mFringeChipWidth + iObj * widthSensor + iObj * mMiddleChipWidth) / rmed, piDeg / 4. + TMath::RadToDeg() * (mFringeChipWidth + (iObj + 1) * widthSensor + iObj * mMiddleChipWidth) / rmed));
        }
      } else if (iEl == 1) {
        quarterLayer[iEl].push_back(new TGeoTubeSeg(rminSensor, rmax, mZLen / 2, phiGap, piDeg / 4. + piDeg * 3 / 4. - phiGap));
      } else if (iEl == 2) {
        quarterLayer[iEl].push_back(new TGeoTubeSeg(rmin, rmax, mZLen / 2, piDeg / 4. + phiGap, piDeg * 3 / 4. - phiGap));
      } else { // all the others are simply quarter cylinders filling all the space
        quarterLayer[iEl].push_back(new TGeoTubeSeg(rmin, rmax, mZLen / 2, piDeg / 4. + phiGap, piDeg * 3 / 4. - phiGap));
      }
    }

    if (iEl == 0) {
      std::string subSensNames = "";
      for (int iObj{0}; iObj < nObjPerElement[iEl] - 1; ++iObj) {
        subSensNames += Form("subsens%dlayer%d+", iObj, mLayerNumber);
      }
      subSensNames += Form("subsens%dlayer%d", nObjPerElement[iEl] - 1, mLayerNumber);
      TGeoCompositeShape* sensor = new TGeoCompositeShape(subSensNames.data());
      volQuarterLayer[iEl] = new TGeoVolume(names[iEl].data(), sensor, med);
      volQuarterLayer[iEl]->SetUniqueID(mChipTypeID);
      volQuarterLayer[iEl]->SetVisibility(true);
      volQuarterLayer[iEl]->SetLineColor(kRed + 1);
    } else {
      volQuarterLayer[iEl] = new TGeoVolume(names[iEl].data(), quarterLayer[iEl][0], med);
      volQuarterLayer[iEl]->SetUniqueID(mChipTypeID);
      if (iEl == 4) {
        // createCarbonFoamStructure(volQuarterLayer[iEl], 0., true);
        volQuarterLayer[iEl]->SetVisibility(true);
        volQuarterLayer[iEl]->SetLineColor(kGray + 2);
      } else if (iEl == 1) {
        volQuarterLayer[iEl]->SetVisibility(true);
        volQuarterLayer[iEl]->SetLineColor(kBlue + 2);
      }

      int id = (iEl == 1) ? 1 : 0;
      LOGP(debug, "Inserting {} id {} inside {}", volQuarterLayer[iEl - 1]->GetName(), id, volQuarterLayer[iEl]->GetName());
      volQuarterLayer[iEl]->AddNode(volQuarterLayer[iEl - 1], id, nullptr);
    }
  }

  TGeoTranslation* translationTopRight = new TGeoTranslation(mGapXDirection / 2, 0., 0.);
  TGeoRotation* rotationTopRight = new TGeoRotation("", -piDeg / 4, 0., 0.);
  TGeoCombiTrans* rotoTranslationTopRight = new TGeoCombiTrans(*translationTopRight, *rotationTopRight);

  TGeoTranslation* translationTopLeft = new TGeoTranslation(-mGapXDirection / 2, 0., 0.);
  TGeoRotation* rotationTopLeft = new TGeoRotation("", piDeg / 4, 0., 0.);
  TGeoCombiTrans* rotoTranslationTopLeft = new TGeoCombiTrans(*translationTopLeft, *rotationTopLeft);

  TGeoVolumeAssembly* halfLayer = new TGeoVolumeAssembly(names[nElements - 2].data());
  halfLayer->AddNode(volQuarterLayer[nElements - 3], 0, rotoTranslationTopRight);
  halfLayer->AddNode(volQuarterLayer[nElements - 3], 1, rotoTranslationTopLeft);

  TGeoTranslation* translationTop = new TGeoTranslation(0., mGapY / 2, 0.);
  TGeoTranslation* translationBottom = new TGeoTranslation(0., -mGapY / 2, 0.);
  TGeoRotation* rotationBottom = new TGeoRotation("", piDeg, 0., 0.);
  TGeoCombiTrans* rotoTranslationBottom = new TGeoCombiTrans(*translationBottom, *rotationBottom);

  TGeoVolumeAssembly* volLayer = new TGeoVolumeAssembly(names[nElements - 1].data());
  volLayer->AddNode(halfLayer, 0, translationTop);
  volLayer->AddNode(halfLayer, 1, rotoTranslationBottom);

  // Finally put everything in the mother volume
  LOGP(debug, "Inserting {} inside {}", volLayer->GetName(), motherVolume->GetName());
  motherVolume->AddNode(volLayer, 1, nullptr);
}

void ITS3Layer::createCarbonFoamStructure(TGeoVolume* motherVolume, double deltaR, bool fourthLayer)
{
  TGeoMedium* medCarbonFoam = (mBuildLevel < 1) ? gGeoManager->GetMedium("IT3_ERGDUOCEL$") : gGeoManager->GetMedium("IT3_AIR$"); // if build level >= 1 we do not put carbon foam but air
  TGeoMedium* medGlue = (mBuildLevel < 2) ? gGeoManager->GetMedium("IT3_IMPREG_FLEECE$") : gGeoManager->GetMedium("IT3_AIR$");   // if build level >= 2 we do not put glue but air
  TGeoMedium* medSi = (mBuildLevel <= 2) ? gGeoManager->GetMedium("IT3_SI$") : gGeoManager->GetMedium("IT3_AIR$");               // if build level > 2 we do not put silicon but air

  double rmaxWoAddMat = mRadius + mChipThickness;
  double rmax = rmaxWoAddMat + mAddMaterial;
  double radiusBetweenLayer = deltaR - mChipThickness;
  double rmedFoam = rmax + radiusBetweenLayer / 2;
  double phiGap = TMath::ASin(mGapPhi / 2.f / mRadius) * TMath::RadToDeg(); // degrees
  double piDeg = TMath::Pi() * TMath::RadToDeg();                           // degrees
  double phiMin = (fourthLayer) ? piDeg / 4 : 0.;                           // degrees
  double phiMax = (fourthLayer) ? piDeg * 3 / 4 : piDeg;                    // degrees

  TGeoTranslation* transSemicircle[2];
  transSemicircle[0] = new TGeoTranslation("transSemicircleFoam0", 0, 0, (mZLen - mLengthSemiCircleFoam) / 2);
  transSemicircle[0]->RegisterYourself();
  transSemicircle[1] = new TGeoTranslation("transSemicircleFoam1", 0, 0, -(mZLen - mLengthSemiCircleFoam) / 2);
  transSemicircle[1]->RegisterYourself();

  TGeoTubeSeg* subGluedFoamBottom[4];
  subGluedFoamBottom[0] = new TGeoTubeSeg(Form("subgluedfoambottom0layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
  subGluedFoamBottom[1] = new TGeoTubeSeg(Form("subgluedfoambottom1layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
  subGluedFoamBottom[2] = new TGeoTubeSeg(Form("subgluedfoambottom2layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, (mZLen - mLengthSemiCircleFoam) / 2, phiMin + phiGap, phiMin + TMath::RadToDeg() * mHeightStripFoam / rmedFoam + phiGap);
  subGluedFoamBottom[3] = new TGeoTubeSeg(Form("subgluedfoambottom3layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, (mZLen - mLengthSemiCircleFoam) / 2, phiMax - TMath::RadToDeg() * (mHeightStripFoam / rmedFoam) - phiGap, phiMax - phiGap);
  TGeoTubeSeg* subGluedFoamTop[4];
  subGluedFoamTop[0] = new TGeoTubeSeg(Form("subgluedfoamtop0layer%d", mLayerNumber), rmax + radiusBetweenLayer - mThickGluedFoam, rmax + radiusBetweenLayer, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
  subGluedFoamTop[1] = new TGeoTubeSeg(Form("subgluedfoamtop1layer%d", mLayerNumber), rmax + radiusBetweenLayer - mThickGluedFoam, rmax + radiusBetweenLayer, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
  subGluedFoamTop[2] = new TGeoTubeSeg(Form("subgluedfoamtop2layer%d", mLayerNumber), rmax + radiusBetweenLayer - mThickGluedFoam, rmax + radiusBetweenLayer, (mZLen - mLengthSemiCircleFoam) / 2, phiMin + phiGap, phiMin + TMath::RadToDeg() * mHeightStripFoam / rmedFoam + phiGap);
  subGluedFoamTop[3] = new TGeoTubeSeg(Form("subgluedfoamtop3layer%d", mLayerNumber), rmax + radiusBetweenLayer - mThickGluedFoam, rmax + radiusBetweenLayer, (mZLen - mLengthSemiCircleFoam) / 2, phiMax - TMath::RadToDeg() * (mHeightStripFoam / rmedFoam) - phiGap, phiMax - phiGap);

  std::string subGluedFoamsNames = "";
  for (int iObj{0}; iObj < 2; ++iObj) {
    subGluedFoamsNames += Form("(subgluedfoambottom%dlayer%d:transSemicircleFoam%d)+", iObj, mLayerNumber, iObj);
  }
  subGluedFoamsNames += Form("subgluedfoambottom2layer%d+", mLayerNumber);
  subGluedFoamsNames += Form("subgluedfoambottom3layer%d+", mLayerNumber);

  for (int iObj{0}; iObj < 2; ++iObj) {
    subGluedFoamsNames += Form("(subgluedfoamtop%dlayer%d:transSemicircleFoam%d)+", iObj, mLayerNumber, iObj);
  }
  subGluedFoamsNames += Form("subgluedfoamtop2layer%d+", mLayerNumber);
  subGluedFoamsNames += Form("subgluedfoamtop3layer%d", mLayerNumber);

  TGeoCompositeShape* gluedfoam = new TGeoCompositeShape(subGluedFoamsNames.data());
  TGeoVolume* volGlue = new TGeoVolume(Form("Glue%d", mLayerNumber), gluedfoam, medGlue);
  motherVolume->AddNode(volGlue, 1, nullptr);

  TGeoTubeSeg* subFoam[4];
  subFoam[0] = new TGeoTubeSeg(Form("subfoam0layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer - mThickGluedFoam, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
  subFoam[1] = new TGeoTubeSeg(Form("subfoam1layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer - mThickGluedFoam, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
  subFoam[2] = new TGeoTubeSeg(Form("subfoam2layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer - mThickGluedFoam, (mZLen - mLengthSemiCircleFoam) / 2, phiMin + phiGap, phiMin + TMath::RadToDeg() * mHeightStripFoam / rmedFoam + phiGap);
  subFoam[3] = new TGeoTubeSeg(Form("subfoam3layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer - mThickGluedFoam, (mZLen - mLengthSemiCircleFoam) / 2, phiMax - TMath::RadToDeg() * (mHeightStripFoam / rmedFoam) - phiGap, phiMax - phiGap);

  std::string subFoamNames = "";
  for (int iObj{0}; iObj < 2; ++iObj) {
    subFoamNames += Form("(subfoam%dlayer%d:transSemicircleFoam%d)+", iObj, mLayerNumber, iObj);
  }
  subFoamNames += Form("subfoam2layer%d+", mLayerNumber);
  subFoamNames += Form("subfoam3layer%d", mLayerNumber);

  TGeoCompositeShape* foam = new TGeoCompositeShape(subFoamNames.data());
  TGeoVolume* volFoam = new TGeoVolume(Form("CarbonFoam%d", mLayerNumber), foam, medCarbonFoam);
  motherVolume->AddNode(volFoam, 1, nullptr);

  if (mAddMaterial > 0.) {
    TGeoTubeSeg* addMat = new TGeoTubeSeg(Form("additionalMaterialLayer%d", mLayerNumber), rmaxWoAddMat, rmax, mLengthSemiCircleFoam / 2, phiMin + phiGap, phiMax - phiGap);
    TGeoVolume* volAddMat = new TGeoVolume(Form("AdditionalMaterial%d", mLayerNumber), addMat, medSi);
    motherVolume->AddNode(volAddMat, 1, nullptr);
  }
}
