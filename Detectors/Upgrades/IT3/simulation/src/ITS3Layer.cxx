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
}

ITS3Layer::~ITS3Layer() = default;

void ITS3Layer::createLayer(TGeoVolume* motherVolume)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("IT3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IT3_AIR$");

  double rmin = mRadius;
  double rmax = rmin + mSensorThickness;
  double radiusBetweenLayer = 0.6 - mSensorThickness; // FIXME: hard coded distance between layers

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
    TGeoMedium* med = (iEl == 0) ? medSi : medAir;
    if (iEl == 4) {
      halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax + radiusBetweenLayer, mZLen / 2, 0., TMath::RadToDeg() * TMath::Pi());
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
      createCarbonFoamStructure(volHalfLayer[iEl]);
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kGray + 2);
    } else if (iEl < 2) {
      halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax, mZLen / 2, 0., TMath::RadToDeg() * TMath::Pi());
      volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kRed + 1);
    } else { // all the others are simply half cylinders filling all the space
      halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax + radiusBetweenLayer, mZLen / 2, 0., TMath::RadToDeg() * TMath::Pi());
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

  TGeoTranslation* translationTop = new TGeoTranslation(0., mGap / 2, 0.);
  TGeoTranslation* translationBottom = new TGeoTranslation(0., -mGap / 2, 0.);
  TGeoRotation* rotationBottom = new TGeoRotation("", 180., 0., 0.);
  TGeoCombiTrans* rotoTranslationBottom =
    new TGeoCombiTrans(*translationBottom, *rotationBottom);

  TGeoVolumeAssembly* volLayer = new TGeoVolumeAssembly(names[nElements - 1].data());
  volLayer->AddNode(volHalfLayer[nElements - 2], 0, translationTop);
  volLayer->AddNode(volHalfLayer[nElements - 2], 1, rotoTranslationBottom);

  // Finally put everything in the mother volume
  LOGP(debug, "Inserting {} inside {}", volLayer->GetName(), motherVolume->GetName());
  motherVolume->AddNode(volLayer, 1, nullptr);
}

void ITS3Layer::createLayerWithDeadZones(TGeoVolume* motherVolume)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("IT3_SI$");
  TGeoMedium* medAir = gGeoManager->GetMedium("IT3_AIR$");

  double rmin = mRadius;
  double rmax = rmin + mSensorThickness;
  double rmed = (rmax + rmin) / 2;
  // width of sensors of layers is calculated from r and chips' widths
  double widthSensor = (TMath::Pi() * rmed - (mNumSubSensorsHalfLayer - 2) * mMiddleChipWidth - 2 * mFringeChipWidth) / mNumSubSensorsHalfLayer;
  double radiusBetweenLayer = 0.6 - mSensorThickness; // FIXME: hard coded distance between layers

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
    TGeoMedium* med = (iEl == 0) ? medSi : medAir;

    for (int iObj{0}; iObj < nObjPerElement[iEl]; ++iObj) {
      if (iEl == 0) { // subsensors (mNumSubSensorsHalfLayer sectors with dead zones)
        if (iObj == 0) {
          halfLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rmin, rmax, mZLen / 2, TMath::RadToDeg() * mFringeChipWidth / rmed, TMath::RadToDeg() * (mFringeChipWidth + widthSensor) / rmed));
        } else if (iObj == mNumSubSensorsHalfLayer - 1) {
          halfLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rmin, rmax, mZLen / 2, TMath::RadToDeg() * (mFringeChipWidth + iObj * widthSensor + iObj * mMiddleChipWidth) / rmed, TMath::RadToDeg() * TMath::Pi() - TMath::RadToDeg() * mFringeChipWidth / rmed));
        } else {
          halfLayer[iEl].push_back(new TGeoTubeSeg(Form("subsens%dlayer%d", iObj, mLayerNumber), rmin, rmax, mZLen / 2, TMath::RadToDeg() * (mFringeChipWidth + iObj * widthSensor + iObj * mMiddleChipWidth) / rmed, TMath::RadToDeg() * (mFringeChipWidth + (iObj + 1) * widthSensor + iObj * mMiddleChipWidth) / rmed));
        }
      } else { // all the others are simply half cylinders filling all the space
        halfLayer[iEl].push_back(new TGeoTubeSeg(rmin, rmax + radiusBetweenLayer, mZLen / 2, 0., TMath::RadToDeg() * TMath::Pi()));
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
        createCarbonFoamStructure(volHalfLayer[iEl]);
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

  TGeoTranslation* translationTop = new TGeoTranslation(0., mGap / 2, 0.);
  TGeoTranslation* translationBottom = new TGeoTranslation(0., -mGap / 2, 0.);
  TGeoRotation* rotationBottom = new TGeoRotation("", 180., 0., 0.);
  TGeoCombiTrans* rotoTranslationBottom =
    new TGeoCombiTrans(*translationBottom, *rotationBottom);

  TGeoVolumeAssembly* volLayer = new TGeoVolumeAssembly(names[nElements - 1].data());
  volLayer->AddNode(volHalfLayer[nElements - 2], 0, translationTop);
  volLayer->AddNode(volHalfLayer[nElements - 2], 1, rotoTranslationBottom);

  // Finally put everything in the mother volume
  LOGP(debug, "Inserting {} inside {}", volLayer->GetName(), motherVolume->GetName());
  motherVolume->AddNode(volLayer, 1, nullptr);
}

void ITS3Layer::createCarbonFoamStructure(TGeoVolume* motherVolume)
{
  TGeoMedium* medCarbonFoam = gGeoManager->GetMedium("IT3_ERGDUOCEL$");
  TGeoMedium* medGlue = gGeoManager->GetMedium("IT3_IMPREG_FLEECE$");

  double rmax = mRadius + mSensorThickness;
  double radiusBetweenLayer = 0.6 - mSensorThickness; // FIXME: hard coded distance between layers
  double rmedFoam = rmax + radiusBetweenLayer / 2;

  TGeoTranslation* transSemicircle[2];
  transSemicircle[0] = new TGeoTranslation("transSemicircleFoam0", 0, 0, (mZLen - mLengthSemiCircleFoam) / 2);
  transSemicircle[0]->RegisterYourself();
  transSemicircle[1] = new TGeoTranslation("transSemicircleFoam1", 0, 0, -(mZLen - mLengthSemiCircleFoam) / 2);
  transSemicircle[1]->RegisterYourself();

  TGeoTubeSeg* subGluedFoam[4];
  subGluedFoam[0] = new TGeoTubeSeg(Form("subgluedfoam0layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, mLengthSemiCircleFoam / 2, 0., TMath::RadToDeg() * TMath::Pi());
  subGluedFoam[1] = new TGeoTubeSeg(Form("subgluedfoam1layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, mLengthSemiCircleFoam / 2, 0., TMath::RadToDeg() * TMath::Pi());
  subGluedFoam[2] = new TGeoTubeSeg(Form("subgluedfoam2layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, (mZLen - mLengthSemiCircleFoam) / 2, 0., TMath::RadToDeg() * mHeightStripFoam / rmedFoam);
  subGluedFoam[3] = new TGeoTubeSeg(Form("subgluedfoam3layer%d", mLayerNumber), rmax, rmax + mThickGluedFoam, (mZLen - mLengthSemiCircleFoam) / 2, TMath::RadToDeg() * (TMath::Pi() - (mHeightStripFoam / rmedFoam)), TMath::RadToDeg() * TMath::Pi());

  std::string subGluedFoamsNames = "";
  for (int iObj{0}; iObj < 2; ++iObj) {
    subGluedFoamsNames += Form("(subgluedfoam%dlayer%d:transSemicircleFoam%d)+", iObj, mLayerNumber, iObj);
  }
  subGluedFoamsNames += Form("subgluedfoam2layer%d+", mLayerNumber);
  subGluedFoamsNames += Form("subgluedfoam3layer%d", mLayerNumber);

  TGeoCompositeShape* gluedfoam = new TGeoCompositeShape(subGluedFoamsNames.data());
  TGeoVolume* volGlue = new TGeoVolume(Form("Glue%d", mLayerNumber), gluedfoam, medGlue);
  motherVolume->AddNode(volGlue, 1, nullptr);

  TGeoTubeSeg* subFoam[4];
  subFoam[0] = new TGeoTubeSeg(Form("subfoam0layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer, mLengthSemiCircleFoam / 2, 0., TMath::RadToDeg() * TMath::Pi());
  subFoam[1] = new TGeoTubeSeg(Form("subfoam1layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer, mLengthSemiCircleFoam / 2, 0., TMath::RadToDeg() * TMath::Pi());
  subFoam[2] = new TGeoTubeSeg(Form("subfoam2layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer, (mZLen - mLengthSemiCircleFoam) / 2, 0., TMath::RadToDeg() * mHeightStripFoam / rmedFoam);
  subFoam[3] = new TGeoTubeSeg(Form("subfoam3layer%d", mLayerNumber), rmax + mThickGluedFoam, rmax + radiusBetweenLayer, (mZLen - mLengthSemiCircleFoam) / 2, TMath::RadToDeg() * (TMath::Pi() - (mHeightStripFoam / rmedFoam)), TMath::RadToDeg() * TMath::Pi());

  std::string subFoamNames = "";
  for (int iObj{0}; iObj < 2; ++iObj) {
    subFoamNames += Form("(subfoam%dlayer%d:transSemicircleFoam%d)+", iObj, mLayerNumber, iObj);
  }
  subFoamNames += Form("subfoam2layer%d+", mLayerNumber);
  subFoamNames += Form("subfoam3layer%d", mLayerNumber);

  TGeoCompositeShape* foam = new TGeoCompositeShape(subFoamNames.data());
  TGeoVolume* volFoam = new TGeoVolume(Form("CarbonFoam%d", mLayerNumber), foam, medCarbonFoam);
  motherVolume->AddNode(volFoam, 1, nullptr);
}
