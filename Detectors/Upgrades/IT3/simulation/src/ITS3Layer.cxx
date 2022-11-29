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

#include <TGeoTube.h>   // for TGeoTube, TGeoTubeSeg
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
    halfLayer[iEl] = new TGeoTubeSeg(rmin, rmax, mZLen / 2, 0., TMath::RadToDeg() * TMath::Pi());
    volHalfLayer[iEl] = new TGeoVolume(names[iEl].data(), halfLayer[iEl], med);
    volHalfLayer[iEl]->SetUniqueID(mChipTypeID);
    if (iEl == 0) {
      volHalfLayer[iEl]->SetVisibility(true);
      volHalfLayer[iEl]->SetLineColor(kRed + 1);
    }

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

  return;
}