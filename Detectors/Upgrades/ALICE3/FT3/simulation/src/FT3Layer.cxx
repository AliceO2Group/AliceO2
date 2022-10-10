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
#include "FT3Simulation/Detector.h"

#include <fairlogger/Logger.h> // for LOG

#include <TGeoManager.h>        // for TGeoManager, gGeoManager
#include <TGeoMatrix.h>         // for TGeoCombiTrans, TGeoRotation, etc
#include <TGeoTube.h>           // for TGeoTube, TGeoTubeSeg
#include <TGeoVolume.h>         // for TGeoVolume, TGeoVolumeAssembly
#include <TGeoCompositeShape.h> // for TGeoCompositeShape
#include "TMathBase.h"          // for Abs
#include <TMath.h>              // for Sin, RadToDeg, DegToRad, Cos, Tan, etc

#include <cstdio> // for snprintf

class TGeoMedium;

using namespace TMath;
using namespace o2::ft3;
using namespace o2::itsmft;

ClassImp(FT3Layer);

FT3Layer::~FT3Layer() = default;

FT3Layer::FT3Layer(Int_t layerDirection, Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut, Float_t Layerx2X0)
{
  // Creates a simple parametrized EndCap layer covering the given
  // pseudorapidity range at the z layer position
  mDirection = layerDirection;
  mLayerNumber = layerNumber;
  mLayerName = layerName;
  mZ = layerDirection ? std::abs(z) : -std::abs(z);
  mx2X0 = Layerx2X0;
  mInnerRadius = rIn;
  mOuterRadius = rOut;
  auto Si_X0 = 9.5;
  mChipThickness = Layerx2X0 * Si_X0;

  LOG(info) << "Creating FT3 Layer " << mLayerNumber << " ; direction " << mDirection;
  LOG(info) << "   Using silicon X0 = " << Si_X0 << " to emulate layer radiation length.";
  LOG(info) << "   Layer z = " << mZ << " ; R_in = " << mInnerRadius << " ; R_out = " << mOuterRadius << " ; x2X0 = " << mx2X0 << " ; ChipThickness = " << mChipThickness;
}

void FT3Layer::createLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber >= 0) {
    // Create tube, set sensitive volume, add to mother volume

    std::string chipName = o2::ft3::GeometryTGeo::getFT3ChipPattern() + std::to_string(mLayerNumber),
                sensName = Form("%s_%d_%d", GeometryTGeo::getFT3SensorPattern(), mDirection, mLayerNumber);
    TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
    TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
    TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);

    TGeoMedium* medSi = gGeoManager->GetMedium("FT3_SI$");
    TGeoMedium* medAir = gGeoManager->GetMedium("FT3_AIR$");

    TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
    TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
    TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);

    LOG(info) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
    chipVol->AddNode(sensVol, 1, nullptr);

    LOG(info) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
    layerVol->AddNode(chipVol, 1, nullptr);

    // Finally put everything in the mother volume
    auto* FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
    auto* FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

    LOG(info) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
    motherVolume->AddNode(layerVol, 1, FwdDiskCombiTrans);

    return;
  }
}
