// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PSRLayer.cxx
/// \brief Implementation of the PSR Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)
/// \author Abhishek Nath (aabhishek.naath@gmail.com)

#include "PSRSimulation/PSRLayer.h"
#include "PSRBase/GeometryTGeo.h"
#include "PSRSimulation/Detector.h"

#include "FairLogger.h" // for LOG

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
using namespace o2::psr;
using namespace o2::itsmft;

ClassImp(PSRLayer);

PSRLayer::~PSRLayer() = default;

//  PSRLayer(Int_t layerDirection, Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut, Float_t sensorThickness, Float_t Layerx2X0);
PSRLayer::PSRLayer(Int_t layerDirection, Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t Pb_t, Float_t sensorThickness, Float_t Layerx2X0)
{
  // Creates a simple parametrized EndCap layer covering the given
  // pseudorapidity range at the z layer position
  mDirection = layerDirection;
  mLayerNumber = layerNumber;
  mLayerName = layerName;
  mZ = layerDirection ? std::abs(z) : -std::abs(z);
  mx2X0 = Layerx2X0;
  mSensorThickness = sensorThickness;
  mInnerRadius = rIn;
  mPb_thick = Pb_t;

  LOG(INFO) << " Using silicon Radiation Length =  " << 9.5 << " to emulate layer radiation length.";

  mChipThickness = Layerx2X0 * 9.5;
  if (mChipThickness < mSensorThickness) {
    LOG(INFO) << " WARNING: Chip cannot be thinner than sensor. Setting minimal chip thickness.";
    mChipThickness = mSensorThickness;
  }
  LOG(INFO) << "Creating PSR Layer " << mLayerNumber << ": z = " << mZ << " ; R_Pb = " << mInnerRadius << " ; R_Si = " << mInnerRadius + mPb_thick << " ; ChipThickness = " << mChipThickness;
}

void PSRLayer::createLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber >= 0) {
    LOG(INFO) << "CHECKING 2";
    // Create tube, set sensitive volume, add to mother volume

    std::string showerlayerName = o2::psr::GeometryTGeo::getPSRShowerlayerPattern() + std::to_string(mLayerNumber),
                chipName = o2::psr::GeometryTGeo::getPSRChipPattern() + std::to_string(mLayerNumber),
                sensName = Form("%s_%d_%d", GeometryTGeo::getPSRSensorPattern(), mDirection, mLayerNumber);

    TGeoTube* showerlayer = new TGeoTube(mInnerRadius, mInnerRadius + mPb_thick, mZ / 2);
    TGeoTube* sensor = new TGeoTube(mInnerRadius + mPb_thick, mInnerRadius + mPb_thick + mSensorThickness, mZ / 2);
    TGeoTube* chip = new TGeoTube(mInnerRadius + mPb_thick, mInnerRadius + mPb_thick + mChipThickness, mZ / 2);
    TGeoTube* layer = new TGeoTube(mInnerRadius, mInnerRadius + mPb_thick + mChipThickness, mZ / 2);

    TGeoMedium* medSi = gGeoManager->GetMedium("PSR_SI$");
    TGeoMedium* medPb = gGeoManager->GetMedium("PSR_PB$");
    TGeoMedium* medAir = gGeoManager->GetMedium("PSR_AIR$");

    TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
    sensVol->SetVisibility(kTRUE);
    sensVol->SetLineColor(2);

    TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
    chipVol->SetVisibility(kTRUE);
    chipVol->SetLineColor(3);

    TGeoVolume* showerlayerVol = new TGeoVolume(showerlayerName.c_str(), chip, medPb);
    showerlayerVol->SetVisibility(kTRUE);
    showerlayerVol->SetLineColor(4);

    TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);
    layerVol->SetVisibility(kTRUE);
    layerVol->SetLineColor(5);

    LOG(INFO) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
    chipVol->AddNode(sensVol, 1, nullptr);

    LOG(INFO) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
    layerVol->AddNode(chipVol, 1, nullptr);

    LOG(INFO) << "Inserting "<< showerlayerVol->GetName() << " inside " << layerVol->GetName();
    layerVol->AddNode(showerlayerVol, 2, nullptr);

    LOG(INFO) << "Inserting " << showerlayerVol->GetName() << " inside " << layerVol->GetName();
    layerVol->AddNode(showerlayerVol, 2, nullptr);

    // Finally put everything in the mother volume
    //auto* FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
    //auto* FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

    LOG(INFO) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
    motherVolume->AddNode(layerVol, 1, nullptr);

    return;
  }
}
