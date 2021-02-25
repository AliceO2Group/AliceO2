// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EC0Layer.cxx
/// \brief Implementation of the EC0Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#include "EC0Simulation/EC0Layer.h"
#include "EC0Base/GeometryTGeo.h"
#include "EC0Simulation/Detector.h"

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
using namespace o2::ec0;
using namespace o2::itsmft;

ClassImp(EC0Layer);

EC0Layer::EC0Layer(Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut, Float_t sensorThickness, Float_t Layerx2X0)
{
  // Creates a simple parametrized EndCap layer covering the given
  // pseudorapidity range at the z layer position
  mLayerNumber = layerNumber;
  mLayerName = layerName;
  mZ = z;
  mx2X0 = Layerx2X0;
  mSensorThickness = sensorThickness;
  mInnerRadius = rIn;
  mOuterRadius = rOut;

  LOG(INFO) << " Using silicon Radiation Length =  " << 9.5 << " to emulate layer radiation length.";

  mChipThickness = Layerx2X0 * 9.5;
  if (mChipThickness < mSensorThickness) {
    LOG(INFO) << " WARNING: Chip cannot be thinner than sensor. Setting minimal chip thickness.";
    mChipThickness = mSensorThickness;
  }
  LOG(INFO) << "Creating EC0 Layer " << mLayerNumber << ": z = " << mZ << " ; R_in = " << mInnerRadius << " ; R_out = " << mOuterRadius << " ; ChipThickness = " << mChipThickness;
}

void EC0Layer::createLayer(TGeoVolume* motherVolume)
{
  if (mLayerNumber >= 0) {
    // Create tube, set sensitive volume, add to mother volume

    std::string chipName = o2::ec0::GeometryTGeo::getEC0ChipPattern() + std::to_string(mLayerNumber),
                sensName = o2::ec0::GeometryTGeo::getEC0SensorPattern() + std::to_string(mLayerNumber);

    TGeoTube* sensor = new TGeoTube(mInnerRadius, mOuterRadius, mSensorThickness / 2);
    TGeoTube* chip = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);
    TGeoTube* layer = new TGeoTube(mInnerRadius, mOuterRadius, mChipThickness / 2);

    TGeoMedium* medSi = gGeoManager->GetMedium("EC0_SI$");
    TGeoMedium* medAir = gGeoManager->GetMedium("EC0_AIR$");

    TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
    TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
    TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), layer, medAir);

    LOG(INFO) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
    chipVol->AddNode(sensVol, 1, nullptr);

    LOG(INFO) << "Inserting " << chipVol->GetName() << " inside " << layerVol->GetName();
    layerVol->AddNode(chipVol, 1, nullptr);

    // Finally put everything in the mother volume
    auto* FwdDiskRotation = new TGeoRotation("FwdDiskRotation", 0, 0, 180);
    auto* FwdDiskCombiTrans = new TGeoCombiTrans(0, 0, mZ, FwdDiskRotation);

    LOG(INFO) << "Inserting " << layerVol->GetName() << " inside " << motherVolume->GetName();
    motherVolume->AddNode(layerVol, 1, FwdDiskCombiTrans);

    return;
  }
}
