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

#include "IOTOFSimulation/TF3Chip.h"

#include "DetectorsBase/MaterialManager.h"
#include "DetectorsBase/Detector.h"

#include <TGeoBBox.h>          // for TGeoBBox
#include <TGeoManager.h>       // for gGeoManager, TGeoManager
#include "TGeoVolume.h"        // for TGeoVolume
#include "TGeoMatrix.h"        // for TGeoMatrix
#include <fairlogger/Logger.h> // for LOG

namespace o2
{

namespace iotof
{

const double TF3Chip::sChipSizeX = 2.6; // cm
const double TF3Chip::sChipSizeZ = 3.0; // cm

//________________________________________________________________________
TGeoVolume* TF3Chip::createChip(const Double_t ychip,
                                const Double_t ysens,
                                const char* chipName,
                                const char* sensName,
                                const Bool_t dummy,
                                const TGeoManager* mgr)
{

  Double_t xchip, zchip;
  Double_t ylen, ypos;

  xchip = 0.5 * sChipSizeX;
  zchip = 0.5 * sChipSizeZ;

  // First create all needed shapes
  ylen = ysens;
  if (ysens > ychip) {
    LOG(warning) << "Sensor half thickness (" << ysens
                 << ") greater than chip half thickness (" << ychip
                 << "), setting equal";
    ylen = ychip;
  }

  // The chip
  TGeoBBox* chip = new TGeoBBox(xchip, 0.5 * ychip, zchip);

  // The sensor
  TGeoBBox* sensor = new TGeoBBox(xchip, 0.5 * ylen, zchip);

  // We have all shapes: now create the real volumes
  TGeoMedium* medSi = mgr->GetMedium("TF3CHIP_SI$");
  if (!medSi) {
    Int_t fieldType;
    Float_t maxField;
    o2::base::Detector::initFieldTrackingParams(fieldType, maxField);
    createMaterials(0, fieldType, maxField);
    medSi = mgr->GetMedium("TF3CHIP_SI$");
  }
  TGeoMedium* medAir = mgr->GetMedium("TF3CHIP_AIR$");

  TGeoMedium* medChip;

  if (dummy) {
    medChip = medAir;
  } else {
    medChip = medSi;
  }

  TGeoVolume* chipVol = new TGeoVolume(chipName, chip, medChip);
  chipVol->SetVisibility(kTRUE);
  chipVol->SetLineColor(kOrange);
  chipVol->SetFillColor(kOrange - 9);
  chipVol->SetLineWidth(3);
  chipVol->SetTransparency(20);

  TGeoVolume* sensVol = new TGeoVolume(sensName, sensor, medChip);
  sensVol->SetVisibility(kTRUE);
  sensVol->SetLineColor(kRed);
  sensVol->SetLineWidth(3);
  sensVol->SetFillColor(kRed - 9);
  sensVol->SetFillStyle(4000); // 0% transparent

  // Now build up the chip
  ypos = chip->GetDY() - sensor->GetDY();
  chipVol->AddNode(sensVol, 1, new TGeoTranslation(0., ypos, 0.));

  // Done, return the chip
  return chipVol;
}

//________________________________________________________________________
void TF3Chip::createMaterials(Int_t id, Int_t ifield, Float_t fieldm)
{
  auto& mgr = o2::base::MaterialManager::Instance();

  Float_t tmaxfd = 0.1;   // Degree
  Float_t stemax = 1.0;   // cm
  Float_t deemax = 0.1;   // Fraction of particle's energy 0<deemax<=1
  Float_t epsil = 1.0E-4; // cm
  Float_t stmin = 0.0;    // cm "Default value used"

  Float_t tmaxfdSi = 0.1;    // Degree
  Float_t stemaxSi = 0.0075; // cm
  Float_t deemaxSi = 0.1;    // Fraction of particle's energy 0<deemax<=1
  Float_t epsilSi = 1.0E-4;  // cm
  Float_t stminSi = 0.0;     // cm "Default value used"

  Float_t tmaxfdAir = 0.1;   // Degree
  Float_t stemaxAir = 1.0;   // cm
  Float_t deemaxAir = 0.1;   // Fraction of particle's energy 0<deemax<=1
  Float_t epsilAir = 1.0E-4; // cm
  Float_t stminAir = 0.0;    // cm "Default value used"

  // AIR
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;

  if (mgr.getMediumID("TF3CHIP", id) < 0) {
    mgr.Mixture("TF3CHIP", id, "AIR$", aAir, zAir, dAir, 4, wAir);
    mgr.Medium("TF3CHIP", id, "AIR$", id, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);
    id++;
  }

  if (mgr.getMediumID("TF3CHIP", id) < 0) {
    mgr.Material("TF3CHIP", id, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
    mgr.Medium("TF3CHIP", id, "SI$", id, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  }
}

} // namespace iotof
} // namespace o2
