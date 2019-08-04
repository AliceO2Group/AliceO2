// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideChip.cxx
/// \brief Creates an ALPIDE chip in simulation
/// \author Mario.Sitta@cern.ch - 24 oct 2017

#include "ITSMFTSimulation/AlpideChip.h"
#include "ITSMFTBase/SegmentationAlpide.h"

#include "DetectorsBase/MaterialManager.h"
#include "DetectorsBase/Detector.h"

#include <TGeoBBox.h>    // for TGeoBBox
#include <TGeoManager.h> // for gGeoManager, TGeoManager
#include "TGeoVolume.h"  // for TGeoVolume
#include "TGeoMatrix.h"  // for TGeoMatrix
#include "FairLogger.h"  // for LOG

using namespace o2::itsmft;

ClassImp(AlpideChip);

//________________________________________________________________________
TGeoVolume* AlpideChip::createChip(const Double_t ychip,
                                   const Double_t ysens,
                                   const char* chipName,
                                   const char* sensName,
                                   const Bool_t dummy,
                                   const TGeoManager* mgr)
{
  //
  // Creates the Alpide Chip
  // Caller should then use TGeoVolume::SetName to proper set the volume name
  //
  // Input:
  //         ychip : the chip Y half dimensions
  //         ysens : the sensor half thickness
  //         chipName,sensName : default volume names (if not passed by caller)
  //         dummy : if true, creates a dummy air volume
  //                 (for material budget studies)
  //         mgr  : the GeoManager (used only to get the proper material)
  //
  // Output:
  //
  // Return:
  //
  // Created:      20 Oct 2017  Mario Sitta  Ported from V3layer
  //

  Double_t xchip, zchip;
  Double_t ylen, ypos;

  xchip = 0.5 * SegmentationAlpide::SensorSizeRows;
  zchip = 0.5 * SegmentationAlpide::SensorSizeCols;

  // First create all needed shapes
  ylen = ysens;
  if (ysens > ychip) {
    LOG(WARNING) << "Sensor half thickness (" << ysens
                 << ") greater than chip half thickness (" << ychip
                 << "), setting equal" << FairLogger::endl;
    ylen = ychip;
  }

  // The chip
  TGeoBBox* chip = new TGeoBBox(xchip, ychip, zchip);

  // The sensor
  TGeoBBox* sensor = new TGeoBBox(xchip, ylen, zchip);

  // The metal layer
  ylen = 0.5 * sMetalLayerThick;
  TGeoBBox* metallay = new TGeoBBox(xchip, ylen, zchip);

  // We have all shapes: now create the real volumes
  TGeoMedium* medSi = mgr->GetMedium("ALPIDE_SI$");
  if (!medSi) {
    Int_t fieldType;
    Float_t maxField;
    o2::base::Detector::initFieldTrackingParams(fieldType, maxField);
    createMaterials(0, fieldType, maxField);
    medSi = mgr->GetMedium("ALPIDE_SI$");
  }
  TGeoMedium* medAir = mgr->GetMedium("ALPIDE_AIR$");
  TGeoMedium* medMetal = mgr->GetMedium("ALPIDE_METALSTACK$");

  TGeoMedium* medChip;

  if (dummy)
    medChip = medAir;
  else
    medChip = medSi;

  TGeoVolume* chipVol = new TGeoVolume(chipName, chip, medChip);
  chipVol->SetVisibility(kTRUE);
  chipVol->SetLineColor(1);
  chipVol->SetFillColor(chipVol->GetLineColor());
  chipVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* sensVol = new TGeoVolume(sensName, sensor, medChip);
  sensVol->SetVisibility(kTRUE);
  sensVol->SetLineColor(8);
  sensVol->SetLineWidth(1);
  sensVol->SetFillColor(sensVol->GetLineColor());
  sensVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume* metalVol = new TGeoVolume("MetalStack", metallay, medMetal);
  metalVol->SetVisibility(kTRUE);
  metalVol->SetLineColor(1);
  metalVol->SetLineWidth(1);
  metalVol->SetFillColor(metalVol->GetLineColor());
  metalVol->SetFillStyle(4000); // 0% transparent

  // Now build up the chip
  ypos = chip->GetDY() - metallay->GetDY();
  chipVol->AddNode(metalVol, 1, new TGeoTranslation(0., ypos, 0.));

  ypos -= (metallay->GetDY() + sensor->GetDY());
  chipVol->AddNode(sensVol, 1, new TGeoTranslation(0., ypos, 0.));

  // Done, return the chip
  return chipVol;
}

//________________________________________________________________________
void AlpideChip::createMaterials(Int_t id, Int_t ifield, Float_t fieldm)
{
  // create common materials with the MFT

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

  // BEOL (Metal interconnection stack in Si sensors)
  Float_t aBEOL[3] = {26.982, 28.086, 15.999};
  Float_t zBEOL[3] = {13, 14, 8}; // Al, Si, O
  Float_t wBEOL[3] = {0.170, 0.388, 0.442};
  Float_t dBEOL = 2.28;

  // AIR
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;

  if (mgr.getMediumID("ALPIDE", id) < 0) {
    mgr.Mixture("ALPIDE", id, "METALSTACK$", aBEOL, zBEOL, dBEOL, 3, wBEOL);
    mgr.Medium("ALPIDE", id, "METALSTACK$", id, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);
    id++;
  }

  if (mgr.getMediumID("ALPIDE", id) < 0) {
    mgr.Mixture("ALPIDE", id, "AIR$", aAir, zAir, dAir, 4, wAir);
    mgr.Medium("ALPIDE", id, "AIR$", id, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);
    id++;
  }

  if (mgr.getMediumID("ALPIDE", id) < 0) {
    mgr.Material("ALPIDE", id, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
    mgr.Medium("ALPIDE", id, "SI$", id, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  }
}
