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

#include <TGeoBBox.h>         // for TGeoBBox
#include <TGeoManager.h>      // for gGeoManager, TGeoManager
#include "TGeoVolume.h"       // for TGeoVolume
#include "TGeoMatrix.h"       // for TGeoMatrix
#include "FairLogger.h" // for LOG

using namespace o2::ITSMFT;

ClassImp(AlpideChip)

//________________________________________________________________________
TGeoVolume* AlpideChip::createChip(const Double_t ychip,
                                   const Double_t ysens,
                                   const char* chipName,
                                   const char* sensName,
                                   const Bool_t dummy,
                                   const TGeoManager *mgr){
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

  xchip = 0.5*SegmentationAlpide::SensorSizeRows;
  zchip = 0.5*SegmentationAlpide::SensorSizeCols;

  // First create all needed shapes
  ylen = ysens;
  if (ysens > ychip) {
    LOG(WARNING) << "Sensor half thickness (" << ysens
                 << ") greater than chip half thickness (" << ychip
                 << "), setting equal" << FairLogger::endl;
    ylen = ychip;
  }

  // The chip
  TGeoBBox *chip = new TGeoBBox(xchip,  ychip, zchip);

  // The sensor
  TGeoBBox *sensor = new TGeoBBox(xchip, ylen, zchip);

  // The metal layer
  ylen = 0.5*sMetalLayerThick;
  TGeoBBox *metallay = new TGeoBBox(xchip, ylen, zchip);


  // We have all shapes: now create the real volumes
  TGeoMedium *medSi    = mgr->GetMedium("ITS_SI$");
  TGeoMedium *medAir   = mgr->GetMedium("ITS_AIR$");
  TGeoMedium *medMetal = mgr->GetMedium("ITS_METALSTACK$");
  TGeoMedium *medChip;

  if (dummy)
    medChip = medAir;
  else
    medChip = medSi;

  TGeoVolume *chipVol = new TGeoVolume(chipName, chip, medChip);
  chipVol->SetVisibility(kTRUE);
  chipVol->SetLineColor(1);
  chipVol->SetFillColor(chipVol->GetLineColor());
  chipVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *sensVol = new TGeoVolume(sensName, sensor, medChip);
  sensVol->SetVisibility(kTRUE);
  sensVol->SetLineColor(8);
  sensVol->SetLineWidth(1);
  sensVol->SetFillColor(sensVol->GetLineColor());
  sensVol->SetFillStyle(4000); // 0% transparent

  TGeoVolume *metalVol = new TGeoVolume("MetalStack", metallay, medMetal);
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
