// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.h
/// \brief Implementation of the conversion from points to digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "FairLogger.h"

#include "MFTSimulation/Digitizer.h"

#include "ITSMFTSimulation/Hit.h"

ClassImp(o2::MFT::Digitizer)

using o2::ITSMFT::Hit;
using o2::ITSMFT::Chip;
using o2::ITSMFT::SimulationAlpide;
using o2::ITSMFT::Digit;
using o2::ITSMFT::SegmentationPixel;

using namespace o2::MFT;

//_____________________________________________________________________________
Digitizer::Digitizer() : mGeometry(), mNumOfChips(0), mChips(), mSimulations(), mDigitContainer() {}

//_____________________________________________________________________________
Digitizer::~Digitizer() = default;

//_____________________________________________________________________________
void Digitizer::init(Bool_t build)
{

}

//_____________________________________________________________________________
void Digitizer::process(TClonesArray* points, TClonesArray* digits)
{

  // Convert points to digits
  for (TIter iter = TIter(points).Begin(); iter != TIter::End(); ++iter) {
    Hit* point = dynamic_cast<Hit*>(*iter);
    Int_t chipID = point->GetDetectorID();
    if (chipID >= mNumOfChips)
      continue;
    mChips[chipID].InsertHit(point);
  }

  for (Int_t i = 0; i < mNumOfChips; i++) {
  }

}

//_____________________________________________________________________________
DigitContainer& Digitizer::process(TClonesArray* hits)
{

  mDigitContainer.reset();
  /*
  // Convert hits to digits
  for (TIter hititer = TIter(hits).Begin(); hititer != TIter::End(); ++hititer) {
    Hit* hit = dynamic_cast<Hit*>(*hititer);
    
    LOG(DEBUG) << "Processing next hit: " << FairLogger::endl;
    LOG(DEBUG) << "=======================" << FairLogger::endl;
    LOG(DEBUG) << *hit << FairLogger::endl;

    Double_t x = 0.5 * (hit->GetX() + hit->GetStartX());
    Double_t y = 0.5 * (hit->GetY() + hit->GetStartY());
    Double_t z = 0.5 * (hit->GetZ() + hit->GetStartZ());
    Double_t charge = hit->GetEnergyLoss();
    Int_t label = hit->GetTrackID();
    Int_t chipID = hit->GetDetectorID();

    LOG(DEBUG) << "Creating new digit" << FairLogger::endl;
    const Double_t glo[3] = { x, y, z };
    Double_t loc[3] = { 0., 0., 0. };
    mGeometry.globalToLocal(chipID, glo, loc);
    const SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);
    Int_t ix, iz;
    seg->localToDetector(loc[0], loc[2], ix, iz);
    if ((ix < 0) || (iz < 0)) {
      LOG(DEBUG) << "Out of the chip" << FairLogger::endl;
      continue;
    }
    Digit* digit = mDigitContainer.addDigit(chipID, ix, iz, charge, hit->GetTime());
    digit->setLabel(0, label);
  }
  */
  return mDigitContainer;

}

