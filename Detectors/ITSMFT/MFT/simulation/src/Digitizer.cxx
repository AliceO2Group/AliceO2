// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.h
/// \brief Implementation of the conversion from hits to digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "FairLogger.h"

#include "ITSMFTSimulation/DigitChip.h"
#include "ITSMFTSimulation/DigitContainer.h"

#include "MFTSimulation/Digitizer.h"

#include "ITSMFTSimulation/Hit.h"

ClassImp(o2::MFT::Digitizer)

using o2::ITSMFT::Hit;
using o2::ITSMFT::Chip;
using o2::ITSMFT::SimulationAlpide;
using o2::ITSMFT::Digit;
using o2::ITSMFT::DigitChip;
using o2::ITSMFT::DigitContainer;
using o2::ITSMFT::SegmentationPixel;

using namespace o2::MFT;

//_____________________________________________________________________________
Digitizer::Digitizer() : mGeometry(), mSimulations(), mDigitContainer() {}

//_____________________________________________________________________________
Digitizer::~Digitizer() = default;

//_____________________________________________________________________________
void Digitizer::init(Bool_t build)
{

  mGeometry.build(kTRUE);

  const Int_t numOfChips = mGeometry.getNumberOfChips();

  mDigitContainer.resize(numOfChips);

  SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);
  DigitChip::setNumberOfRows(seg->getNumberOfRows());

  Double_t param[] = {
    50,     // ALPIDE threshold
    -1.315, // ACSFromBGPar0
    0.5018, // ACSFromBGPar1
    1.084,  // ACSFromBGPar2
    0.      // ALPIDE Noise per chip
  };
  for (Int_t i = 0; i < numOfChips; i++) {
    mSimulations.emplace_back(param, i, mGeometry.getMatrixSensorToITS(i));
  }

}

//_____________________________________________________________________________
void Digitizer::process(TClonesArray* hits, TClonesArray* digits)
{

  const Int_t numOfChips = mGeometry.getNumberOfChips();
  mDigitContainer.reset();

  // Convert hits to digits
  const SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);
  for (TIter iter = TIter(hits).Begin(); iter != TIter::End(); ++iter) {
    Hit* hit = dynamic_cast<Hit*>(*iter);
    Int_t chipID = hit->GetDetectorID();
    if (chipID >= numOfChips)
      continue;
    mSimulations[chipID].InsertHit(hit);
  }

  for (auto &simulation : mSimulations) {
    simulation.generateClusters(seg, &mDigitContainer);
    simulation.clearSimulation();
  }

  mDigitContainer.fillOutputContainer(digits);

}

//_____________________________________________________________________________
DigitContainer& Digitizer::process(TClonesArray* hits)
{

  mDigitContainer.reset();

  return mDigitContainer;

}

