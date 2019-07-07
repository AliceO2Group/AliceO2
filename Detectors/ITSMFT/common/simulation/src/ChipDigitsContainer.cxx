// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  ChipDigitsContainer.cpp
//  ALICEO2
//

#include "ITSMFTSimulation/ChipDigitsContainer.h"
#include "ITSMFTSimulation/DigiParams.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include <TRandom.h>

using namespace o2::itsmft;
using Segmentation = o2::itsmft::SegmentationAlpide;

ClassImp(o2::itsmft::ChipDigitsContainer);

//______________________________________________________________________
void ChipDigitsContainer::addNoise(UInt_t rofMin, UInt_t rofMax, const o2::itsmft::DigiParams* params)
{
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  constexpr float ns2sec = 1e-9;

  float mean = params->getNoisePerPixel() * Segmentation::NPixels;
  int nel = params->getChargeThreshold() * 1.1; // RS: TODO: need realistic spectrum of noise above the threshold

  for (UInt_t rof = rofMin; rof <= rofMax; rof++) {
    nhits = gRandom->Poisson(mean);
    for (Int_t i = 0; i < nhits; ++i) {
      row = gRandom->Integer(Segmentation::NRows);
      col = gRandom->Integer(Segmentation::NCols);
      // RS TODO: why the noise was added with 0 charge? It should be above the threshold!
      auto key = getOrderingKey(rof, row, col);
      if (!findDigit(key)) {
        addDigit(key, rof, row, col, nel, o2::MCCompLabel(true));
      }
    }
  }
}
