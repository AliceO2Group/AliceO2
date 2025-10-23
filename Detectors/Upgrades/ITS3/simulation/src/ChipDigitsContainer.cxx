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

#include "ITS3Simulation/ChipDigitsContainer.h"

using namespace o2::its3;

ChipDigitsContainer::ChipDigitsContainer(UShort_t idx)
  : o2::itsmft::ChipDigitsContainer(idx) {}

bool ChipDigitsContainer::isIB() const
{
  return innerBarrel;
}

void ChipDigitsContainer::addNoise(UInt_t rofMin, UInt_t rofMax, const o2::its3::DigiParams* params)
{
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  constexpr float ns2sec = 1e-9;
  float mean = 0.f;
  int nel = 0;

  if (isIB()) {
    // Inner barrel: use ITS3-specific noise interface with IB segmentation.
    mean = params->getIBNoisePerPixel() * SegmentationIB::NPixels;
    nel = static_cast<int>(params->getIBChargeThreshold() * 1.1);
  } else {
    // Outer barrel: use base class noise interface with OB segmentation.
    mean = params->getNoisePerPixel() * SegmentationOB::NPixels;
    nel = static_cast<int>(params->getChargeThreshold() * 1.1);
  }

  for (UInt_t rof = rofMin; rof <= rofMax; ++rof) {
    nhits = gRandom->Poisson(mean);
    for (Int_t i = 0; i < nhits; ++i) {
      row = gRandom->Integer(maxRows);
      col = gRandom->Integer(maxCols);
      if (mNoiseMap && mNoiseMap->isNoisy(mChipIndex, row, col)) {
        continue;
      }
      if (mDeadChanMap && mDeadChanMap->isNoisy(mChipIndex, row, col)) {
        continue;
      }
      auto key = getOrderingKey(rof, row, col);
      if (!findDigit(key)) {
        addDigit(key, rof, row, col, nel, o2::MCCompLabel(true));
      }
    }
  }
}
// namespace its3
