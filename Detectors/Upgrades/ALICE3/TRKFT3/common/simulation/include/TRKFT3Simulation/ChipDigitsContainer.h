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

#ifndef ALICEO2_TRK_CHIPDIGITSCONTAINER_
#define ALICEO2_TRK_CHIPDIGITSCONTAINER_

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/ChipDigitsContainer.h"
#include "TRKBase/SegmentationChip.h"
#include "TRKBase/Specs.h"
#include "TRKFT3Simulation/DigiParams.h"
#include <fairlogger/Logger.h>
#include <TRandom.h>

namespace o2::trkft3
{

class ChipDigitsContainer : public o2::itsmft::ChipDigitsContainer
{
 public:
  explicit ChipDigitsContainer(UShort_t idx = 0);

  using Segmentation = o2::trk::SegmentationChip;

  /// Get global ordering key made of readout frame, column and row
  static ULong64_t getOrderingKey(UInt_t roframe, UShort_t row, UShort_t col)
  {
    return (static_cast<ULong64_t>(roframe) << (8 * sizeof(UInt_t))) + (static_cast<ULong64_t>(col) << (8 * sizeof(Short_t))) + row;
  }

  /// Adds noise digits, deleted the one using the itsmft::DigiParams interface
  void addNoise(UInt_t rofMin, UInt_t rofMax, const o2::itsmft::DigiParams* params, int maxRows = o2::itsmft::SegmentationAlpide::NRows, int maxCols = o2::itsmft::SegmentationAlpide::NCols) = delete;
  template <int DetID>
  void addNoise(UInt_t rofMin, UInt_t rofMax, const o2::trkft3::DigiParams<DetID>* params, int subDetID, int layer);

  ClassDefNV(ChipDigitsContainer, 1);
};

} // namespace o2::trkft3

template <int DetID>
void o2::trkft3::ChipDigitsContainer::addNoise(UInt_t rofMin, UInt_t rofMax, const o2::trkft3::DigiParams<DetID>* params, int subDetID, int layer)
{
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  float mean = 0.f;
  int nel = 0;
  int maxRows = 0;
  int maxCols = 0;

  if (subDetID == 0) {
    maxRows = o2::trk::constants::VD::petal::layer::nRows[layer];
    maxCols = o2::trk::constants::VD::petal::layer::nCols;
  } else {
    maxRows = o2::trk::constants::moduleMLOT::chip::nRows;
    maxCols = o2::trk::constants::moduleMLOT::chip::nCols;
  }
  mean = params->getNoisePerPixel() * maxRows * maxCols;
  nel = static_cast<int>(params->getChargeThreshold() * 1.1);

  LOG(debug) << "Adding noise for chip " << mChipIndex << " with mean " << mean << " and charge " << nel;

  for (UInt_t rof = rofMin; rof <= rofMax; rof++) {
    nhits = gRandom->Poisson(mean);
    for (Int_t i = 0; i < nhits; ++i) {
      row = gRandom->Integer(maxRows);
      col = gRandom->Integer(maxCols);
      LOG(debug) << "Generated noise hit at ROF " << rof << ", row " << row << ", col " << col;
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

#endif // ALICEO2_TRK_CHIPDIGITSCONTAINER_
