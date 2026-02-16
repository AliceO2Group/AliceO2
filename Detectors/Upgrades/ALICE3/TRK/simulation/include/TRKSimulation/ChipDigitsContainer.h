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
#include "TRKSimulation/DigiParams.h"
#include <TRandom.h>

namespace o2::trk
{

class ChipDigitsContainer : public o2::itsmft::ChipDigitsContainer
{
 public:
  explicit ChipDigitsContainer(UShort_t idx = 0);

  using Segmentation = SegmentationChip;

  /// Get global ordering key made of readout frame, column and row
  static ULong64_t getOrderingKey(UInt_t roframe, UShort_t row, UShort_t col)
  {
    return (static_cast<ULong64_t>(roframe) << (8 * sizeof(UInt_t))) + (static_cast<ULong64_t>(col) << (8 * sizeof(Short_t))) + row;
  }

  ClassDefNV(ChipDigitsContainer, 1);
};

} // namespace o2::trk

#endif // ALICEO2_TRK_CHIPDIGITSCONTAINER_
