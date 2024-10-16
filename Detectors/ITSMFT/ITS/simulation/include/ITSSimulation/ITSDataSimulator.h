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

/// \file   ITSDataSimulator.h
/// \brief  Infrastructure to simulate ALPIDE chip data.
/// \author knaumov@cern.ch

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTReconstruction/PixelData.h"

namespace o2
{
namespace itsmft
{

class ITSDataSimulator
{
 public:
  const static uint32_t MaxChipID = 24119;
  const static uint32_t MaxPixelsPerChip =
    SegmentationAlpide::NRows * SegmentationAlpide::NCols;

  ITSDataSimulator(int32_t seed, uint32_t numberOfChips,
                   uint32_t maxPixelsPerChip, bool doDigits, bool doErrors)
    : mSeed(seed), mNumberOfChips(numberOfChips), mMaxPixelsPerChip(maxPixelsPerChip), mDoDigits(doDigits), mDoErrors(doErrors)
  {
    srand(mSeed);
  }

  ~ITSDataSimulator() = default;

  // Simulate fired pixels for a chip
  std::vector<PixelData> generateChipData();

  void simulate();

 private:
  int32_t mSeed;
  uint32_t mMaxPixelsPerChip;
  uint32_t mNumberOfChips;
  bool mDoDigits;
  bool mDoErrors;
};

} // namespace itsmft
} // namespace o2
