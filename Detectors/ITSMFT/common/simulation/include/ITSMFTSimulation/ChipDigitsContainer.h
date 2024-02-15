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

//
/// \file ChipDigitsContainer.h
/// \brief transient container for single chip digits accumulation
//

#ifndef ALICEO2_ITSMFT_CHIPDIGITSCONTAINER_
#define ALICEO2_ITSMFT_CHIPDIGITSCONTAINER_

#include "SimulationDataFormat/MCCompLabel.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/PreDigit.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include <map>

namespace o2
{
namespace itsmft
{
class DigiParams;

/// @class ChipDigitsContainer
/// @brief Container for similated points connected to a given chip

class ChipDigitsContainer
{
 public:
  /// Default constructor
  ChipDigitsContainer(UShort_t idx = 0) : mChipIndex(idx){};

  std::map<ULong64_t, o2::itsmft::PreDigit>& getPreDigits() { return mDigits; }
  bool isEmpty() const { return mDigits.empty(); }
  size_t getNDigits() const { return mDigits.size(); }
  void setNoiseMap(const o2::itsmft::NoiseMap* mp) { mNoiseMap = mp; }
  void setDeadChanMap(const o2::itsmft::NoiseMap* mp) { mDeadChanMap = mp; }
  void setChipIndex(UShort_t ind) { mChipIndex = ind; }
  UShort_t getChipIndex() const { return mChipIndex; }

  o2::itsmft::PreDigit* findDigit(ULong64_t key);
  void addDigit(ULong64_t key, UInt_t roframe, UShort_t row, UShort_t col, int charge, o2::MCCompLabel lbl);
  void addNoise(UInt_t rofMin, UInt_t rofMax, const o2::itsmft::DigiParams* params, int maxRows = o2::itsmft::SegmentationAlpide::NRows, int maxCols = o2::itsmft::SegmentationAlpide::NCols);

  /// Get global ordering key made of readout frame, column and row
  static ULong64_t getOrderingKey(UInt_t roframe, UShort_t row, UShort_t col)
  {
    return (static_cast<ULong64_t>(roframe) << (8 * sizeof(UInt_t))) + (col << (8 * sizeof(Short_t))) + row;
  }

  /// Get ROFrame from the ordering key
  static UInt_t key2ROFrame(ULong64_t key)
  {
    return static_cast<UInt_t>(key >> (8 * sizeof(UInt_t)));
  }

  bool isDisabled() const { return mDisabled; }
  void disable(bool v) { mDisabled = v; }

 protected:
  UShort_t mChipIndex = 0;                           ///< chip index
  bool mDisabled = false;
  const o2::itsmft::NoiseMap* mNoiseMap = nullptr;
  const o2::itsmft::NoiseMap* mDeadChanMap = nullptr;
  std::map<ULong64_t, o2::itsmft::PreDigit> mDigits; ///< Map of fired pixels, possibly in multiple frames

  ClassDefNV(ChipDigitsContainer, 1);
};

//_______________________________________________________________________
inline o2::itsmft::PreDigit* ChipDigitsContainer::findDigit(ULong64_t key)
{
  // finds the digit corresponding to global key
  auto digitentry = mDigits.find(key);
  return digitentry != mDigits.end() ? &(digitentry->second) : nullptr;
}

//_______________________________________________________________________
inline void ChipDigitsContainer::addDigit(ULong64_t key, UInt_t roframe, UShort_t row, UShort_t col,
                                          int charge, o2::MCCompLabel lbl)
{
  mDigits.emplace(std::make_pair(key, o2::itsmft::PreDigit(roframe, row, col, charge, lbl)));
}
} // namespace itsmft
} // namespace o2

#endif /* defined(ALICEO2_ITSMFT_CHIPCONTAINER_) */
