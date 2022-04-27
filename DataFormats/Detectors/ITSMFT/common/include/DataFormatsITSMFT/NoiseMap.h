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

/// \file NoiseMap.h
/// \brief Definition of the ITSMFT NoiseMap
#ifndef ALICEO2_ITSMFT_NOISEMAP_H
#define ALICEO2_ITSMFT_NOISEMAP_H

#include "Rtypes.h" // for Double_t, ULong_t, etc
#include <iostream>
#include <climits>
#include <cassert>
#include <vector>
#include <map>
#include "Framework/Logger.h"
#include "gsl/span"

namespace o2
{

namespace itsmft
{

class CompClusterExt;

/// \class NoiseMap
/// \brief NoiseMap class for the ITS and MFT
///

class NoiseMap
{

 public:
  /// Constructor, initializing values for position, charge and readout frame
  NoiseMap(std::vector<std::map<int, int>>& noise) { mNoisyPixels.swap(noise); }

  /// Constructor
  NoiseMap() = default;
  /// Constructor
  NoiseMap(int nchips)
  {
    mNoisyPixels.assign(nchips, std::map<int, int>());
  }
  /// Destructor
  ~NoiseMap() = default;

  /// Get the noise level for this pixels
  float getNoiseLevel(int chip, int row, int col) const
  {
    assert(chip < (int)mNoisyPixels.size());
    const auto keyIt = mNoisyPixels[chip].find(getKey(row, col));
    if (keyIt != mNoisyPixels[chip].end()) {
      return keyIt->second;
    }
    return 0;
  }

  void increaseNoiseCount(int chip, int row, int col)
  {
    assert(chip < (int)mNoisyPixels.size());
    mNoisyPixels[chip][getKey(row, col)]++;
  }

  int dumpAboveThreshold(int t = 3) const
  {
    int n = 0;
    auto chipID = mNoisyPixels.size();
    while (chipID--) {
      const auto& map = mNoisyPixels[chipID];
      for (const auto& pair : map) {
        if (pair.second <= t) {
          continue;
        }
        n++;
        auto key = pair.first;
        auto row = key2Row(key);
        auto col = key2Col(key);
        std::cout << "chip, row, col, noise: " << chipID << ' ' << row << ' ' << col << ' ' << pair.second << '\n';
      }
    }
    return n;
  }
  int dumpAboveProbThreshold(float p = 1e-7) const
  {
    return dumpAboveThreshold(p * mNumOfStrobes);
  }

  void applyProbThreshold(float t, long int n)
  {
    // Remove from the maps all pixels with the firing probability below the threshold
    mProbThreshold = t;
    mNumOfStrobes = n;
    for (auto& map : mNoisyPixels) {
      for (auto it = map.begin(); it != map.end();) {
        float prob = float(it->second) / mNumOfStrobes;
        if (prob < mProbThreshold) {
          it = map.erase(it);
        } else {
          ++it;
        }
      }
    }
  }
  float getProbThreshold() const { return mProbThreshold; }
  long int getNumOfStrobes() const { return mNumOfStrobes; }

  bool isNoisy(int chip, int row, int col) const
  {
    assert(chip < (int)mNoisyPixels.size());
    return (mNoisyPixels[chip].find(getKey(row, col)) != mNoisyPixels[chip].end());
  }

  bool isNoisyOrFullyMasked(int chip, int row, int col) const
  {
    assert(chip < (int)mNoisyPixels.size());
    return isNoisy(chip, row, col) || isFullChipMasked(chip);
  }

  bool isNoisy(int chip) const
  {
    assert(chip < (int)mNoisyPixels.size());
    return !mNoisyPixels[chip].empty();
  }

  // Methods required by the calibration framework
  void print();
  void fill(const gsl::span<const CompClusterExt> data);
  void merge(const NoiseMap* prev) {}
  const std::map<int, int>* getChipMap(int chip) const { return chip < (int)mNoisyPixels.size() ? &mNoisyPixels[chip] : nullptr; }

  void maskFullChip(int chip, bool cleanNoisyPixels = false)
  {
    if (cleanNoisyPixels) {
      resetChip(chip);
    }
    increaseNoiseCount(chip, -1, -1);
  }

  bool isFullChipMasked(int chip) const
  {
    return isNoisy(chip, -1, -1);
  }

  void resetChip(int chip)
  {
    assert(chip < (int)mNoisyPixels.size());
    mNoisyPixels[chip].clear();
  }

 private:
  static constexpr int SHIFT = 10, MASK = (0x1 << SHIFT) - 1;
  int getKey(int row, int col) const { return (row << SHIFT) + col; }
  int key2Row(int key) const { return key >> SHIFT; }
  int key2Col(int key) const { return key & MASK; }
  std::vector<std::map<int, int>> mNoisyPixels; ///< Internal noise map representation
  long int mNumOfStrobes = 0;                   ///< Accumulated number of ALPIDE strobes
  float mProbThreshold = 0;                     ///< Probability threshold for noisy pixels

  ClassDefNV(NoiseMap, 2);
};
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_NOISEMAP_H */
