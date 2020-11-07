// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <vector>
#include <map>

namespace o2
{

namespace itsmft
{
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
    if (chip > mNoisyPixels.size()) {
      return 0;
    }
    auto key = row * 1024 + col;
    const auto keyIt = mNoisyPixels[chip].find(key);
    if (keyIt != mNoisyPixels[chip].end()) {
      return keyIt->second;
    }
    return 0;
  }

  void increaseNoiseCount(int chip, int row, int col)
  {
    if (chip > mNoisyPixels.size()) {
      return;
    }
    auto key = row * 1024 + col;
    mNoisyPixels[chip][key]++;
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
        auto row = key / 1024;
        auto col = key % 1024;
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
    if (chip > mNoisyPixels.size()) {
      return false;
    }
    auto key = row * 1024 + col;
    const auto keyIt = mNoisyPixels[chip].find(key);
    if (keyIt != mNoisyPixels[chip].end()) {
      return true;
    }
    return false;
  }

 private:
  std::vector<std::map<int, int>> mNoisyPixels; ///< Internal noise map representation
  long int mNumOfStrobes = 0;                   ///< Accumulated number of ALPIDE strobes
  float mProbThreshold = 0;                     ///< Probability threshold for noisy pixels

  ClassDefNV(NoiseMap, 2);
};
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_NOISEMAP_H */
