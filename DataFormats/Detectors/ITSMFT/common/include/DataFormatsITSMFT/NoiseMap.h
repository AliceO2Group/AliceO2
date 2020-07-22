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

  /// Destructor
  ~NoiseMap() = default;

  /// Get the noise level for this pixels
  int getNoiseLevel(int chip, int row, int col) const
  {
    if (chip > mNoisyPixels.size())
      return 0;
    auto key = row * 1024 + col;
    const auto keyIt = mNoisyPixels[chip].find(key);
    if (keyIt != mNoisyPixels[chip].end())
      return keyIt->second;
    return 0;
  }

 private:
  std::vector<std::map<int, int>> mNoisyPixels; ///< Internal noise map representation

  ClassDefNV(NoiseMap, 1);
};
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_NOISEMAP_H */
