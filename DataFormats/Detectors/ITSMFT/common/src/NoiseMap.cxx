// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file NoiseMap.cxx
/// \brief Implementation of the ITSMFT NoiseMap

#include "DataFormatsITSMFT/NoiseMap.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "Framework/Logger.h"

ClassImp(o2::itsmft::NoiseMap);

using namespace o2::itsmft;

void NoiseMap::print()
{
  int nc = 0, np = 0;
  for (const auto& map : mNoisyPixels) {
    if (!map.empty()) {
      nc++;
    }
    np += map.size();
  }
  LOG(INFO) << "Number of noisy chips: " << nc;
  LOG(INFO) << "Number of noisy pixels: " << np;
  LOG(INFO) << "Number of of strobes: " << mNumOfStrobes;
  LOG(INFO) << "Probability threshold: " << mProbThreshold;
}

void NoiseMap::fill(const gsl::span<const CompClusterExt> data)
{
  for (const auto& c : data) {
    if (c.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID) {
      // For the noise calibration, we use "pass1" clusters...
      continue;
    }

    auto id = c.getSensorID();
    auto row = c.getRow();
    auto col = c.getCol();

    // A simplified 1-pixel calibration
    increaseNoiseCount(id, row, col);
  }
}
