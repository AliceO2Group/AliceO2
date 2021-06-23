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

/// @file   NoiseCalibrator.h

#ifndef O2_ITS_NOISECALIBRATOR
#define O2_ITS_NOISECALIBRATOR

#include <string>

#include "DataFormatsITSMFT/NoiseMap.h"
#include "gsl/span"

namespace o2
{

namespace itsmft
{
class CompClusterExt;
class ROFRecord;
} // namespace itsmft

namespace its
{

class NoiseCalibrator
{
 public:
  NoiseCalibrator() = default;
  NoiseCalibrator(bool one, float prob)
  {
    m1pix = one;
    mProbabilityThreshold = prob;
  }
  ~NoiseCalibrator() = default;

  void setThreshold(unsigned int t) { mThreshold = t; }

  bool processTimeFrame(gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                        gsl::span<const unsigned char> const& patterns,
                        gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  void finalize();

  const o2::itsmft::NoiseMap& getNoiseMap() const { return mNoiseMap; }

 private:
  o2::itsmft::NoiseMap mNoiseMap{24120};
  float mProbabilityThreshold = 3e-6f;
  unsigned int mThreshold = 100;
  unsigned int mNumberOfStrobes = 0;
  bool m1pix = true;
};

} // namespace its
} // namespace o2

#endif /* O2_ITS_NOISECALIBRATOR */
