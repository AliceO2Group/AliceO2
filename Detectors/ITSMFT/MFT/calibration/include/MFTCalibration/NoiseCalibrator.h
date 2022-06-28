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

#ifndef O2_MFT_NOISECALIBRATOR
#define O2_MFT_NOISECALIBRATOR

#include <string>

#include "DataFormatsITSMFT/NoiseMap.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "gsl/span"

namespace o2
{

namespace itsmft
{
class Digit;
class ROFRecord;
} // namespace itsmft

namespace mft
{

class NoiseCalibrator
{
 public:
  NoiseCalibrator() = default;
  NoiseCalibrator(float prob, float relErr = 0.2) : mProbabilityThreshold(prob), mProbRelErr(relErr)
  {
    mMinROFs = 1.1 * o2::itsmft::NoiseMap::getMinROFs(prob, relErr);
    LOGP(info, "Expect at least {} ROFs needed to apply threshold {} with relative error {}", mMinROFs, mProbabilityThreshold, mProbRelErr);
  }
  ~NoiseCalibrator() = default;

  bool processTimeFrame(calibration::TFType tf,
                        gsl::span<const o2::itsmft::Digit> const& digits,
                        gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  bool processTimeFrame(calibration::TFType tf,
                        gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                        gsl::span<const unsigned char> const& patterns,
                        gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  void finalize();
  void setMinROFs(long n) { mMinROFs = n; }
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

  const o2::itsmft::NoiseMap& getNoiseMap() const { return mNoiseMap; }

 private:
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  o2::itsmft::NoiseMap mNoiseMap{936};
  float mProbabilityThreshold = 1e-6f;
  float mProbRelErr = 0.2; // relative error on channel noise to apply the threshold
  long mMinROFs = 0;
  unsigned int mNumberOfStrobes = 0;
};

} // namespace mft
} // namespace o2

#endif /* O2_MFT_NOISECALIBRATOR */
