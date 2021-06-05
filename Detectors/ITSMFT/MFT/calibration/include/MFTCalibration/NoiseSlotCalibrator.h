// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseSlotCalibrator.h

#ifndef O2_MFT_NOISESLOTCALIBRATOR
#define O2_MFT_NOISESLOTCALIBRATOR

#include <string>

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "gsl/span"

namespace o2
{

namespace itsmft
{
class ROFRecord;
} // namespace itsmft

namespace mft
{

class NoiseSlotCalibrator : public o2::calibration::TimeSlotCalibration<o2::itsmft::CompClusterExt, o2::itsmft::NoiseMap>
{
  using Slot = calibration::TimeSlot<o2::itsmft::NoiseMap>;

 public:
  NoiseSlotCalibrator() { setUpdateAtTheEndOfRunOnly(); }
  NoiseSlotCalibrator(float prob)
  {
    mProbabilityThreshold = prob;
    setUpdateAtTheEndOfRunOnly();
    setSlotLength(std::numeric_limits<TFType>::max);
  }
  ~NoiseSlotCalibrator() final = default;

  void setThreshold(unsigned int t) { mThreshold = t; }

  bool processTimeFrame(calibration::TFType tf,
                        gsl::span<const o2::itsmft::Digit> const& digits,
                        gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  bool processTimeFrame(calibration::TFType tf,
                        gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                        gsl::span<const unsigned char> const& patterns,
                        gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  void finalize()
  {
    LOG(INFO) << "Number of processed strobes is " << mNumberOfStrobes;
    auto& slot = getSlots().back();
    slot.getContainer()->applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
  }

  const o2::itsmft::NoiseMap& getNoiseMap(long& start, long& end)
  {
    const auto& slot = getSlots().back();
    start = slot.getTFStart();
    end = slot.getTFEnd();
    return *(slot.getContainer());
  }

  // Functions overloaded from the calibration framework
  bool process(calibration::TFType tf, const gsl::span<const o2::itsmft::CompClusterExt> data) final;

  // Functions required by the calibration framework
  void initOutput() final {}
  Slot& emplaceNewSlot(bool, calibration::TFType, calibration::TFType) final;
  void finalizeSlot(Slot& slot) final;
  bool hasEnoughData(const Slot& slot) const final;

 private:
  float mProbabilityThreshold = 1e-6f;
  unsigned int mThreshold = 100;
  unsigned int mNumberOfStrobes = 0;
};

} // namespace mft
} // namespace o2

#endif /* O2_MFT_NOISESLOTCALIBRATOR */
