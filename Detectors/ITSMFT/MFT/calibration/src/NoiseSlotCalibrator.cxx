// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseSlotCalibrator.cxx

#include "MFTCalibration/NoiseSlotCalibrator.h"

#include "FairLogger.h"
#include "TFile.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
using Slot = calibration::TimeSlot<o2::itsmft::NoiseMap>;

namespace mft
{
bool NoiseSlotCalibrator::processTimeFrame(gsl::span<const o2::itsmft::Digit> const& digits,
                                           gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  calibration::TFType nTF = rofs[0].getBCData().orbit / mHBFperTF;
  LOG(INFO) << "Processing TF# " << nTF;

  auto& slotTF = getSlotForTF(nTF);
  auto& noiseMap = *(slotTF.getContainer());

  for (const auto& rof : rofs) {
    auto digitsInFrame = rof.getROFData(digits);
    for (const auto& d : digitsInFrame) {
      auto id = d.getChipIndex();
      auto row = d.getRow();
      auto col = d.getColumn();

      noiseMap.increaseNoiseCount(id, row, col);
    }
  }

  mNumberOfStrobes += rofs.size();
  return hasEnoughData(slotTF);
}

// Functions overloaded from the calibration framework
bool NoiseSlotCalibrator::process(calibration::TFType tf, const gsl::span<const o2::itsmft::CompClusterExt> data)
{
  LOG(WARNING) << "Only 1-pix noise calibraton is possible !";
  return calibration::TimeSlotCalibration<o2::itsmft::CompClusterExt, o2::itsmft::NoiseMap>::process(tf, data);
}

// Functions required by the calibration framework

Slot& NoiseSlotCalibrator::emplaceNewSlot(bool front, calibration::TFType tstart, calibration::TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<o2::itsmft::NoiseMap>(936));
  return slot;
}

bool NoiseSlotCalibrator::hasEnoughData(const Slot&) const
{
  return (mNumberOfStrobes * mProbabilityThreshold >= mThreshold) ? true : false;
}

void NoiseSlotCalibrator::finalizeSlot(Slot& slot)
{
  LOG(INFO) << "Number of processed strobes is " << mNumberOfStrobes;
  o2::itsmft::NoiseMap* map = slot.getContainer();
  map->applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
}

} // namespace mft
} // namespace o2
