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

/// @file   NoiseSlotCalibrator.cxx

#include "MFTCalibration/NoiseSlotCalibrator.h"

#include <fairlogger/Logger.h>
#include "TFile.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
using Slot = calibration::TimeSlot<o2::itsmft::NoiseMap>;

namespace mft
{
bool NoiseSlotCalibrator::processTimeFrame(calibration::TFType nTF,
                                           gsl::span<const o2::itsmft::Digit> const& digits,
                                           gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  LOG(detail) << "Processing TF# " << nTF;

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
  noiseMap.addStrobes(rofs.size());
  mNumberOfStrobes += rofs.size();
  return hasEnoughData(slotTF);
}

bool NoiseSlotCalibrator::processTimeFrame(calibration::TFType nTF,
                                           gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                                           gsl::span<const unsigned char> const& patterns,
                                           gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  LOG(detail) << "Processing TF# " << nTF;

  auto& slotTF = getSlotForTF(nTF);
  auto& noiseMap = *(slotTF.getContainer());

  auto pattIt = patterns.begin();
  for (const auto& rof : rofs) {
    auto clustersInFrame = rof.getROFData(clusters);
    for (const auto& c : clustersInFrame) {
      if (c.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID) {
        // For the noise calibration, we use "pass1" clusters...
        continue;
      }
      o2::itsmft::ClusterPattern patt(pattIt);

      auto id = c.getSensorID();
      auto row = c.getRow();
      auto col = c.getCol();
      auto colSpan = patt.getColumnSpan();
      auto rowSpan = patt.getRowSpan();

      // Fast 1-pixel calibration
      if ((rowSpan == 1) && (colSpan == 1)) {
        noiseMap.increaseNoiseCount(id, row, col);
        continue;
      }

      // All-pixel calibration
      auto nBits = rowSpan * colSpan;
      int ic = 0, ir = 0;
      for (unsigned int i = 2; i < patt.getUsedBytes() + 2; i++) {
        unsigned char tempChar = patt.getByte(i);
        int s = 128; // 0b10000000
        while (s > 0) {
          if ((tempChar & s) != 0) {
            noiseMap.increaseNoiseCount(id, row + ir, col + ic);
          }
          ic++;
          s >>= 1;
          if ((ir + 1) * ic == nBits) {
            break;
          }
          if (ic == colSpan) {
            ic = 0;
            ir++;
          }
        }
        if ((ir + 1) * ic == nBits) {
          break;
        }
      }
    }
  }
  noiseMap.addStrobes(rofs.size());
  mNumberOfStrobes += rofs.size();
  return hasEnoughData(slotTF);
}

// Functions overloaded from the calibration framework
bool NoiseSlotCalibrator::process(calibration::TFType tf, const gsl::span<const o2::itsmft::CompClusterExt> data)
{
  LOG(warning) << "Only 1-pix noise calibraton is possible !";
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

bool NoiseSlotCalibrator::hasEnoughData(const Slot& slot) const
{
  return slot.getContainer()->getNumberOfStrobes() > mMinROFs ? true : false;
}

void NoiseSlotCalibrator::finalizeSlot(Slot& slot)
{
  o2::itsmft::NoiseMap* map = slot.getContainer();
  LOG(info) << "Number of processed strobes is " << map->getNumberOfStrobes();
  map->applyProbThreshold(mProbabilityThreshold, map->getNumberOfStrobes(), mProbRelErr);
}

} // namespace mft
} // namespace o2
