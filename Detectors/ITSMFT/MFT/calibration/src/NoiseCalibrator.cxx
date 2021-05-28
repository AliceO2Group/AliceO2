// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibrator.cxx

#include "MFTCalibration/NoiseCalibrator.h"

#include "FairLogger.h"
#include "TFile.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
namespace mft
{
bool NoiseCalibrator::processTimeFrame(calibration::TFType tf,
                                       gsl::span<const o2::itsmft::Digit> const& digits,
                                       gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  static int nTF = 0;
  LOG(INFO) << "Processing TF# " << nTF++;

  for (const auto& rof : rofs) {
    auto digitsInFrame = rof.getROFData(digits);
    for (const auto& d : digitsInFrame) {
      auto id = d.getChipIndex();
      auto row = d.getRow();
      auto col = d.getColumn();

      mNoiseMap.increaseNoiseCount(id, row, col);
    }
  }
  mNumberOfStrobes += rofs.size();
  return (mNumberOfStrobes * mProbabilityThreshold >= mThreshold) ? true : false;
}

bool NoiseCalibrator::processTimeFrame(calibration::TFType tf,
                                       gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                                       gsl::span<const unsigned char> const& patterns,
                                       gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  static int nTF = 0;
  LOG(INFO) << "Processing TF# " << nTF++;

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
        mNoiseMap.increaseNoiseCount(id, row, col);
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
            mNoiseMap.increaseNoiseCount(id, row + ir, col + ic);
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
  mNumberOfStrobes += rofs.size();
  return (mNumberOfStrobes * mProbabilityThreshold >= mThreshold) ? true : false;
}

void NoiseCalibrator::finalize()
{
  LOG(INFO) << "Number of processed strobes is " << mNumberOfStrobes;
  mNoiseMap.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
}

} // namespace mft
} // namespace o2
