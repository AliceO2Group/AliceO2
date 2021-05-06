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
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
namespace mft
{
bool NoiseCalibrator::processTimeFrame(gsl::span<const o2::itsmft::Digit> const& digits,
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

void NoiseCalibrator::finalize()
{
  LOG(INFO) << "Number of processed strobes is " << mNumberOfStrobes;
  mNoiseMap.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
}

} // namespace mft
} // namespace o2
