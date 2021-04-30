// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibratorDigits.cxx

#include "MFTCalibration/NoiseCalibratorDigits.h"

#include "FairLogger.h"
#include "TFile.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"

namespace o2
{
namespace mft
{
bool NoiseCalibratorDigits::processTimeFrame(gsl::span<const o2::itsmft::Digit> const& digits,
                                             gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  const o2::itsmft::ChipMappingMFT maping;
  auto chipMap = maping.getChipMappingData();

  static int nTF = 0;
  LOG(INFO) << "Processing TF# " << nTF++;

  for (const auto& rof : rofs) {
    auto digitsInFrame = rof.getROFData(digits);
    for (const auto& d : digitsInFrame) {
      auto id = d.getChipIndex();
      auto row = d.getRow();
      auto col = d.getColumn();

      int half = chipMap[id].half;
      int layer = chipMap[id].layer;
      int disk = chipMap[id].disk;
      int face = layer % 2;

        if (half == 0 && face == 0) {
	  if (mIsEmpty[0]){
        	  mPath[0] = mPath[0] + "-d" + std::to_string(disk);
		  mIsEmpty[0]=false;
	  }
	  mNoiseMapH0F0.increaseNoiseCount(id, row, col);
        } else if (half == 0 && face == 1) {
	  if (mIsEmpty[1]){
        	  mPath[1] = mPath[1] + "-d" + std::to_string(disk);
		  mIsEmpty[1]=false;
	  }
          mNoiseMapH0F1.increaseNoiseCount(id, row, col);
        } else if (half == 1 && face == 0) {
	  if (mIsEmpty[2]){
        	  mPath[2] = mPath[2] + "-d" + std::to_string(disk);
		  mIsEmpty[2]=false;
	  }
          mNoiseMapH1F0.increaseNoiseCount(id, row, col);
        } else if (half == 1 && face == 1) {
	  if (mIsEmpty[3]){
        	  mPath[3] = mPath[3] + "-d" + std::to_string(disk);
		  mIsEmpty[3]=false;
	  }
          mNoiseMapH1F1.increaseNoiseCount(id, row, col);
        }
    }
  }
  mNumberOfStrobes += rofs.size();
  return (mNumberOfStrobes * mProbabilityThreshold >= mThreshold) ? true : false;
}

void NoiseCalibratorDigits::finalize()
{
  LOG(INFO) << "Number of processed strobes is " << mNumberOfStrobes;
  mNoiseMapH0F0.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
  mNoiseMapH0F1.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
  mNoiseMapH1F0.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
  mNoiseMapH1F1.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);
}

} // namespace mft
} // namespace o2
