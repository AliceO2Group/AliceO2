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

#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "FT0Calibration//GlobalOffsetsContainer.h"
#include "DataFormatsFT0/GlobalOffsetsInfoObject.h"
#include <numeric>
#include <algorithm>
#include "MathUtils/fit.h"
#include <TFitResult.h>
#include <gsl/span>

using namespace o2::ft0;
using o2::math_utils::fitGaus;

bool GlobalOffsetsContainer::hasEnoughEntries() const
{
  return mEntries < mMinEntries;
}
void GlobalOffsetsContainer::fill(const gsl::span<const GlobalOffsetsInfoObject>& data)
{
  // fill container
  for (auto& entry : data) {
    if (std::abs(entry.getT0AC()) < RANGE) {
      updateFirstCreation(entry.getTimeStamp());
      auto time = entry.getT0AC();
      time += RANGE;
      mHisto[(time)]++;
      mEntries++;
    }
  }
}
void GlobalOffsetsContainer::merge(GlobalOffsetsContainer* prev)
{
  // merge data of 2 slots
  for (int i = mHisto.size(); i--;) {
    mHisto[i] += prev->mHisto[i];
  }
  mEntries += prev->mEntries;
  mFirstCreation = std::min(mFirstCreation, prev->mFirstCreation);
}

int GlobalOffsetsContainer::getMeanGaussianFitValue() const
{
  if (!hasEnoughEntries()) {
    return 0;
  }
  if (mEntries == 0) {
    return 0;
  }
  float sum = 0;
  for (int ic = 0; ic < NBINS; ic++) {
    sum += float(ic - RANGE) * float(mHisto[ic]);
    //    LOG(info)<<" histo "<<ic<<" "<< float(mHisto[ic]);
  }
  std::array<double, 3> fitValues;
  double fitres = fitGaus(NBINS, mHisto.data(), -RANGE, RANGE, fitValues);
  if (fitres >= 0) {
    LOG(info) << "Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2] << " size " << mEntries << " mean " << sum / mEntries;
    return (static_cast<int>(std::round(fitValues[1])));
  } else {
    LOG(warning) << "Fit failed with result = " << fitres;
    return 0;
  }
}
GlobalOffsetsCalibrationObject GlobalOffsetsContainer::generateCalibrationObject(long, long, const std::string&) const
{
  GlobalOffsetsCalibrationObject calibrationObject;
  calibrationObject.mCollisionTimeOffsets = getMeanGaussianFitValue();
  LOG(info) << "GlobalOffsetsCalibrationObjectAlgorithm generate CalibrationObject for T0"
            << " = " << calibrationObject.mCollisionTimeOffsets;
  return calibrationObject;
}

void GlobalOffsetsContainer::print() const
{
  LOG(info) << "Container keep data for LHC phase calibration:";
  LOG(info) << "Gaussian mean time AC side " << getMeanGaussianFitValue() << " based on" << mEntries << " entries";
}
