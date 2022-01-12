
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
#include "DetectorsRaw/HBFUtils.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Calibration/LHCClockDataHisto.h"
#include <iostream>

namespace o2::ft0
{

using o2::math_utils::fitGaus;

//_____________________________________________
void LHCClockDataHisto::fill(const gsl::span<const FT0CalibrationInfoObject>& data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto ch = data[i].getChannelIndex();
    auto time = data[i].getTime();
    /*auto tot = data[i].getTot();
    //  auto corr = calibApi->getTimeCalibration(ch, tot); // we take into account LHCphase, offsets and time slewing
    //dt -= corr;

    // printf("ch=%d - tot=%f - corr=%f -> dtcorr = %f (range=%f, bin=%d)\n",ch,tot,corr,dt,range,int((dt+range)*v2Bin));
    */
    if (std::abs(time) < RANGE) {
      time += RANGE;
      mHisto[(time)]++;
      mEntries++;
    }
  }
}

//_____________________________________________
void LHCClockDataHisto::merge(const LHCClockDataHisto* prev)
{
  // merge data of 2 slots
  for (int i = mHisto.size(); i--;) {
    mHisto[i] += prev->mHisto[i];
  }
  mEntries += prev->mEntries;
}

void LHCClockDataHisto::print() const
{
  LOG(info) << mEntries << " entries";
}

bool LHCClockDataHisto::hasEnoughEntries() const
{
  return mEntries > mMinEntries;
}

int LHCClockDataHisto::getGaus() const
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
  }
  std::cout << std::endl;
  std::array<double, 3> fitValues;
  double fitres = fitGaus(NBINS, mHisto.data(), -RANGE, RANGE, fitValues);
  if (fitres >= 0) {
    LOG(info) << "Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2] << " size " << mEntries << " mean " << sum / mEntries;
    return (static_cast<int>(std::round(fitValues[1])));
  } else {
    LOG(error) << "Fit failed with result = " << fitres;
    return 0;
  }
}

} // namespace o2::ft0
