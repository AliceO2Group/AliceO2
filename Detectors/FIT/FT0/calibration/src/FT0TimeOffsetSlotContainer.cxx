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

#include "FT0Calibration/FT0TimeOffsetSlotContainer.h"
#include "FT0Base/Geometry.h"
#include "FT0Base/FT0DigParam.h"
#include <numeric>
#include <algorithm>
#include <iterator>
#include <vector>
#include "MathUtils/fit.h"
#include "TH1.h"
#include "TFitResult.h"

using namespace o2::ft0;

FT0TimeOffsetSlotContainer::FT0TimeOffsetSlotContainer(std::size_t minEntries)
  : mMinEntries(minEntries) {}

FT0TimeOffsetSlotContainer::FT0TimeOffsetSlotContainer(FT0TimeOffsetSlotContainer const& other)
  : mMinEntries(other.mMinEntries), mHistogram(other.mHistogram) {}

FT0TimeOffsetSlotContainer& FT0TimeOffsetSlotContainer::operator=(FT0TimeOffsetSlotContainer const& other)
{
  mMinEntries = other.mMinEntries;
  mHistogram = other.mHistogram;
  return *this;
}

bool FT0TimeOffsetSlotContainer::hasEnoughEntries() const
{
  return *std::min_element(mEntriesPerChannel.begin(), mEntriesPerChannel.end()) > mMinEntries;
}
void FT0TimeOffsetSlotContainer::fill(const gsl::span<const float>& data)
{
  // Per TF procedure
  if (mIsFirstTF) {
    // To make histogram parameters dynamic, depending on TimeSpectraProcessor output
    mHistogram.adoptExternal(data);
    mIsFirstTF = false;
  } else {
    o2::dataformats::FlatHisto2D<float> hist;
    hist.adoptExternal(data);
    mHistogram.add(hist);
  }
}

void FT0TimeOffsetSlotContainer::merge(FT0TimeOffsetSlotContainer* prev)
{
  mHistogram.add(prev->mHistogram);
}

int16_t FT0TimeOffsetSlotContainer::getMeanGaussianFitValue(std::size_t channelID) const
{
  int meanGaus{0};
  int sigmaGaus{0};
  auto hist = mHistogram.createSliceYTH1F(channelID + 1);
  TFitResultPtr resultFit = hist->Fit("gaus", "0SQ", "", -200, 200);
  if ((Int_t)resultFit == 0) {
    meanGaus = int(resultFit->Parameters()[1]);
    sigmaGaus = int(resultFit->Parameters()[2]);
  }
  const auto meanHist = hist->GetMean();
  const auto rmsHist = hist->GetRMS();
  if (resultFit != 0 || std::abs(meanGaus - meanHist) > 20 || rmsHist < 1 || sigmaGaus > 30) { // to be used fot test with laser
    LOG(debug) << "Bad gaus fit: meanGaus " << meanGaus << " sigmaGaus " << sigmaGaus << " meanHist " << meanHist << " rmsHist " << rmsHist << "resultFit " << ((int)resultFit);
    meanGaus = meanHist;
  }
  return static_cast<int16_t>(meanGaus);
}

FT0ChannelTimeCalibrationObject FT0TimeOffsetSlotContainer::generateCalibrationObject() const
{
  FT0ChannelTimeCalibrationObject calibrationObject;
  for (unsigned int iCh = 0; iCh < sNCHANNELS; ++iCh) {
    calibrationObject.mTimeOffsets[iCh] = getMeanGaussianFitValue(iCh);
  }
  return calibrationObject;
}

void FT0TimeOffsetSlotContainer::print() const
{
  // QC will do that part
}
