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

#include "FT0Calibration/FT0ChannelTimeTimeSlotContainer.h"
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

int FT0ChannelTimeTimeSlotContainer::sGausFitBins = 999; // NOT USED

FT0ChannelTimeTimeSlotContainer::FT0ChannelTimeTimeSlotContainer(std::size_t minEntries)
  : mMinEntries(minEntries)
{
  for (int ich = 0; ich < NCHANNELS; ++ich) {
    mHistogram[ich].reset(new TH1F(Form("hTime%i", ich), "time", NUMBER_OF_HISTOGRAM_BINS, -HISTOGRAM_RANGE, HISTOGRAM_RANGE));
  }
}

FT0ChannelTimeTimeSlotContainer::FT0ChannelTimeTimeSlotContainer(FT0ChannelTimeTimeSlotContainer const& other)
  : mMinEntries(other.mMinEntries)
{
  for (int ich = 0; ich < NCHANNELS; ++ich) {
    mHistogram[ich].reset(new TH1F(*other.mHistogram[ich]));
  }
}

FT0ChannelTimeTimeSlotContainer& FT0ChannelTimeTimeSlotContainer::operator=(FT0ChannelTimeTimeSlotContainer const& other)
{
  mMinEntries = other.mMinEntries;
  for (int ich = 0; ich < NCHANNELS; ++ich) {
    mHistogram[ich].reset(new TH1F(*other.mHistogram[ich]));
  }
  return *this;
}

bool FT0ChannelTimeTimeSlotContainer::hasEnoughEntries() const
{
  return *std::min_element(mEntriesPerChannel.begin(), mEntriesPerChannel.end()) > mMinEntries;
}
void FT0ChannelTimeTimeSlotContainer::fill(const gsl::span<const FT0CalibrationInfoObject>& data)
{
  for (auto& entry : data) {
    updateFirstCreation(entry.getTimeStamp());
    const auto chID = entry.getChannelIndex();
    const auto chTime = entry.getTime();
    if (chID < NCHANNELS && std::abs(chTime) < o2::ft0::FT0DigParam::mTime_trg_gate && entry.getAmp() > o2::ft0::FT0DigParam::mAmpThresholdForReco) {
      mHistogram[chID]->Fill(chTime);
      ++mEntriesPerChannel[chID];
      LOG(debug) << "@@@@entries " << mEntriesPerChannel[chID] << " chID " << int(chID) << " time " << chTime << " tiestamp " << uint64_t(entry.getTimeStamp());
    }
  }
}

void FT0ChannelTimeTimeSlotContainer::merge(FT0ChannelTimeTimeSlotContainer* prev)
{
  for (unsigned int iCh = 0; iCh < NCHANNELS; ++iCh) {
    mHistogram[iCh]->Add(prev->mHistogram[iCh].get(), 1);
    mEntriesPerChannel[iCh] += prev->mEntriesPerChannel[iCh];
    LOG(debug) << " entries " << mEntriesPerChannel[iCh] << " " << prev->mEntriesPerChannel[iCh];
  }
  mFirstCreation = std::min(mFirstCreation, prev->mFirstCreation);
}

int16_t FT0ChannelTimeTimeSlotContainer::getMeanGaussianFitValue(std::size_t channelID) const
{

  if (0 == mEntriesPerChannel[channelID]) {
    return 0;
  }
  LOG(debug) << " for channel " << int(channelID) << " entries " << mEntriesPerChannel[channelID] << " hist entries " << mHistogram[channelID]->GetEntries() << " mean " << mHistogram[channelID]->GetMean() << " RMS " << mHistogram[channelID]->GetRMS();

  int outputGaussianFitValues = -99999;
  int sigma;
  TFitResultPtr r = mHistogram[channelID]->Fit("gaus", "0SQ", "", -sGausFitBins, sGausFitBins);
  if ((Int_t)r == 0) {
    outputGaussianFitValues = int(r->Parameters()[1]);
    sigma = int(r->Parameters()[2]);
  };
  //  if (r != 0 || std::abs(outputGaussianFitValues - mHistogram[channelID]->GetMean()) > 20 || mHistogram[channelID]->GetRMS() < 3 || sigma > 30) {
  if (r != 0 || std::abs(outputGaussianFitValues - mHistogram[channelID]->GetMean()) > 20 || mHistogram[channelID]->GetRMS() < 1 || sigma > 30) { // to be used fot test with laser
    LOG(info) << "!!! Bad gauss fit " << outputGaussianFitValues << " sigma " << sigma << " mean " << mHistogram[channelID]->GetMean() << " RMS " << mHistogram[channelID]->GetRMS();
    outputGaussianFitValues = mHistogram[channelID]->GetMean();
  }

  return static_cast<int16_t>(outputGaussianFitValues);
}

void FT0ChannelTimeTimeSlotContainer::print() const
{
  //QC will do that part
}
