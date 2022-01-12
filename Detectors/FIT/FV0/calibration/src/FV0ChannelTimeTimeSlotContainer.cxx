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

#include "FV0Calibration/FV0ChannelTimeTimeSlotContainer.h"
#include "FV0Base/Constants.h"
#include <numeric>
#include <algorithm>
#include "MathUtils/fit.h"

using namespace o2::fv0;

int FV0ChannelTimeTimeSlotContainer::sGausFitBins = 999; // NOT USED

FV0ChannelTimeTimeSlotContainer::FV0ChannelTimeTimeSlotContainer(std::size_t minEntries)
  : mMinEntries(minEntries)
{

  mHistogram = boost::histogram::make_histogram(boost::histogram::axis::integer<>(-HISTOGRAM_RANGE, HISTOGRAM_RANGE, "channel_times"),
                                                boost::histogram::axis::integer<>(0, Constants::nFv0Channels, "channel_ID"));
}

bool FV0ChannelTimeTimeSlotContainer::hasEnoughEntries() const
{
  return *std::min_element(mEntriesPerChannel.begin(), mEntriesPerChannel.end()) > mMinEntries;
}
void FV0ChannelTimeTimeSlotContainer::fill(const gsl::span<const FV0CalibrationInfoObject>& data)
{

  for (auto& entry : data) {

    const auto chID = entry.getChannelIndex();
    const auto chTime = entry.getTime();

    //i dont really know when should it be marked as invalid
    if (chID < Constants::nFv0Channels) {
      mHistogram(chTime, chID);
      ++mEntriesPerChannel[chID];
    }
    //else {
    //  LOG(fatal) << "Invalid channel data";
    //}
  }
}

void FV0ChannelTimeTimeSlotContainer::merge(FV0ChannelTimeTimeSlotContainer* prev)
{

  mHistogram += prev->mHistogram;
  for (unsigned int iCh = 0; iCh < Constants::nFv0Channels; ++iCh) {
    mEntriesPerChannel[iCh] += prev->mEntriesPerChannel[iCh];
  }
}

int16_t FV0ChannelTimeTimeSlotContainer::getMeanGaussianFitValue(std::size_t channelID) const
{

  static constexpr size_t MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR = 1;

  if (0 == mEntriesPerChannel[channelID]) {
    return 0;
  }
  LOG(debug) << " for channel " << int(channelID) << " entries " << mEntriesPerChannel[channelID];

  std::vector<double> channelHistogramData(NUMBER_OF_HISTOGRAM_BINS, 0);

  std::vector<double> outputGaussianFitValues;
  double binWidth = (HISTOGRAM_RANGE - (-HISTOGRAM_RANGE)) / NUMBER_OF_HISTOGRAM_BINS;
  double minGausFitRange = 0;
  double maxGausFitRange = 0;
  double MaxValOfHistogram = 0.0;

  for (int iBin = 0; iBin < NUMBER_OF_HISTOGRAM_BINS; ++iBin) {
    channelHistogramData[iBin] = mHistogram.at(iBin, channelID);
  }

  int maxElementIndex = std::max_element(channelHistogramData.begin(), channelHistogramData.end()) - channelHistogramData.begin();
  int maxElement = *std::max_element(channelHistogramData.begin(), channelHistogramData.end());

  // calculating the min & max range values to fit gaussian
  minGausFitRange = (-HISTOGRAM_RANGE + (maxElementIndex - sGausFitBins) * binWidth + binWidth / 2.0);
  maxGausFitRange = (-HISTOGRAM_RANGE + (maxElementIndex + sGausFitBins) * binWidth + binWidth / 2.0);

  double returnCode = math_utils::fitGaus<double>(NUMBER_OF_HISTOGRAM_BINS, channelHistogramData.data(),
                                                  minGausFitRange, maxGausFitRange, outputGaussianFitValues);

  MaxValOfHistogram = (-HISTOGRAM_RANGE + maxElementIndex * binWidth + binWidth / 2.0);

  if (returnCode < 0) {
    LOG(error) << "Gaussian fit error!";
    return static_cast<int16_t>(std::round(MaxValOfHistogram));
    //return 0;
  }

  return static_cast<int16_t>(std::round(outputGaussianFitValues[MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR]));
}

void FV0ChannelTimeTimeSlotContainer::print() const
{
  //QC will do that part
}
