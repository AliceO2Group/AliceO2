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
#include <numeric>
#include <algorithm>
#include "MathUtils/fit.h"

using namespace o2::ft0;

int FT0ChannelTimeTimeSlotContainer::sGausFitBins = 999; // NOT USED

FT0ChannelTimeTimeSlotContainer::FT0ChannelTimeTimeSlotContainer(std::size_t minEntries)
  : mMinEntries(minEntries)
{

  mHistogram = boost::histogram::make_histogram(boost::histogram::axis::integer<>(-HISTOGRAM_RANGE, HISTOGRAM_RANGE, "channel_times"),
                                                boost::histogram::axis::integer<>(0, o2::ft0::Nchannels_FT0, "channel_ID"));
}

bool FT0ChannelTimeTimeSlotContainer::hasEnoughEntries() const
{
  return *std::min_element(mEntriesPerChannel.begin(), mEntriesPerChannel.end()) > mMinEntries;
}
void FT0ChannelTimeTimeSlotContainer::fill(const gsl::span<const FT0CalibrationInfoObject>& data)
{

  for (auto& entry : data) {

    const auto chID = entry.getChannelIndex();
    const auto chTime = entry.getTime();

    //i dont really know when should it be marked as invalid
    if (chID < o2::ft0::Geometry::Nchannels) {
      mHistogram(chTime, chID);
      ++mEntriesPerChannel[chID];
    } else {
      LOG(FATAL) << "Invalid channel data";
    }
  }
}

void FT0ChannelTimeTimeSlotContainer::merge(FT0ChannelTimeTimeSlotContainer* prev)
{

  mHistogram += prev->mHistogram;
  for (unsigned int iCh = 0; iCh < o2::ft0::Geometry::Nchannels; ++iCh) {
    mEntriesPerChannel[iCh] += prev->mEntriesPerChannel[iCh];
  }
}

int16_t FT0ChannelTimeTimeSlotContainer::getMeanGaussianFitValue(std::size_t channelID) const
{

  static constexpr size_t MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR = 1;

  if (0 == mEntriesPerChannel[channelID]) {
    return 0;
  }
  LOG(DEBUG) << " for channel " << int(channelID) << " entries " << mEntriesPerChannel[channelID];

  std::vector<double> channelHistogramData(NUMBER_OF_HISTOGRAM_BINS);

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
    LOG(ERROR) << "Gaussian fit error!";
    return static_cast<int16_t>(std::round(MaxValOfHistogram));
  }

  return static_cast<int16_t>(std::round(outputGaussianFitValues[MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR]));
}

void FT0ChannelTimeTimeSlotContainer::print() const
{
  //QC will do that part
}
