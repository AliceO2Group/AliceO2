// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  for (unsigned int iCh = 0; iCh < o2::ft0::Nchannels_FT0; ++iCh) {
    mEntriesPerChannel[iCh] += prev->mEntriesPerChannel[iCh];
  }
}

int16_t FT0ChannelTimeTimeSlotContainer::getMeanGaussianFitValue(std::size_t channelID) const
{

  static constexpr size_t MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR = 1;

  if (0 == mEntriesPerChannel[channelID]) {
    return 0;
  }

  std::vector<double> channelHistogramData(NUMBER_OF_HISTOGRAM_BINS);
  std::vector<double> outputGaussianFitValues;
  for (int iBin = 0; iBin < NUMBER_OF_HISTOGRAM_BINS; ++iBin) {
    channelHistogramData[iBin] = mHistogram.at(iBin, channelID);
  }

  double returnCode = math_utils::fitGaus<double>(NUMBER_OF_HISTOGRAM_BINS, channelHistogramData.data(),
                                                  -HISTOGRAM_RANGE, HISTOGRAM_RANGE, outputGaussianFitValues);
  if (returnCode < 0) {
    LOG(ERROR) << "Gaussian fit error!";
    return 0;
  }

  return static_cast<int16_t>(std::round(outputGaussianFitValues[MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR]));
}
void FT0ChannelTimeTimeSlotContainer::print() const
{
  //QC will do that part
}
