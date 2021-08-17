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

#include "DataFormatsFT0/GlobalOffsetsContainer.h"
#include <numeric>
#include <algorithm>
#include "MathUtils/fit.h"
#include <gsl/span>

using namespace o2::ft0;

GlobalOffsetsContainer::GlobalOffsetsContainer(std::size_t minEntries)
  : mMinEntries(minEntries)
{

  mHistogram = boost::histogram::make_histogram(boost::histogram::axis::integer<>(-HISTOGRAM_RANGE, HISTOGRAM_RANGE, "collision_times"),
                                                boost::histogram::axis::integer<>(0, 3, "side"));
}

bool GlobalOffsetsContainer::hasEnoughEntries() const
{
  return *std::min_element(mEntriesCollTime.begin(), mEntriesCollTime.end()) > mMinEntries;
}
void GlobalOffsetsContainer::fill(const gsl::span<const o2::ft0::RecoCalibInfoObject>& data)
{
  LOG(INFO)<<"@@@GlobalOffsetsContainer::fill data size  "<<data.size();
  for (auto& entry : data) {
    if (std::abs(entry.getT0A()) < 2000 && std::abs(entry.getT0C()) < 2000 && std::abs(entry.getT0AC()) < 2000) {
      mHistogram(0, entry.getT0A());
      mHistogram(1, entry.getT0C());
      mHistogram(2, entry.getT0AC());
      LOG(INFO) << "@@@@ GlobalOffsetsContainer::fill " << entry.getT0A() << " " << entry.getT0C() << " " << entry.getT0AC();
    } else {
      LOG(FATAL) << "Invalid channel data";
    }
  }
}

void GlobalOffsetsContainer::merge(GlobalOffsetsContainer* prev)
{
  mHistogram += prev->mHistogram;
}

int16_t GlobalOffsetsContainer::getMeanGaussianFitValue(std::size_t channelID) const
{

  static constexpr size_t MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR = 1;

  if (0 == mHistogram.size()) {
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
void GlobalOffsetsContainer::print() const
{
  //QC will do that part
}
