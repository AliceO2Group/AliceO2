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

#include "TPCCalibration/RobustAverage.h"
#include "Framework/Logger.h"
#include <numeric>

float o2::tpc::RobustAverage::getFilteredAverage(const float sigma)
{
  if (mValues.empty()) {
    return 0;
  }

  const float mean = getMean();
  const float stdev = getStdDev(mean);
  return getFilteredMean(mean, stdev, sigma);
}

float o2::tpc::RobustAverage::getStdDev(const float mean)
{
  const auto size = mValues.size();
  mTmpValues.resize(size);
  std::transform(mValues.begin(), mValues.end(), mTmpValues.begin(), [mean](const float val) { return val - mean; });
  const float sqsum = std::inner_product(mTmpValues.begin(), mTmpValues.end(), mTmpValues.begin(), decltype(mTmpValues)::value_type(0));
  const float stdev = std::sqrt(sqsum / size);
  return stdev;
}

float o2::tpc::RobustAverage::getFilteredMean(const float mean, const float stdev, const float sigma)
{
  std::sort(mValues.begin(), mValues.end());
  const float sigmastddev = sigma * stdev;
  const float minVal = mean - sigmastddev;
  const float maxVal = mean + sigmastddev;
  const auto upper = std::upper_bound(mValues.begin(), mValues.end(), maxVal);
  const auto lower = std::lower_bound(mValues.begin(), mValues.end(), minVal);
  return getMean(lower, upper);
}

void o2::tpc::RobustAverage::print() const
{
  LOGP(info, "PRINTING STORED VALUES");
  for (auto val : mValues) {
    LOGP(info, "{}", val);
  }
}

float o2::tpc::RobustAverage::getMean(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) const
{
  return std::accumulate(begin, end, decltype(mValues)::value_type(0)) / std::distance(begin, end);
}

float o2::tpc::RobustAverage::getMedian()
{
  if (mValues.empty()) {
    return 0;
  }
  size_t n = mValues.size() / 2;
  std::nth_element(mValues.begin(), mValues.begin() + n, mValues.end());
  return mValues[n];
}

float o2::tpc::RobustAverage::getWeightedMean(std::vector<float>::const_iterator beginValues, std::vector<float>::const_iterator endValues, std::vector<float>::const_iterator beginWeight, std::vector<float>::const_iterator endWeight) const
{
  return std::inner_product(beginValues, endValues, beginWeight, decltype(mValues)::value_type(0)) / std::accumulate(beginWeight, endWeight, decltype(mWeights)::value_type(0));
}

void o2::tpc::RobustAverage::clear()
{
  mValues.clear();
  mWeights.clear();
}

void o2::tpc::RobustAverage::addValue(const float value, const float weight)
{
  mValues.emplace_back(value);
  mWeights.emplace_back(weight);
}

void o2::tpc::RobustAverage::addValue(const float value)
{
  mValues.emplace_back(value);
}
