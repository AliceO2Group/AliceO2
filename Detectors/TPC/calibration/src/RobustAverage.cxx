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

void o2::tpc::RobustAverage::reserve(const unsigned int maxValues)
{
  mValues.reserve(maxValues);
  mWeights.reserve(maxValues);
}

std::pair<float, float> o2::tpc::RobustAverage::getFilteredAverage(const float sigma, const float interQuartileRange)
{
  if (mValues.empty()) {
    return std::pair<float, float>(0, 0);
  }

  /*
    1. Sort the values
    2. Use only the values in the Interquartile Range
    3. Get the median
    4. Calculate the std dev of the selected values with the median as reference
    5. Get mean of the selected points which are in the range of the std dev
  */

  // 1.  Sort the values
  sort();

  // 2. Use only the values in the Interquartile Range (inner n%)
  const auto upper = mValues.begin() + mValues.size() * interQuartileRange;
  const auto lower = mValues.begin() + mValues.size() * (1 - interQuartileRange);

  // 3. Get the median
  const float median = mValues[mValues.size() / 2];

  if (upper == lower) {
    return std::pair<float, float>(median, 0);
  }

  // 4. Calculate the std dev of the selected values with the median
  const float stdev = getStdDev(median, lower, upper);

  // 5. Get mean of the selected points which are in the range of the std dev
  return std::pair<float, float>(getFilteredMean(median, stdev, sigma), stdev);
}

std::tuple<float, float, float, unsigned int> o2::tpc::RobustAverage::filterPointsMedian(const float maxAbsMedian, const float sigma)
{
  if (mValues.empty()) {
    return {0, 0, 0, 0};
  }

  // 1.  Sort the values
  sort();

  // 2. median
  const float median = mValues[mValues.size() / 2];

  // 3. select points larger and smaller than specified max value
  const auto upperV0 = std::upper_bound(mValues.begin(), mValues.end(), median + maxAbsMedian);
  const auto lowerV0 = std::lower_bound(mValues.begin(), mValues.end(), median - maxAbsMedian);

  if (upperV0 == lowerV0) {
    return {0, 0, 0, 0};
  }

  // 4. get RMS of selected values
  const float stdev = getStdDev(median, lowerV0, upperV0);

  // 5. define RMS cut
  const float sigmastddev = sigma * stdev;
  const float minVal = median - sigmastddev;
  const float maxVal = median + sigmastddev;
  const auto upper = std::upper_bound(mValues.begin(), mValues.end(), maxVal);
  const auto lower = std::lower_bound(mValues.begin(), mValues.end(), minVal);

  if (upper == lower) {
    return {0, 0, 0, 0};
  }

  const int indexUp = upper - mValues.begin();
  const int indexLow = lower - mValues.begin();

  // 7. get filtered median
  const float medianFilterd = mValues[(indexUp - indexLow) / 2 + indexLow];

  // 8. weighted mean
  const auto upperW = mWeights.begin() + indexUp;
  const auto lowerW = mWeights.begin() + indexLow;
  const float wMean = getWeightedMean(lower, upper, lowerW, upperW);

  // 9. number of points passing the cuts
  const int nEntries = indexUp - indexLow;

  return {medianFilterd, wMean, stdev, nEntries};
}

void o2::tpc::RobustAverage::sort()
{
  const size_t nVals = mValues.size();
  if (mValues.size() != mWeights.size()) {
    LOGP(warning, "values and errors haave different size");
    return;
  }
  std::vector<std::size_t> tmpIdx(nVals);
  std::iota(tmpIdx.begin(), tmpIdx.end(), 0);
  std::sort(tmpIdx.begin(), tmpIdx.end(), [&](std::size_t i, std::size_t j) { return (mValues[i] < mValues[j]); });

  std::vector<float> mValues_tmp;
  std::vector<float> mWeights_tmp;
  mValues_tmp.reserve(nVals);
  mWeights_tmp.reserve(nVals);
  for (int i = 0; i < nVals; ++i) {
    const int idx = tmpIdx[i];
    mValues_tmp.emplace_back(mValues[idx]);
    mWeights_tmp.emplace_back(mWeights[idx]);
  }
  mValues.swap(mValues_tmp);
  mWeights.swap(mWeights_tmp);
}

float o2::tpc::RobustAverage::getStdDev(const float mean, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end)
{
  const auto size = std::distance(begin, end);
  if (size == 0) {
    return 0;
  }
  mTmpValues.resize(size);
  std::transform(begin, end, mTmpValues.begin(), [mean](const float val) { return val - mean; });
  const float sqsum = std::inner_product(mTmpValues.begin(), mTmpValues.end(), mTmpValues.begin(), decltype(mTmpValues)::value_type(0));
  const float stdev = std::sqrt(sqsum / size);
  return stdev;
}

float o2::tpc::RobustAverage::getFilteredMean(const float mean, const float stdev, const float sigma) const
{
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
