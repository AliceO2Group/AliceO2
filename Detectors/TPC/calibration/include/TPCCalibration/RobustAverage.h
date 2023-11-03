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

/// \file RobustAverage.h
/// \brief class for performing robust averaging and outlier filtering
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_ROBUSTAVERAGE_H_
#define ALICEO2_ROBUSTAVERAGE_H_

#include <vector>

namespace o2::tpc
{

/// class to perform filtering of outliers and robust averaging of a set of values.
/// This class is more or less a dummy for now... TODO add more sophisticated methods
///
/// Usage using existing data:
/// 1. std::vector<float> values{1., 2., 2.3};
/// 2. o2::tpc::RobustAverage rob(std::move(values));
/// 3. float average = rob.getFilteredAverage(3);
///
/// Usage using copy of data:
/// 1. o2::tpc::RobustAverage rob(3);
/// 2. rob.addValue(1.);
///    rob.addValue(2.);
///    rob.addValue(2.3);
/// 3. float average = rob.getFilteredAverage(3);
///

class RobustAverage
{
 public:
  /// constructor
  /// \param maxValues maximum number of values which will be averaged. Copy of values will be done.
  /// \param withWeights also storing weights for weighted mean
  RobustAverage(const unsigned int maxValues, bool withWeights = true) : mUseWeights{withWeights}
  {
    mValues.reserve(maxValues);
    if (mUseWeights) {
      mWeights.reserve(maxValues);
    }
  }

  /// default constructor
  RobustAverage() = default;

  /// constructor
  /// \param values values which will be averaged and filtered. Move operator is used here!
  RobustAverage(std::vector<float>&& values) : mValues{std::move(values)} {};

  /// reserve memory for member
  /// \param maxValues maximum number of values which will be averaged. Copy of values will be done.
  void reserve(const unsigned int maxValues);

  /// clear the stored values
  void clear();

  /// \param value value which will be added to the list of stored values for averaging
  /// \param weight weight of the value (weight will only be stored if flag to weights is true)
  void addValue(const float value, const float weight = 1);

  /// returns the filtered average value
  /// \param sigma maximum accepted standard deviation: sigma*stdev
  ///\param interQuartileRange number of points in inner quartile to consider
  std::pair<float, float> getFilteredAverage(const float sigma = 3, const float interQuartileRange = 0.9);

  /// remove all the point which are abs(val - val_median)>maxAbsMedian
  std::tuple<float, float, float, unsigned int> filterPointsMedian(const float maxAbsMedian, const float sigma = 5);

  /// \return returns mean of stored values
  float getMean() const { return mValues.empty() ? 0 : getMean(mValues.begin(), mValues.end()); }

  /// returns truncated mean fir range min and max
  float getTrunctedMean(float low, float high);

  /// \return returns the median
  float getMedian();

  /// \return returns weighted mean of stored values
  float getWeightedMean() const { return mUseWeights ? getWeightedMean(mValues.begin(), mValues.end(), mWeights.begin(), mWeights.end()) : 0; }

  /// \return returns standard deviation of stored values
  float getStdDev() { return getStdDev(getMean(), mValues.begin(), mValues.end()); }

  /// \return returns stored values
  const auto& getValues() const { return mValues; }

  void setValues(const std::vector<float>& values) { mValues = values; }

  /// \return returns stored values
  const auto& getWeigths() const { return mWeights; }

  /// values which will be averaged and filtered
  void print() const;

  // sorting values and weights
  void sort();

  /// returns if weights are stored
  bool getUseWeights() const { return mUseWeights; }

  /// returns if weights are stored
  void setUseWeights(bool useweights) { mUseWeights = useweights; }

  /// \return returns the quantile value - linear interpolation or median unbiased used -
  /// \param quantile quantile to get
  /// \param type interpolation type: type=0 use linear interpolation, type=1 use unbiased median (in case of low statistics)
  float getQuantile(float quantile, int type);

 private:
  std::vector<float> mValues{};    ///< values which will be averaged and filtered
  std::vector<float> mWeights{};   ///< weights of each value
  std::vector<float> mTmpValues{}; ///< tmp vector used for calculation of std dev
  bool mUseWeights{};              ///< also storing weights

  float getMean(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) const;

  float getWeightedMean(std::vector<float>::const_iterator beginValues, std::vector<float>::const_iterator endValues, std::vector<float>::const_iterator beginWeight, std::vector<float>::const_iterator endWeight) const;

  /// \return returns standard deviation of stored values
  /// \param mean mean of stored values
  float getStdDev(const float mean, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end);

  /// performing outlier filtering of the stored values by defining range of included values in terms of standard deviation
  /// \param mean mean of the stored values
  /// \param stdev standard deviation of the values
  /// \param sigma maximum accepted standard deviation: sigma*stdev
  float getFilteredMean(const float mean, const float stdev, const float sigma) const;
};

} // namespace o2::tpc

#endif
