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
  RobustAverage(const unsigned int maxValues) { mValues.reserve(maxValues); }

  /// default constructor
  RobustAverage() = default;

  /// constructor
  /// \param values values which will be averaged and filtered. Move operator is used here!
  RobustAverage(std::vector<float>&& values) : mValues{std::move(values)} {};

  /// reserve memory for member
  /// \param maxValues maximum number of values which will be averaged. Copy of values will be done.
  void reserve(const unsigned int maxValues) { mValues.reserve(maxValues); }

  /// clear the stored values
  void clear() { mValues.clear(); }

  /// \param value value which will be added to the list of stored values for averaging
  void addValue(const float value) { mValues.emplace_back(value); }

  /// returns the filtered average value
  /// \param sigma maximum accepted standard deviation: sigma*stdev
  float getFilteredAverage(const float sigma = 3);

  /// \return returns mean of stored values
  float getMean() const { return getMean(mValues.begin(), mValues.end()); }

  /// values which will be averaged and filtered
  void print() const;

 private:
  std::vector<float> mValues{};    ///< values which will be averaged and filtered
  std::vector<float> mTmpValues{}; ///< tmp vector used for calculation of std dev

  float getMean(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) const;

  /// performing outlier filtering of the stored values
  float getStdDev(const float mean);

  /// performing outlier filtering of the stored values by defining range of included values in terms of standard deviation
  /// \param mean mean of the stored values
  /// \param stdev standard deviation of the values
  /// \param sigma maximum accepted standard deviation: sigma*stdev
  float getFilteredMean(const float mean, const float stdev, const float sigma);
};

} // namespace o2::tpc

#endif
