// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <numeric>

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

  /// constructor
  /// \param values values which will be averaged and filtered. Move operator is used here!
  RobustAverage(std::vector<float>&& values) : mValues{std::move(values)} {};

  /// clear the stored values
  void clear() { mValues.clear(); }

  /// \param value value which will be added to the list of stored values for averaging
  void addValue(const float value) { mValues.emplace_back(value); }

  /// returns the filtered average value
  /// \param sigma maximum accepted standard deviation: sigma*stdev
  float getFilteredAverage(const float sigma = 3);

  /// values which will be averaged and filtered
  void print() const;

 private:
  std::vector<float> mValues{}; ///< values which will be averaged and filtered

  /// \return returns mean of stored values
  float getMean() const { return std::accumulate(mValues.begin(), mValues.end(), decltype(mValues)::value_type(0)) / mValues.size(); }

  /// performing outlier filtering of the stored values
  float getStdDev(const float mean) const;

  /// performing outlier filtering of the stored values by defining range of included values in terms of standard deviation
  /// \param mean mean of the stored values
  /// \param stdev standard deviation of the values
  /// \param sigma maximum accepted standard deviation: sigma*stdev
  void filterOutliers(const float mean, const float stdev, const float sigma);
};

} // namespace o2::tpc

#endif
