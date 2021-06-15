// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCCalibration/RobustAverage.h"
#include "Framework/Logger.h"

float o2::tpc::RobustAverage::getFilteredAverage(const float sigma)
{
  if (mValues.empty()) {
    return 0;
  }

  const float mean = getMean();
  const float stdev = getStdDev(mean);
  filterOutliers(mean, stdev, sigma);

  if (mValues.empty()) {
    return 0;
  }

  return getMean();
}

float o2::tpc::RobustAverage::getStdDev(const float mean) const
{
  std::vector<float> diff(mValues.size());
  std::transform(mValues.begin(), mValues.end(), diff.begin(), [mean](const float val) { return val - mean; });
  const float sqsum = std::inner_product(diff.begin(), diff.end(), diff.begin(), decltype(diff)::value_type(0));
  const float stdev = std::sqrt(sqsum / diff.size());
  return stdev;
}

void o2::tpc::RobustAverage::filterOutliers(const float mean, const float stdev, const float sigma)
{
  std::sort(mValues.begin(), mValues.end());
  const float sigmastddev = sigma * stdev;
  const float minVal = mean - sigmastddev;
  const float maxVal = mean + sigmastddev;
  const auto upper = std::upper_bound(mValues.begin(), mValues.end(), maxVal);
  mValues.erase(upper, mValues.end());
  const auto lower = std::lower_bound(mValues.begin(), mValues.end(), minVal);
  mValues.erase(mValues.begin(), lower);
}

void o2::tpc::RobustAverage::print() const
{
  LOGP(info, "PRINTING STORED VALUES");
  for (auto val : mValues) {
    LOGP(info, "{}", val);
  }
}
