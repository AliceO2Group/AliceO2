// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file StatAccumulator.h
/// \brief
/// \author ruben.shahoyan@cern.ch michael.lettrich@cern.ch

#ifndef MATHUTILS_INCLUDE_MATHUTILS_DETAIL_STATACCUMULATOR_H_
#define MATHUTILS_INCLUDE_MATHUTILS_DETAIL_STATACCUMULATOR_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <tuple>
#endif

namespace o2
{
namespace math_utils
{
namespace detail
{

struct StatAccumulator {
  // mean / RMS accumulator
  double sum = 0.;
  double sum2 = 0.;
  double wsum = 0.;
  int n = 0;

  void add(float v, float w = 1.)
  {
    const auto c = v * w;
    sum += c;
    sum2 += c * v;
    wsum += w;
    n++;
  }
  double getMean() const { return wsum > 0. ? sum / wsum : 0.; }

#ifndef GPUCA_GPUCODE_DEVICE
  template <typename T = float>
  std::tuple<T, T> getMeanRMS2() const
  {
    T mean = 0;
    T rms2 = 0;

    if (wsum) {
      const T wi = 1. / wsum;
      mean = sum * wi;
      rms2 = sum2 * wi - mean * mean;
    }

    return {mean, rms2};
  }
#endif

  StatAccumulator& operator+=(const StatAccumulator& other)
  {
    sum += other.sum;
    sum2 += other.sum2;
    wsum += other.wsum;
    return *this;
  }

  StatAccumulator operator+(const StatAccumulator& other) const
  {
    StatAccumulator res = *this;
    res += other;
    return res;
  }

  void clear()
  {
    sum = sum2 = wsum = 0.;
    n = 0;
  }
};

} // namespace detail
} // namespace math_utils
} // namespace o2

#endif /* MATHUTILS_INCLUDE_MATHUTILS_DETAIL_STATACCUMULATOR_H_ */
