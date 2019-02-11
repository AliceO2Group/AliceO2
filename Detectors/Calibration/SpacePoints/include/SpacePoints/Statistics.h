// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Statistics.h
/// \brief Ported methods from AliRoot/STAT/TStatToolkit.h needed for TPC space point calibration
///
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_CALIB_STATISTICS_H_
#define ALICEO2_CALIB_STATISTICS_H_

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <fairlogger/Logger.h>

namespace o2
{

namespace calib
{

namespace stat
{
// fill the 'index' vector with sorted indices of the vector 'values'
// the input vector 'values' is not modified
// compare to TMath::Sort()
template <typename T>
inline void SortData(std::vector<T> const& values, std::vector<size_t>& index)
{
  if (values.size() != index.size()) {
    LOG(error) << "Vector with values must have same size as vector for indices";
    return;
  }
  std::iota(index.begin(), index.end(), static_cast<size_t>(0));
  std::sort(index.begin(), index.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
}
template void SortData<unsigned short>(std::vector<unsigned short> const& values, std::vector<size_t>& index);
template void SortData<float>(std::vector<float> const& values, std::vector<size_t>& index);

//
// LTM : Trimmed mean of unbinned array
//
// Robust statistic to estimate properties of the distribution
// To handle binning error special treatment
// for definition of unbinned data see:
//     http://en.wikipedia.org/w/index.php?title=Trimmed_estimator&oldid=582847999
//
// Function parameters:
//     data    - data vector (unsorted)
//     index   - vector with indices for sorted data
//     params  - vector storing the following parameters as result
//             - 0 - area
//             - 1 - mean
//             - 2 - rms
//             - 3 - error estimate of mean
//             - 4 - error estimate of RMS
//             - 5 - first accepted element (of sorted array)
//             - 6 - last accepted  element (of sorted array)
//
//
inline bool LTMUnbinned(const std::vector<float>& data, std::vector<size_t>& index, std::vector<float>& params, float fracKeep)
{
  int nPoints = data.size();
  std::vector<float> w(2 * nPoints);
  int nKeep = nPoints * fracKeep;
  if (nKeep > nPoints) {
    nKeep = nPoints;
  }
  if (nKeep < 2) {
    return false;
  }
  // sort in increasing order
  SortData(data, index);
  // build cumulants
  double sum1 = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < nPoints; i++) {
    double x = data[index[i]];
    sum1 += x;
    sum2 += x * x;
    w[i] = sum1;
    w[i + nPoints] = sum2;
  }
  double maxRMS = sum2 + 1e6;
  params[0] = nKeep;
  int limI = nPoints - nKeep + 1; // lowest possible bin to accept
  for (int i = 0; i < limI; i++) {
    const int limJ = i + nKeep - 1; // highest accepted bin
    sum1 = static_cast<double>(w[limJ]) - static_cast<double>(i ? w[i - 1] : 0.);
    sum2 = static_cast<double>(w[nPoints + limJ]) - static_cast<double>(i ? w[nPoints + i - 1] : 0.);
    const double mean = sum1 / nKeep;
    const double rms2 = sum2 / nKeep - mean * mean;
    if (rms2 > maxRMS) {
      continue;
    }
    maxRMS = rms2;
    params[1] = mean;
    params[2] = rms2;
    params[5] = i;
    params[6] = limJ;
  }
  //
  if (params[2] < 0) {
    LOG(error) << "Rounding error: RMS = " << params[2] << " < 0";
    return false;
  }
  params[2] = std::sqrt(params[2]);
  params[3] = params[2] / std::sqrt(params[0]); // error on mean
  params[4] = params[3] / std::sqrt(2.0);       // error on RMS
  return true;
}

inline void Reorder(std::vector<float>& data, const std::vector<size_t>& index)
{
  // rearange data in order given by index
  if (data.size() != index.size()) {
    LOG(error) << "Reordering not possible if number of elements in index container different from the data container";
    return;
  }
  std::vector<float> tmp(data);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = tmp[index[i]];
  }
}

//
// LTM : Trimmed mean of unbinned array
//
// Robust statistic to estimate properties of the distribution
// To handle binning error special treatment
// for definition of unbinned data see:
//     http://en.wikipedia.org/w/index.php?title=Trimmed_estimator&oldid=582847999
//
// Function parameters:
//     data    - data vector (unsorted)
//     params  - vector storing the following parameters as result
//             - 0 - area
//             - 1 - mean
//             - 2 - rms
//             - 3 - error estimate of mean
//             - 4 - error estimate of RMS
//             - 5 - first accepted element (of sorted array)
//             - 6 - last accepted  element (of sorted array)
//
//
inline bool LTMUnbinnedSig(const std::vector<float>& data, std::vector<size_t>& index, std::vector<float>& params, float fracKeepMin, float sigTgt, bool sorted = false)
{
  int nPoints = data.size();
  std::vector<float> w(2 * nPoints);

  if (!sorted) {
    // sort in increasing order
    SortData(data, index);
  } else {
    // array is already sorted
    for (int i = 0; i < nPoints; ++i) {
      index[i] = i;
    }
  }
  // build cumulants
  double sum1 = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < nPoints; i++) {
    double x = data[index[i]];
    sum1 += x;
    sum2 += x * x;
    w[i] = sum1;
    w[i + nPoints] = sum2;
  }
  int keepMax = nPoints;
  int keepMin = fracKeepMin * nPoints;
  if (keepMin > keepMax) {
    keepMin = keepMax;
  }
  float sigTgt2 = sigTgt * sigTgt;
  //
  while (true) {
    double maxRMS = sum2 + 1e6;
    int keepN = (keepMax + keepMin) / 2;
    if (keepN < 2) {
      return false;
    }
    params[0] = keepN;
    int limI = nPoints - keepN + 1;
    for (int i = 0; i < limI; ++i) {
      const int limJ = i + keepN - 1;
      sum1 = w[limJ] - (i ? w[i - 1] : 0.);
      sum2 = w[limJ + nPoints] - (i ? w[nPoints + i - 1] : 0.);
      const double mean = sum1 / keepN;
      const double rms2 = sum2 / keepN - mean * mean;
      if (rms2 > maxRMS) {
        continue;
      }
      maxRMS = rms2;
      params[1] = mean;
      params[2] = rms2;
      params[5] = i;
      params[6] = limJ;
    }
    if (maxRMS < sigTgt2) {
      keepMin = keepN;
    } else {
      keepMax = keepN;
    }
    if (keepMin >= keepMax - 1) {
      break;
    }
  }
  params[2] = std::sqrt(params[2]);
  params[3] = params[2] / std::sqrt(params[0]); // error on mean
  params[4] = params[3] / std::sqrt(2.0);       // error on RMS
  return true;
}

} // namespace stat

} // namespace calib

} // namespace o2

#endif
