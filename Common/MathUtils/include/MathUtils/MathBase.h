// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MATHUTILS_MATHBASE_H_
#define ALICEO2_MATHUTILS_MATHBASE_H_

/// \file   MathBase.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <array>

#include "Rtypes.h"
#include "TLinearFitter.h"
#include "TVectorD.h"
#include "TMath.h"

#include <fairlogger/Logger.h>

namespace o2
{
namespace math_utils
{
namespace math_base
{
/// fast fit of an array with ranges (histogram) with gaussian function
///
/// Fitting procedure:
/// 1. Step - make logarithm
/// 2. Linear  fit (parabola) - more robust, always converges, fast
///
/// \param[in]  nbins size of the array and number of histogram bins
/// \param[in]  arr   array with elements
/// \param[in]  xMin  minimum range of the array
/// \param[in]  xMax  maximum range of the array
/// \param[out] param return paramters of the fit (0-Constant, 1-Mean, 2-Sigma, 3-Sum)
///
/// \return chi2 or exit code
///          >0: the chi2 returned by TLinearFitter
///          -3: only three points have been used for the calculation - no fitter was used
///          -2: only two points have been used for the calculation - center of gravity was uesed for calculation
///          -1: only one point has been used for the calculation - center of gravity was uesed for calculation
///          -4: invalid result!!
///
//template <typename T>
//Double_t  fitGaus(const size_t nBins, const T *arr, const T xMin, const T xMax, std::vector<T>& param);
template <typename T>
Double_t fitGaus(const size_t nBins, const T* arr, const T xMin, const T xMax, std::vector<T>& param)
{
  static TLinearFitter fitter(3, "pol2");
  static TMatrixD mat(3, 3);
  static Double_t kTol = mat.GetTol();
  fitter.StoreData(kFALSE);
  fitter.ClearPoints();
  TVectorD par(3);
  TVectorD sigma(3);
  TMatrixD A(3, 3);
  TMatrixD b(3, 1);
  T rms = TMath::RMS(nBins, arr);
  T max = TMath::MaxElement(nBins, arr);
  T binWidth = (xMax - xMin) / T(nBins);

  Float_t meanCOG = 0;
  Float_t rms2COG = 0;
  Float_t sumCOG = 0;

  Float_t entries = 0;
  Int_t nfilled = 0;

  param.resize(4);
  param[0] = 0.;
  param[1] = 0.;
  param[2] = 0.;
  param[3] = 0.;

  for (Int_t i = 0; i < nBins; i++) {
    entries += arr[i];
    if (arr[i] > 0)
      nfilled++;
  }

  // TODO: Check why this is needed
  if (max < 4)
    return -4;
  if (entries < 12)
    return -4;

  if (rms < kTol)
    return -4;

  param[3] = entries;

  Int_t npoints = 0;
  for (Int_t ibin = 0; ibin < nBins; ibin++) {
    Float_t entriesI = arr[ibin];
    if (entriesI > 1) {
      Double_t xcenter = xMin + (ibin + 0.5) * binWidth;
      Double_t error = 1. / TMath::Sqrt(entriesI);
      Double_t val = TMath::Log(Float_t(entriesI));
      fitter.AddPoint(&xcenter, val, error);
      if (npoints < 3) {
        A(npoints, 0) = 1;
        A(npoints, 1) = xcenter;
        A(npoints, 2) = xcenter * xcenter;
        b(npoints, 0) = val;
        meanCOG += xcenter * entriesI;
        rms2COG += xcenter * entriesI * xcenter;
        sumCOG += entriesI;
      }
      npoints++;
    }
  }

  Double_t chi2 = 0;
  if (npoints >= 3) {
    if (npoints == 3) {
      //analytic calculation of the parameters for three points
      A.Invert();
      TMatrixD res(1, 3);
      res.Mult(A, b);
      par[0] = res(0, 0);
      par[1] = res(0, 1);
      par[2] = res(0, 2);
      chi2 = -3.;
    } else {
      // use fitter for more than three points
      fitter.Eval();
      fitter.GetParameters(par);
      fitter.GetCovarianceMatrix(mat);
      chi2 = fitter.GetChisquare() / Double_t(npoints);
    }
    if (TMath::Abs(par[1]) < kTol)
      return -4;
    if (TMath::Abs(par[2]) < kTol)
      return -4;

    param[1] = T(par[1] / (-2. * par[2]));
    param[2] = T(1. / TMath::Sqrt(TMath::Abs(-2. * par[2])));
    Double_t lnparam0 = par[0] + par[1] * param[1] + par[2] * param[1] * param[1];
    if (lnparam0 > 307)
      return -4;
    param[0] = TMath::Exp(lnparam0);

    return chi2;
  }

  if (npoints == 2) {
    //use center of gravity for 2 points
    meanCOG /= sumCOG;
    rms2COG /= sumCOG;
    param[0] = max;
    param[1] = meanCOG;
    param[2] = TMath::Sqrt(TMath::Abs(meanCOG * meanCOG - rms2COG));
    chi2 = -2.;
  }
  if (npoints == 1) {
    meanCOG /= sumCOG;
    param[0] = max;
    param[1] = meanCOG;
    param[2] = binWidth / TMath::Sqrt(12);
    chi2 = -1.;
  }
  return chi2;
}

/// struct for returning statistical parameters
///
/// \todo make type templated?
/// \todo use Vc
struct StatisticsData {
  double mCOG{0};    ///< calculated centre of gravity
  double mStdDev{0}; ///< standard deviation
  double mSum{0};    ///< sum of values
};

/// calculate statistical parameters on a binned array
///
/// The function assumes a binned array of
/// \param nBins size of the array
/// \param xMin lower histogram bound
/// \param xMax upper histogram bound
/// \todo make return type templated?
template <typename T>
StatisticsData getStatisticsData(const T* arr, const size_t nBins, const double xMin, const double xMax)
{
  double mean = 0;
  double rms2 = 0;
  double sum = 0;
  size_t npoints = 0;

  double binWidth = (xMax - xMin) / (double)nBins;

  StatisticsData data;
  // in case something went wrong the COG is the histogram lower limit
  data.mCOG = xMin;

  for (size_t ibin = 0; ibin < nBins; ++ibin) {
    double entriesI = (double)arr[ibin];
    double xcenter = xMin + (ibin + 0.5) * binWidth; // +0.5 to shift to bin centre
    if (entriesI > 0) {
      mean += xcenter * entriesI;
      rms2 += xcenter * entriesI * xcenter;
      sum += entriesI;
      ++npoints;
    }
  }
  if (sum == 0)
    return data;
  mean /= sum;

  data.mCOG = mean;
  // exception in case of only one bin is filled
  // set the standard deviation to bin width over sqrt(12)
  rms2 /= sum;
  if (npoints == 1) {
    data.mStdDev = binWidth / std::sqrt(12.);
  } else {
    data.mStdDev = std::sqrt(std::abs(rms2 - mean * mean));
  }

  data.mSum = sum;

  return data;
}

/// median of values in a std::vector
///
/// we need to make a copy of the vector since we need to sort it
/// based on this discussion: https://stackoverflow.com/questions/1719070/what-is-the-right-approach-when-using-stl-container-for-median-calculation/1719155#1719155
/// \todo Is there a better way to do this?
template <typename T, typename R = double>
R median(std::vector<T> v)
{
  if (v.empty()) {
    return R{};
  }
  auto n = v.size() / 2;
  nth_element(v.begin(), v.begin() + n, v.end());
  auto med = R{v[n]};
  if (!(v.size() & 1)) { //If the set size is even
    auto max_it = max_element(v.begin(), v.begin() + n);
    med = R{(*max_it + med) / 2.0};
  }
  return med;
}

/// Fills the index vector with sorted indices of the input vector.
/// The input vector is not modified (similar to TMath::Sort()).
/// \param values Vector to be indexed
/// \param index Vector to hold the sorted indices (must have the same size as values)
template <typename T>
void SortData(std::vector<T> const& values, std::vector<size_t>& index)
{
  if (values.size() != index.size()) {
    LOG(error) << "Vector with values must have same size as vector for indices";
    return;
  }
  std::iota(index.begin(), index.end(), static_cast<size_t>(0));
  std::sort(index.begin(), index.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
}

/// LTM : Trimmed mean of unbinned array
///
/// Robust statistic to estimate properties of the distribution
/// To handle binning error special treatment
/// for definition of unbinned data see:
///     http://en.wikipedia.org/w/index.php?title=Trimmed_estimator&oldid=582847999
/// \param data Input vector (unsorted)
/// \param index Vector with indices of sorted input data
/// \param params Array with characteristics of distribution
/// \param fracKeep Fraction of data to be kept
/// \return Flag if successfull
/// Characteristics:
/// -# area
/// -# mean
/// -# rms
/// -# error estimate of mean
/// -# error estimate of RMS
/// -# first accepted element (of sorted array)
/// -# last accepted  element (of sorted array)
template <typename T>
bool LTMUnbinned(const std::vector<T>& data, std::vector<size_t>& index, std::array<float, 7>& params, float fracKeep)
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

/// Rearranges the input vector in the order given by the index vector
/// \param data Input vector
/// \param index Index vector
template <typename T>
void Reorder(std::vector<T>& data, const std::vector<size_t>& index)
{
  // rearange data in order given by index
  if (data.size() != index.size()) {
    LOG(error) << "Reordering not possible if number of elements in index container different from the data container";
    return;
  }
  std::vector<T> tmp(data);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = tmp[index[i]];
  }
}

/// Compare this function to LTMUnbinned.
/// A target sigma of the distribution can be specified and it will be trimmed to match that target.
/// \param data Input vector (unsorted)
/// \param index Vector with indices of sorted input data
/// \param params Array with characteristics of distribution
/// \param fracKeepMin Minimum fraction to keep of the input data
/// \param sigTgt Target distribution sigma
/// \param sorted Flag if the data is already sorted
/// \return Flag if successfull
template <typename T>
bool LTMUnbinnedSig(const std::vector<T>& data, std::vector<size_t>& index, std::array<float, 7>& params, float fracKeepMin, float sigTgt, bool sorted = false)
{
  int nPoints = data.size();
  std::vector<double> wx(nPoints);
  std::vector<double> wx2(nPoints);

  if (!sorted) {
    // sort in increasing order
    SortData(data, index);
  } else {
    // array is already sorted
    std::iota(index.begin(), index.end(), 0);
  }
  // build cumulants
  double sum1 = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < nPoints; i++) {
    double x = data[index[i]];
    sum1 += x;
    sum2 += x * x;
    wx[i] = sum1;
    wx2[i] = sum2;
  }
  int keepMax = nPoints;
  int keepMin = fracKeepMin * nPoints;
  if (keepMin > keepMax) {
    keepMin = keepMax;
  }
  float sigTgt2 = sigTgt * sigTgt;
  //
  while (true) {
    double maxRMS = wx2.back() + 1e6;
    int keepN = (keepMax + keepMin) / 2;
    if (keepN < 2) {
      return false;
    }
    params[0] = keepN;
    int limI = nPoints - keepN + 1;
    for (int i = 0; i < limI; ++i) {
      const int limJ = i + keepN - 1;
      sum1 = wx[limJ] - (i ? wx[i - 1] : 0.);
      sum2 = wx2[limJ] - (i ? wx2[i - 1] : 0.);
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
} // namespace math_base
} // namespace math_utils
} // namespace o2
#endif
