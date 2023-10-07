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
#include "TF1.h"
#include "Foption.h"
#include "HFitInterface.h"
#include "TFitResultPtr.h"
#include "TFitResult.h"
#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Math/WrappedMultiTF1.h"
#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include "Framework/Logger.h"

namespace o2
{
namespace math_utils
{
/// fit 1D array of histogrammed data with generic root function
///
/// The code was extracted out of ROOT to be able to do fitting on an array with histogrammed data
/// instead of root histograms.
/// It is a stripped down version, so does not provide the same functionality.
/// To be used with care.
///
/// \param[in]  nbins size of the array and number of histogram bins
/// \param[in]  arr   array with elements
/// \param[in]  xMin  minimum range of the array
/// \param[in]  xMax  maximum range of the array
/// \param[in] func fit function
///
///
template <typename T>
TFitResultPtr fit(const size_t nBins, const T* arr, const T xMin, const T xMax, TF1& func, std::string_view option = "")
{
  Foption_t fitOption;
  ROOT::Fit::FitOptionsMake(ROOT::Fit::kHistogram, option.data(), fitOption);

  ROOT::Fit::DataRange range(xMin, xMax);
  ROOT::Fit::DataOptions opt;
  ROOT::Fit::BinData fitdata(opt, range);
  fitdata.Initialize(nBins, 1);

  // create an empty TFitResult
  std::shared_ptr<TFitResult> tfr(new TFitResult());
  // create the fitter from an empty fit result
  //std::shared_ptr<ROOT::Fit::Fitter> fitter(new ROOT::Fit::Fitter(std::static_pointer_cast<ROOT::Fit::FitResult>(tfr) ) );
  ROOT::Fit::Fitter fitter(tfr);
  //ROOT::Fit::FitConfig & fitConfig = fitter->Config();

  const double binWidth = double(xMax - xMin) / double(nBins);

  for (Int_t ibin = 0; ibin < nBins; ibin++) {
    const double x = double(xMin) + double(ibin + 0.5) * binWidth;
    const double y = double(arr[ibin]);
    const double ey = std::sqrt(y);
    fitdata.Add(x, y, ey);
  }

  const int special = func.GetNumber();
  const int npar = func.GetNpar();
  bool linear = func.IsLinear();
  if (special == 299 + npar) {
    linear = kTRUE; // for polynomial functions
  }
  // do not use linear fitter in these case
  if (fitOption.Bound || fitOption.Like || fitOption.Errors || fitOption.Gradient || fitOption.More || fitOption.User || fitOption.Integral || fitOption.Minuit) {
    linear = kFALSE;
  }

  if (special != 0 && !fitOption.Bound && !linear) {
    if (special == 100) {
      ROOT::Fit::InitGaus(fitdata, &func); // gaussian
    } else if (special == 400) {
      ROOT::Fit::InitGaus(fitdata, &func); // landau (use the same)
    } else if (special == 200) {
      ROOT::Fit::InitExpo(fitdata, &func); // exponential
    }
  }

  if ((linear || fitOption.Gradient)) {
    fitter.SetFunction(ROOT::Math::WrappedMultiTF1(func));
  } else {
    fitter.SetFunction(static_cast<const ROOT::Math::IParamMultiFunction&>(ROOT::Math::WrappedMultiTF1(func)));
  }

  // standard least square fit
  const bool fitok = fitter.Fit(fitdata, fitOption.ExecPolicy);
  if (!fitok) {
    LOGP(warning, "bad fit");
  }

  return TFitResultPtr(tfr);
}

/// fast median estimate of gaussian parameters for histogrammed data
///
/// \param[in]  nbins size of the array and number of histogram bins
/// \param[in]  arr   array with elements
/// \param[in]  xMin  minimum range of the array
/// \param[in]  xMax  maximum range of the array
/// \param[out] param return paramters of the fit (0-Constant, 1-Mean, 2-Sigma)
///
/// \return false on failure (empty data)

template <typename T>
bool medmadGaus(size_t nBins, const T* arr, const T xMin, const T xMax, std::array<double, 3>& param)
{
  int bStart = 0, bEnd = -1;
  double sum = 0, binW = double(xMax - xMin) / nBins, medVal = xMin;
  for (int i = 0; i < (int)nBins; i++) {
    auto v = arr[i];
    if (v) {
      if (!sum) {
        bStart = i;
      }
      sum += v;
      bEnd = i;
    }
  }
  if (bEnd < bStart) {
    return false;
  }
  bEnd++;
  double cum = 0, thresh = 0.5 * sum, frac0 = 0;
  int bid = bStart, prevbid = bid;
  while (bid < bEnd) {
    if (arr[bid] > 0) {
      cum += arr[bid];
      if (cum > thresh) {
        frac0 = 1. + (thresh - cum) / float(arr[bid]);
        medVal = xMin + binW * (bid + frac0);
        int bdiff = bid - prevbid - 1;
        if (bdiff > 0) {
          medVal -= bdiff * binW * 0.5; // account for the gap
          bid -= bdiff / 2;
        }
        break;
      }
      prevbid = bid;
    }
    bid++;
  }
  cum = 0.;
  double edgeL = frac0 + bid, edgeR = edgeL, dist = 0., wL = 0, wR = 0;
  while (1) {
    float amp = 0.;
    int bL = edgeL, bR = edgeR; // left and right bins
    if (edgeL > bStart) {
      wL = edgeL - bL;
      amp += arr[bL];
    } else {
      wL = 1.;
    }
    if (edgeR < bEnd) {
      wR = 1. + bR - edgeR;
      amp += arr[bR];
    } else {
      wR = 1.;
    }
    auto wdt = std::min(wL, wR);
    if (wdt < 1e-5) {
      wdt = std::max(wL, wR);
    }
    if (amp > 0) {
      amp *= wdt;
      cum += amp;
      if (cum >= thresh) {
        dist += wdt * (cum - thresh) / amp * 0.5;
        break;
      }
    }
    dist += wdt;
    edgeL -= wdt;
    edgeR += wdt;
  }
  constexpr double SQRT2PI = 2.5066283;
  param[1] = medVal;
  param[2] = dist * binW * 1.4826; // MAD -> sigma
  param[0] = sum * binW / (param[2] * SQRT2PI);
  return true;
}

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

  for (size_t i = 0; i < nBins; i++) {
    entries += arr[i];
    if (arr[i] > 0) {
      nfilled++;
    }
  }

  // TODO: Check why this is needed
  if (max < 4) {
    return -4;
  }
  if (entries < 12) {
    return -4;
  }

  if (rms < kTol) {
    return -4;
  }

  param[3] = entries;

  Int_t npoints = 0;
  for (size_t ibin = 0; ibin < nBins; ibin++) {
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
    if (TMath::Abs(par[1]) < kTol) {
      return -4;
    }
    if (TMath::Abs(par[2]) < kTol) {
      return -4;
    }
    param[1] = T(par[1] / (-2. * par[2]));
    param[2] = T(1. / TMath::Sqrt(TMath::Abs(-2. * par[2])));
    Double_t lnparam0 = par[0] + par[1] * param[1] + par[2] * param[1] * param[1];
    if (lnparam0 > 307) {
      return -4;
    }
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

// more optimal implementation of guassian fit via log-normal fit, appropriate for MT calls
// Only bins with values above minVal will be accounted.
// If applyMAD is true, the fit is done whithin the nSigmaMAD range of the preliminary estimate by MAD
template <typename T>
double fitGaus(size_t nBins, const T* arr, const T xMin, const T xMax, std::array<double, 3>& param,
               ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>>* covMat = nullptr,
               int minVal = 2, bool applyMAD = true)
{
  double binW = double(xMax - xMin) / nBins, s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, sy0 = 0, sy1 = 0, sy2 = 0, syy = 0;
  int np = 0;
  int bStart = 0, bEnd = (int)nBins;
  const float nSigmaMAD = 2.;
  if (applyMAD) {
    std::array<double, 3> madPar;
    if (!medmadGaus(nBins, arr, xMin, xMax, madPar)) {
      return -10;
    }
    bStart = std::max(bStart, int((madPar[1] - nSigmaMAD * madPar[2] - xMin) / binW));
    bEnd = std::min(bEnd, 1 + int((madPar[1] + nSigmaMAD * madPar[2] - xMin) / binW));
  }
  float x = xMin + (bStart - 0.5) * binW;
  for (int i = bStart; i < bEnd; i++) {
    x += binW;
    auto v = arr[i];
    if (v < 0) {
      throw std::runtime_error("Log-normal fit is possible only with non-negative data");
    }
    if (v < minVal) {
      continue;
    }
    double y = std::log(v), err2i = v, err2iX = err2i, err2iY = err2i * y;
    s0 += err2iX;
    s1 += (err2iX *= x);
    s2 += (err2iX *= x);
    s3 += (err2iX *= x);
    s4 += (err2iX *= x);
    sy0 += err2iY;
    syy += err2iY * y;
    sy1 += (err2iY *= x);
    sy2 += (err2iY *= x);
    np++;
  }
  if (np < 1) {
    return -10;
  }
  auto recover = [&param, binW, np, s0, s1, s2, sy0]() {
    param[0] = std::exp(sy0 / s0); // recover center of gravity
    param[1] = s1 / s0;            // mean x;
    param[2] = np == 1 ? binW / std::sqrt(12) : std::sqrt(std::abs(param[1] * param[1] - s2 / s0));
  };
  if (np < 3) {
    recover();
    return -np;
  }
  ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>> m33{};
  ROOT::Math::SVector<double, 3> v3{sy0, sy1, sy2};
  m33(0, 0) = s0;
  m33(1, 0) = s1;
  m33(1, 1) = m33(2, 0) = s2;
  m33(2, 1) = s3;
  m33(2, 2) = s4;
  int res = 0;
  auto m33i = m33.Inverse(res);
  if (res) {
    recover();
    LOG(error) << np << " points collected, matrix inversion failed " << m33;
    return -10;
  }
  auto v = m33i * v3;
  if (v(2) >= 0.) { // fit failed, use mean amd RMS
    recover();
    return -3;
  }

  double chi2 = v(0) * v(0) * s0 + v(1) * v(1) * s2 + v(2) * v(2) * s4 + syy +
                2. * (v(0) * v(1) * s1 + v(0) * v(2) * s2 + v(1) * v(2) * s3 - v(0) * sy0 - v(1) * sy1 - v(2) * sy2);
  param[1] = -0.5 * v(1) / v(2);
  param[2] = 1. / std::sqrt(-2. * v(2));
  param[0] = std::exp(v(0) - param[1] * param[1] * v(2));
  if (std::isnan(param[0]) || std::isnan(param[1]) || std::isnan(param[2])) {
    recover();
    return -3;
  }
  if (covMat) {
    // build jacobian of transformation from log-normal to normal params
    ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepStd<double, 3, 3>> j33{};
    j33(0, 0) = param[0];
    j33(0, 1) = param[0] * param[1];
    j33(0, 2) = j33(0, 1) * param[1];
    j33(1, 1) = -0.5 / v(2);
    j33(1, 2) = -param[1] / v(2);
    j33(2, 2) = param[2] * j33(1, 1);
    *covMat = ROOT::Math::Similarity(j33, m33i);
  }
  return np > 3 ? chi2 / (np - 3.) : 0.;
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
  if (sum == 0) {
    return data;
  }
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

//___________________________________________________________________
template <typename T>
T selKthMin(int k, int np, T* arr)
{
  // Returns the k th smallest value in the array. The input array will be rearranged
  // to have this value in location arr[k] , with all smaller elements moved before it
  // (in arbitrary order) and all larger elements after (also in arbitrary order).
  // From Numerical Recipes in C++

  int i, j, mid, ir = np - 1, l = 0;
  T a;
  for (;;) {
    if (ir <= l + 1) {
      if (ir == l + 1 && arr[ir] < arr[l]) {
        std::swap(arr[l], arr[ir]);
      }
      return arr[k];
    } else {
      int mid = (l + ir) >> 1, i = l + 1;
      std::swap(arr[mid], arr[i]);
      if (arr[i] > arr[ir]) {
        std::swap(arr[i], arr[ir]);
      }
      if (arr[l] > arr[ir]) {
        std::swap(arr[l], arr[ir]);
      }
      if (arr[i] > arr[l]) {
        std::swap(arr[i], arr[l]);
      }
      j = ir;
      a = arr[l];
      for (;;) {
        do {
          i++;
        } while (arr[i] < a);
        do {
          j--;
        } while (arr[j] > a);
        if (j < i) {
          break;
        }
        std::swap(arr[i], arr[j]);
      }
      arr[l] = arr[j];
      arr[j] = a;
      if (j >= k) {
        ir = j - 1;
      }
      if (j <= k) {
        l = i;
      }
    }
  }
}

//___________________________________________________________________
template <typename T>
T MAD2Sigma(int np, T* y)
{
  // Sigma calculated from median absolute deviations, https://en.wikipedia.org/wiki/Median_absolute_deviation
  // the input array is modified
  if (np < 2) {
    return 0;
  }
  int nph = np >> 1;
  float median = (np & 0x1) ? selKthMin(nph, np, y) : 0.5 * (selKthMin(nph - 1, np, y) + selKthMin(nph, np, y));
  // build abs differences to median
  for (int i = np; i--;) {
    y[i] = std::abs(y[i] - median);
  }
  // now get median of abs deviations
  median = (np & 0x1) ? selKthMin(nph, np, y) : 0.5 * (selKthMin(nph - 1, np, y) + selKthMin(nph, np, y));
  return median * 1.4826; // convert to Gaussian sigma
}

} // namespace math_utils
} // namespace o2
#endif
