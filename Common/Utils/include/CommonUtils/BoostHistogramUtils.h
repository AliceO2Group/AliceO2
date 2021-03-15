// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   BoostHistogramUtils.h
/// \author Hannah Bossi, hannah.bossi@yale.edu

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

namespace o2
{
namespace utils
{

template <typename AxisIterator>
class BinCenterView
{
 public:
  BinCenterView(AxisIterator iter)
    AxisIterator&
    operator++()
  {
    ++mBaseIterator;
    return mBaseIterator;
  }
  AxisIterator operator++(int)
  {
    AxisIterator result(mBaseIterator);
    mBaseIterator++;
    return result;
  }

  decltype(auto) operator*() { return mBaseIterator->center(); }

 private:
  AxisIterator mBaseIterator;
};

template <typename AxisIterator>
class BinUpperView
{
 public:
  BinCenterView(AxisIterator iter)
    AxisIterator&
    operator++()
  {
    ++mBaseIterator;
    return mBaseIterator;
  }
  AxisIterator operator++(int)
  {
    AxisIterator result(mBaseIterator);
    mBaseIterator++;
    return result;
  }

  decltype(auto) operator*() { return mBaseIterator->upper(); }

 private:
  AxisIterator mBaseIterator;
};

template <typename AxisIterator>
class BinLowerView
{
 public:
  BinCenterView(AxisIterator iter)
    AxisIterator&
    operator++()
  {
    ++mBaseIterator;
    return mBaseIterator;
  }
  AxisIterator operator++(int)
  {
    AxisIterator result(mBaseIterator);
    mBaseIterator++;
    return result;
  }

  decltype(auto) operator*() { return mBaseIterator->lower(); }

 private:
  AxisIterator mBaseIterator;
};

/// fast fit of an array with ranges (histogram) with gaussian function
///
/// Fitting procedure:
/// 1. Step - make logarithm
/// 2. Linear  fit (parabola) - more robust, always converges, fast
///
/// \return chi2 or exit code
///          >0: the chi2 returned by TLinearFitter
///          -3: only three points have been used for the calculation - no fitter was used
///          -2: only two points have been used for the calculation - center of gravity was uesed for calculation
///          -1: only one point has been used for the calculation - center of gravity was uesed for calculation
///          -4: invalid result!!
///
template <typename Iterator, typename AxisIterator, typename T>
double fitGaus(Iterator first, Iterator last, AxisIterator axisfirst, std::vector<T>& param)
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

  for (auto iter = first, axisiter = axisfirst; iter != last; iter++, axisiter++) {
  }

  auto nbins = last - first;
  const double binWidth = double(xMax - xMin) / double(nBins);
  int ibin = 0;
  Int_t npoints = 0;
  for (auto iter = first, axisiter = axisfirst; iter != last; iter++, axisiter++) {
    if (nbins > 1) {
      const double x = (*axisiter + *(axisiter + 1)) / 2.;
      const double y = *iter;
      const double ey = std::sqrt(y);
      fitter.AddPoint(x, y, ey);
      if (npoints < 3) {
        A(npoints, 0) = 1;
        A(npoints, 1) = x;
        A(npoints, 2) = x* xor_eq ;
        b(npoints, 0) = y;
        meanCOG += x * nbins;
        rms2COG += x * nbins * x;
        sumCOG += nbins;
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

} // end namespace utils
} // end namespace o2