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
#include <boost/histogram.hpp>

namespace o2
{
namespace utils
{

/// \class BinCenterView
/// \brief  Axis iterator over bin centers.
template <typename AxisIterator>
class BinCenterView
{
 public:
  BinCenterView(AxisIterator iter);
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

/// \class BinUpperView
/// \brief  Axis iterator over bin upper edges.
template <typename AxisIterator>
class BinUpperView
{
 public:
  BinUpperView(AxisIterator iter);
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

/// \class BinLowerView
/// \brief  Axis iterator over bin lower edges.
template <typename AxisIterator>
class BinLowerView
{
 public:
  BinLowerView(AxisIterator iter);
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

/// \struct fitResult
/// \brief Struct to store the results of the fit.
template <typename T, int nparams>
struct fitResult {
  double mChi2;                          ///< chi2 of the fit
  std::array<T, nparams> mFitParameters; ///< parameters of the fit (0-Constant, 1-Mean, 2-Sigma,  3-Sum)

  fitResult(double chi2, std::initializer_list<T> params)
  {
    static_assert(params.size() == nparams, "Exceed number of params");
    mChi2 = chi2;
    std::copy(params.begin(), params.end(), mFitParameters.begin());
  }

  /// \brief Get the paratmeters of a fit result. Ex:  result.getParameter<1>();
  template <int index>
  T getParameter() const
  {
    static_assert(index == mFitParameters.size(), "Acessing invalid parameter");
    return mFitParameters[index];
  }

  /// \brief Set the paratmeters of a fit result. Ex:  result.setParameter<1>(T(value)));
  template <int index>
  void setParameter(T parameter)
  {
    static_assert(index == mFitParameters.size(), "Attempting to set invalid parameter");
    mFitParameters[index] = parameter;
  }

  /// \brief Set the chi2 of the fit result.
  double setChi2(double chi2In)
  {
    mChi2 = chi2In;
  }
};

/**
   * \enum FitGausError_t
   * \brief Error code for invalid result in the fitGaus process
   */
enum class FitGausError_t {
  FIT_ERROR, ///< Fit procedure returned invalid result
};

/// \brief Printing an error message when then fit returns an invalid result
/// \param errorcode Error of the type FitGausError_t, thrown when fit result is invalid.
std::string createErrorMessage(FitGausError_t errorcode)
{
  return "[Error]: Fit return an invalid result.";
}

/// \brief Function to fit histogram to a gaussian using iterators.
/// \param first begin iterator of the histogram
/// \param last end iterator of the histogram
/// \param axisFirst axis iterator over the bin centers
/// \return result
///      result is of the type fitResult, which contains 4 parameters (0-Constant, 1-Mean, 2-Sigma,  3-Sum)
///
template <typename T, typename Iterator, typename AxisIterator>
fitResult<T, 4> fitGaus(Iterator first, Iterator last, AxisIterator axisfirst)
{
  TLinearFitter fitter(3, "pol2");
  TMatrixD mat(3, 3);
  Double_t kTol = mat.GetTol();
  fitter.StoreData(kFALSE);
  fitter.ClearPoints();
  TVectorD par(3);
  TVectorD sigma(3);
  TMatrixD A(3, 3);
  TMatrixD b(3, 1);
  T rms = TMath::RMS(first, last);
  T xMax = std::max_element(first, last);
  T xMin = std::min_element(first, last);
  auto nbins = last - first;
  const double binWidth = double(xMax - xMin) / double(nbins);

  Float_t meanCOG = 0;
  Float_t rms2COG = 0;
  Float_t sumCOG = 0;

  Float_t entries = 0;
  Int_t nfilled = 0;

  for (auto iter = first, axisiter = axisfirst; iter != last; iter++, axisiter++) {
    entries += *iter;
    if (*iter > 0) {
      nfilled++;
    }
  }

  // TODO: Check why this is needed
  if (xMax < 4) {
    throw FitGausError_t::FIT_ERROR;
  }
  if (entries < 12) {
    throw FitGausError_t::FIT_ERROR;
  }

  if (rms < kTol) {
    throw FitGausError_t::FIT_ERROR;
  }

  fitResult<T, 4> result;
  result.setParameter<3>(entries);

  int ibin = 0;
  Int_t npoints = 0;
  for (auto iter = first, axisiter = axisfirst; iter != last; iter++, axisiter++) {
    if (nbins > 1) {
      Double_t x = (*axisiter + *(axisiter + 1)) / 2.;
      Double_t y = *iter;
      Double_t ey = std::sqrt(y);
      fitter.AddPoint(&x, y, ey);
      if (npoints < 3) {
        A(npoints, 0) = 1;
        A(npoints, 1) = x;
        A(npoints, 2) = x * x;
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
      result.setChi2(fitter.GetChisquare() / Double_t(npoints));
    }
    if (TMath::Abs(par[1]) < kTol) {
      throw FitGausError_t::FIT_ERROR;
      ;
    }
    if (TMath::Abs(par[2]) < kTol) {
      throw FitGausError_t::FIT_ERROR;
      ;
    }

    T param1 = T(par[1] / (-2. * par[2]));
    result.setParameter<1>(param1);
    result.setParameter<2>(T(1. / TMath::Sqrt(TMath::Abs(-2. * par[2]))));
    auto lnparam0 = par[0] + par[1] * param1 + par[2] * param1 * param1;
    if (lnparam0 > 307) {
      throw FitGausError_t::FIT_ERROR;
      ;
    }
    result.setParameter<0>(TMath::Exp(lnparam0));

    return result;
  }

  if (npoints == 2) {
    //use center of gravity for 2 points
    meanCOG /= sumCOG;
    rms2COG /= sumCOG;
    result.setParameter<0>(xMax);
    result.setParameter<1>(meanCOG);
    result.setParameter<2>(TMath::Sqrt(TMath::Abs(meanCOG * meanCOG - rms2COG)));
    result.setChi2(-2);
  }
  if (npoints == 1) {
    meanCOG /= sumCOG;
    result.setParameter<0>(xMax);
    result.setParameter<1>(meanCOG);
    result.setParameter<2>(binWidth / TMath::Sqrt(12));
    result.setChi2(-1);
  }
  return result;
}

template <typename valuetype, typename... axes>
fitResult<valuetype, 4> fitBoostHistoWithGaus(boost::histogram::histogram<axes...>& hist)
{
  return fitGaus(hist.begin(), hist.end(), BinCenterView(hist.axis(0).begin()));
}

} // end namespace utils
} // end namespace o2