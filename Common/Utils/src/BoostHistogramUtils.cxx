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

/// \file   BoostHistogramUtils.cxx
/// \author Hannah Bossi, hannah.bossi@yale.edu

#include "CommonUtils/BoostHistogramUtils.h"

namespace o2
{
namespace utils
{

std::string createErrorMessageFitGaus(o2::utils::FitGausError_t errorcode)
{
  switch (errorcode) {
    case FitGausError_t::FIT_ERROR_MIN:
      return "Gaus fit failed! xMax < 4";
    case FitGausError_t::FIT_ERROR_MAX:
      return "Gaus fit failed! xMax too large";
    case FitGausError_t::FIT_ERROR_ENTRIES:
      return "Gaus fit failed! entries < 12";
    case FitGausError_t::FIT_ERROR_KTOL_MEAN:
      return "Gaus fit failed! std::abs(par[1]) < kTol";
    case FitGausError_t::FIT_ERROR_KTOL_SIGMA:
      return "Gaus fit failed! std::abs(par[2]) < kTol";
    case FitGausError_t::FIT_ERROR_KTOL_RMS:
      return "Gaus fit failed! RMS < kTol";
    default:
      return "Gaus fit failed! Unknown error code";
  }
  return "Gaus fit failed! Unknown error code";
}

boostHisto1d_VarAxis boosthistoFromRoot_1D(TH1D* inHist1D)
{
  // first setup the proper boost histogram
  int nBins = inHist1D->GetNbinsX();
  std::vector<double> binEdges;
  for (int i = 0; i < nBins + 1; i++) {
    binEdges.push_back(inHist1D->GetBinLowEdge(i + 1));
  }
  boostHisto1d_VarAxis mHisto = boost::histogram::make_histogram(boost::histogram::axis::variable<>(binEdges));

  // trasfer the acutal values
  for (Int_t x = 1; x < nBins + 1; x++) {
    mHisto.at(x - 1) = inHist1D->GetBinContent(x);
  }
  return mHisto;
}

boostHisto2d_VarAxis boostHistoFromRoot_2D(TH2D* inHist2D)
{
  // Get Xaxis binning
  const int nBinsX = inHist2D->GetNbinsX();
  std::vector<double> binEdgesX;
  for (int i = 0; i < nBinsX + 1; i++) {
    binEdgesX.push_back(inHist2D->GetXaxis()->GetBinLowEdge(i + 1));
  }
  // Get Yaxis binning
  const int nBinsY = inHist2D->GetNbinsY();
  std::vector<double> binEdgesY;
  for (int i = 0; i < nBinsY + 1; i++) {
    binEdgesY.push_back(inHist2D->GetYaxis()->GetBinLowEdge(i + 1));
  }

  boostHisto2d_VarAxis mHisto = boost::histogram::make_histogram(boost::histogram::axis::variable<>(binEdgesX), boost::histogram::axis::variable<>(binEdgesY));

  // trasfer the acutal values
  for (Int_t x = 1; x < nBinsX + 1; x++) {
    for (Int_t y = 1; y < nBinsY + 1; y++) {
      mHisto.at(x - 1, y - 1) = inHist2D->GetBinContent(x, y);
    }
  }
  return mHisto;
}
/// \brief Printing an error message when then fit returns an invalid result
/// \param errorcode Error of the type FitGausError_t, thrown when fit result is invalid.
std::string createErrorMessage(o2::utils::FitGausError_t errorcode)
{
  return "[Error]: Fit return an invalid result.";
}
} // namespace utils
} // namespace o2