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


// \brief Convert a 2D root histogram to a Boost histogram
decltype(auto) boostHistoFromRoot_2D(TH2D* inHist2D)
{
  // first setup the proper boost histogram
  int nBinsX = inHist2D->GetNbinsX();
  int xMin = inHist2D->GetXaxis()->GetXmin();
  int xMax = inHist2D->GetXaxis()->GetXmax();
  const char* xTitle = inHist2D->GetXaxis()->GetTitle();
  int nBinsY = inHist2D->GetNbinsY();
  int yMin = inHist2D->GetYaxis()->GetXmin();
  int yMax = inHist2D->GetYaxis()->GetXmax();
  const char* yTitle = inHist2D->GetYaxis()->GetTitle();
  auto mHisto = boost::histogram::make_histogram(boost::histogram::axis::regular<>(nBinsX, xMin, xMax, xTitle), boost::histogram::axis::regular<>(nBinsY, yMin,
                                                                                                                                                  yMax, yTitle));
  // trasfer the acutal values
  for (Int_t x = 1; x < nBinsX + 1; x++) {
    for (Int_t y = 1; y < nBinsY + 1; y++) {
      mHisto.at(x, y) = inHist2D->GetBinContent(x, y);
    }
  }
  return mHisto;
}


/// \brief Convert a 1D root histogram to a Boost histogram
decltype(auto) boosthistoFromRoot_1D(TH1D* inHist1D)
{
  // first setup the proper boost histogram
  int nBins = inHist1D->GetNbinsX();
  int xMin = inHist1D->GetXaxis()->GetXmin();
  int xMax = inHist1D->GetXaxis()->GetXmax();
  const char* title = inHist1D->GetXaxis()->GetTitle();
  auto mHisto = boost::histogram::make_histogram(boost::histogram::axis::regular<>(nBins, xMin, xMax, title));

  // trasfer the acutal values
  for (Int_t x = 1; x < nBins + 1; x++) {
    mHisto.at(x) = inHist1D->GetBinContent(x);
  }
  return mHisto;
}

/// \brief Printing an error message when then fit returns an invalid result
/// \param errorcode Error of the type FitGausError_t, thrown when fit result is invalid.
std::string createErrorMessage(o2::utils::FitGausError_t errorcode){
  return "[Error]: Fit return an invalid result.";
}