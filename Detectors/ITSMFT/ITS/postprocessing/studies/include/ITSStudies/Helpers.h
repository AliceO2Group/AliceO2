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

///
/// @author Antonio Palasciano, antonio.palasciano@ba.infn.it
///

#ifndef O2_ITS_STUDY_HELPERS_H
#define O2_ITS_STUDY_HELPERS_H

#include <vector>
#include "TH1F.h"
#include "TH2F.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TPaveText.h"

namespace o2
{
namespace its
{
namespace study
{
namespace helpers
{

/// Some utility functions for postprocessing ITS
///

/// get a vector containing binning info for constant sized bins on a log axis
std::vector<double> makeLogBinning(const int nbins, const double min, const double max);

/// Set nice style for single 1D histograms
void setStyleHistogram1D(TH1F& histo, int color);
void setStyleHistogram1D(TH1F& histo, int color, TString title, TString titleYaxis, TString titleXaxis);
void setStyleHistogram1DMeanValues(TH1F& histo, int color);
void setStyleHistogram2D(TH2F& histo);

/// prepare canvas with two TH1F plots
TCanvas* prepareSimpleCanvas2Histograms(TH1F& h1, int color1, TH1F& h2, int color2);
TCanvas* prepareSimpleCanvas2Histograms(TH1F& h1, int color1, TString nameHisto1, TH1F& h2, int color2, TString nameHisto2, bool logScale = true);
TCanvas* prepareSimpleCanvas2Histograms(TH1F& h1, int color1, TString nameHisto1, TH1F& h2, int color2, TString nameHisto2, TString intRate);
TCanvas* prepareSimpleCanvas2DcaMeanValues(TH1F& h1, int color1, TString nameHisto1, TH1F& h2, int color2, TString nameHisto2);

/// plot canvas with TH2D + TH1D(Mean and Sigma) from slice
TCanvas* plot2DwithMeanAndSigma(TH2F& h2D, TH1F& hMean, TH1F& hSigma, int color);

/// prepare TPaveText with labels
void paveTextITS(TPaveText* pave, TString intRate);

/// Convert TH1F in TGraphAsymmetricError
void ConvertTH1ToTGraphAsymmError(TH1F& hMean, TH1F& hSigma, TGraphAsymmErrors*& gr);

} // namespace helpers
} // namespace study
} // namespace its
} // namespace o2

#endif