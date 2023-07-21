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

namespace o2
{
namespace its
{
namespace studies
{
namespace helpers
{

/// Some utility functions for postprocessing ITS
///

/// get a vector containing binning info for constant sized bins on a log axis
std::vector<double> makeLogBinning(const int nbins, const double min, const double max);
/*
/// Set nice style for single 1D histograms
void setStyleHistogram1D(TH1F& histo);

/// Set nice style for vector of 1D histograms
void setStyleHistogram1D(std::vector<TH1F>& histos);

/// Set nice style for 2D histograms
void setStyleHistogram2D(TH2F& histo);

/// Set nice style for vector of 2D histograms
void setStyleHistogram2D(std::vector<TH2F>& histos);

/// set nice style for 1D histograms ptr
void setStyleHistogram(TH1F& histo);

 // set nice style of histograms in a map of vectors
void setStyleHistogramsInMap(std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& mapOfvectors);
// set nice style of histograms in a map
void setStyleHistogramsInMap(std::unordered_map<std::string_view, std::unique_ptr<TH1>>& mapOfHisto); */

} // namespace helpers
} // namespace studies
} // namespace its
} // namespace o2

#endif