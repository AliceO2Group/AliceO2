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
/// @file   Helpers.h
/// @author Thomas Klemenz, thomas.klemenz@tum.de
///

#ifndef AliceO2_TPC_HELPERS_H
#define AliceO2_TPC_HELPERS_H

// root includes
#include "TH1F.h"
#include "TH2F.h"

#include <vector>
#include <string>
#include "TPCBase/CalDet.h"

namespace o2
{
namespace tpc
{
namespace qc
{
namespace helpers
{

/// Some utility functions for qc
///

/// get a vector containing binning info for constant sized bins on a log axis
std::vector<double> makeLogBinning(const int nbins, const double min, const double max);

/// Set nice style for single 1D histograms
void setStyleHistogram1D(TH1& histo);

/// Set nice style for vector of 1D histograms
void setStyleHistogram1D(std::vector<TH1F>& histos);

/// Set nice style for 2D histograms
void setStyleHistogram2D(TH2& histo);

/// Set nice style for vector of 2D histograms
void setStyleHistogram2D(std::vector<TH2F>& histos);

/// set nice style for 1D histograms ptr
void setStyleHistogram(TH1& histo);

// set nice style of histograms in a map of vectors
void setStyleHistogramsInMap(std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& mapOfvectors);
// set nice style of histograms in a map
void setStyleHistogramsInMap(std::unordered_map<std::string_view, std::unique_ptr<TH1>>& mapOfHisto);
// set nice style of histograms in a map of vectors
void setStyleHistogramsInMap(std::unordered_map<std::string, std::vector<std::unique_ptr<TH1>>>& mapOfvectors);
// set nice style of histograms in a map
void setStyleHistogramsInMap(std::unordered_map<std::string, std::unique_ptr<TH1>>& mapOfHisto);
/// Check if at least one pad in refPedestal and pedestal differs by 3*refNoise to see if new ZS calibration data should be uploaded to the FECs.
/// @param refPedestal
/// @param refNoise
/// @param pedestal
/// @return true if refPedestal - pedestal > 3*refNoise on at least one pad
bool newZSCalib(const o2::tpc::CalDet<float>& refPedestal, const o2::tpc::CalDet<float>& refNoise, const o2::tpc::CalDet<float>& pedestal);

} // namespace helpers
} // namespace qc
} // namespace tpc
} // namespace o2

#endif
