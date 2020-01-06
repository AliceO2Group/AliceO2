// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <vector>

class TH1F;
class TH2F;

namespace o2
{
namespace tpc
{
namespace qc
{
namespace helpers
{

///Some utility functions for qc
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

} // namespace helpers
} // namespace qc
} // namespace tpc
} // namespace o2

#endif