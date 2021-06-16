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
#ifndef O2_FRAMEWORK_ROOTANALYSISHELPERS_H_
#define O2_FRAMEWORK_ROOTANALYSISHELPERS_H_

#include "Framework/TableConsumer.h"
#include <ROOT/RDataFrame.hxx>

namespace o2::analysis
{

/// Do a single loop on all the entries of the @a input table
ROOT::RDataFrame doSingleLoopOn(std::unique_ptr<framework::TableConsumer>& input);

/// Do a double loop on all the entries with the same value for the \a grouping
/// of the @a input table, where the entries for the outer index are prefixed
/// with `<name>_` while the entries for the inner loop are prefixed with
/// `<name>bar_`.
ROOT::RDataFrame doSelfCombinationsWith(std::unique_ptr<framework::TableConsumer>& input,
                                        std::string name = "p",
                                        std::string grouping = "eventID");

} // namespace o2::analysis

#endif // O2_FRAMEWORK_ROOTANALYSISHELPERS_H_
