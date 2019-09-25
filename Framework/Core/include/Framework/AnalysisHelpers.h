// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_AnalysisHelpers_H_DEFINED
#define o2_framework_AnalysisHelpers_H_DEFINED

#include "Framework/AnalysisHelpers.h"

#include <ROOT/RDataFrame.hxx>
#include <string>

using namespace ROOT::RDF;

namespace o2
{
namespace framework
{
class TableConsumer;
}

namespace analysis
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

} // namespace analysis
} // namespace o2

#endif // o2_framework_AnalysisHelpers_H_DEFINED
