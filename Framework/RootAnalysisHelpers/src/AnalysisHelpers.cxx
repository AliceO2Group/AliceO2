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
#include "Framework/AnalysisHelpers.h"
#include "Framework/RCombinedDS.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableConsumer.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>

using namespace ROOT::RDF;

namespace o2
{
namespace analysis
{

ROOT::RDataFrame doSingleLoopOn(std::unique_ptr<framework::TableConsumer>& input)
{
  auto flat = std::make_unique<RArrowDS>(input->asArrowTable(), std::vector<std::string>{});
  ROOT::RDataFrame rdf(std::move(flat));
  return rdf;
}

ROOT::RDataFrame doSelfCombinationsWith(std::unique_ptr<framework::TableConsumer>& input, std::string name, std::string grouping)
{
  auto table = input->asArrowTable();
  using Index = RCombinedDSBlockJoinIndex<int>;
  auto left = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
  auto right = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
  auto combined = std::make_unique<RCombinedDS>(std::move(left), std::move(right), std::make_unique<Index>(grouping, true, BlockCombinationRule::StrictlyUpper), name + "_", name + "bar_");

  ROOT::RDataFrame rdf(std::move(combined));
  return rdf;
}

} // namespace analysis
} // namespace o2
