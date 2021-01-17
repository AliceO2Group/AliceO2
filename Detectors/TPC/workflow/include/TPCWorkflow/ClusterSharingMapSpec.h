// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  ClusterSharingMapSpec.h
/// @brief Device to produce TPC clusters sharing map
/// \author ruben.shahoyan@cern.ch

#ifndef O2_TPC_CLUSTERSHARINGMAP_SPEC
#define O2_TPC_CLUSTERSHARINGMAP_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace tpc
{

class ClusterSharingMapSpec : public o2::framework::Task
{
 public:
  ~ClusterSharingMapSpec() override = default;
  void run(framework::ProcessingContext& pc) final;
};

o2::framework::DataProcessorSpec getClusterSharingMapSpec()
{

  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", "TPC", "CLUSREFS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusTPC", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("TPC", "CLSHAREDMAP", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{
    "tpc-clusters-sharing-map-producer",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<ClusterSharingMapSpec>()},
    o2::framework::Options{}};
}

} // namespace tpc
} // namespace o2

#endif
