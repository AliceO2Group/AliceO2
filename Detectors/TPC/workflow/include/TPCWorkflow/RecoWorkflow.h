// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RECOWORKFLOW_H
#define O2_TPC_RECOWORKFLOW_H
/// @file   RecoWorkflow.h
/// @author Matthias Richter
/// @since  2018-09-26
/// @brief  Workflow definition for the TPC reconstruction

#include "Framework/WorkflowSpec.h"
#include <vector>
#include <array>
#include <string>
#include <numeric> // std::iota

namespace o2
{
namespace framework
{
struct InputSpec;
}
namespace tpc
{

namespace reco_workflow
{
/// define input and output types of the workflow
enum struct InputType { Digitizer,        // directly read digits from channel {TPC:DIGITS}
                        Digits,           // read digits from file
                        ClustersHardware, // read hardware clusters in raw page format from file
                        Clusters,         // read native clusters from file
                        CompClusters,     // read compressed cluster container
                        CompClustersCTF,  // compressed clusters from CTF, as flat format
                        EncodedClusters,  // read encoded clusters
                        ZSRaw,
};

/// Output types of the workflow, workflow layout is built depending on configured types
/// - Digits           simulated digits
/// - ClustersHardware the first attempt of a raw format storing ClusterHardware in 8k pages
/// - Clusters         decoded clusters, ClusterNative, as input to the tracker
/// - Tracks           tracks
/// - CompClusters     compressed clusters, CompClusters container
/// - EncodedClusters  the encoded CompClusters container
/// - ZSRaw            TPC zero-suppressed raw data
enum struct OutputType { Digits,
                         ClustersHardware,
                         Clusters,
                         Tracks,
                         CompClusters,
                         EncodedClusters,
                         DisableWriter,
                         SendClustersPerSector,
                         ZSRaw,
                         QA,
                         NoSharedClusterMap,
};

using CompletionPolicyData = std::vector<framework::InputSpec>;

/// create the workflow for TPC reconstruction
framework::WorkflowSpec getWorkflow(CompletionPolicyData* policyData,             //
                                    std::vector<int> const& tpcSectors,           //
                                    std::vector<int> const& laneConfiguration,    //
                                    bool propagateMC = true, unsigned nLanes = 1, //
                                    std::string const& cfgInput = "digitizer",    //
                                    std::string const& cfgOutput = "tracks",      //
                                    int caClusterer = 0,                          //
                                    int zsOnTheFly = 0,
                                    int zs10bit = 0,
                                    float zsThreshold = 2.0f);

static inline framework::WorkflowSpec getWorkflow(CompletionPolicyData* policyData,             //
                                                  std::vector<int> const& tpcSectors,           //
                                                  bool propagateMC = true, unsigned nLanes = 1, //
                                                  std::string const& cfgInput = "digitizer",    //
                                                  std::string const& cfgOutput = "tracks",      //
                                                  int caClusterer = 0,                          //
                                                  int zsOnTheFly = 0,
                                                  int zs10bit = 0,
                                                  float zsThreshold = 2.0f)
{
  // create a default lane configuration with ids [0, nLanes-1]
  std::vector<int> laneConfiguration(nLanes);
  std::iota(laneConfiguration.begin(), laneConfiguration.end(), 0);
  return getWorkflow(policyData, tpcSectors, laneConfiguration, propagateMC, nLanes, cfgInput, cfgOutput, caClusterer, zsOnTheFly, zs10bit, zsThreshold);
}

static inline framework::WorkflowSpec getWorkflow(CompletionPolicyData* policyData,             //
                                                  bool propagateMC = true, unsigned nLanes = 1, //
                                                  std::string const& cfgInput = "digitizer",    //
                                                  std::string const& cfgOutput = "tracks",      //
                                                  int caClusterer = 0,                          //
                                                  int zsOnTheFly = 0,
                                                  int zs10bit = 0,
                                                  float zsThreshold = 2.0f)
{
  // create a default lane configuration with ids [0, nLanes-1]
  std::vector<int> laneConfiguration(nLanes);
  std::iota(laneConfiguration.begin(), laneConfiguration.end(), 0);
  return getWorkflow(policyData, {}, laneConfiguration, propagateMC, nLanes, cfgInput, cfgOutput, caClusterer, zsOnTheFly, zs10bit, zsThreshold);
}

} // end namespace reco_workflow
} // end namespace tpc
} // end namespace o2
#endif //O2_TPC_RECOWORKFLOW_H
