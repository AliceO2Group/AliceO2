// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  MatchTPCITSWorkflow.cxx

#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/PublisherSpec.h"
#include "FITWorkflow/FT0RecPointReaderSpec.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "GlobalTrackingWorkflow/MatchTPCITSWorkflow.h"
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DataFormatsTPC/Constants.h"

namespace o2
{
namespace globaltracking
{

framework::WorkflowSpec getMatchTPCITSWorkflow(bool useMC, bool useFIT)
{
  framework::WorkflowSpec specs;

  bool passFullITSClusters = true;  // temporarily pass full clusters,
  bool passCompITSClusters = false; // eventually only compact of recpoints will be passed

  specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
  specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(useMC, passFullITSClusters, passCompITSClusters));

  specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));

  std::vector<int> tpcClusSectors = o2::RangeTokenizer::tokenize<int>("0-35");
  std::vector<int> tpcClusLanes = tpcClusSectors;
  specs.emplace_back(o2::tpc::getPublisherSpec(o2::tpc::PublisherConf{
                                                 "tpc-native-cluster-reader",
                                                 "tpcrec",
                                                 {"clusterbranch", "TPCClusterNative", "Branch with TPC native clusters"},
                                                 {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                                 OutputSpec{"TPC", "CLUSTERNATIVE"},
                                                 OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                                 tpcClusSectors,
                                                 tpcClusLanes},
                                               useMC));

  specs.emplace_back(o2::globaltracking::getTPCITSMatchingSpec(useMC, useFIT, tpcClusLanes));
  specs.emplace_back(o2::globaltracking::getTrackWriterTPCITSSpec(useMC));

  if (useFIT) {
    specs.emplace_back(o2::ft0::getFT0RecPointReaderSpec(useMC));
  }

  return specs;
}

} // namespace globaltracking
} // namespace o2
