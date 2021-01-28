// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TrackInterpolationWorkflow.cxx

#include <vector>

#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/PublisherSpec.h"
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "TOFWorkflowUtils/ClusterReaderSpec.h"
#include "TOFWorkflow/TOFMatchedReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "TPCInterpolationWorkflow/TPCResidualWriterSpec.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"
#include "TPCInterpolationWorkflow/TrackInterpolationWorkflow.h"

namespace o2
{
namespace tpc
{

framework::WorkflowSpec getTPCInterpolationWorkflow(bool disableRootInp, bool disableRootOut)
{
  framework::WorkflowSpec specs;
  bool useMC = false;
  std::vector<int> tpcClusSectors = o2::RangeTokenizer::tokenize<int>("0-35");
  std::vector<int> tpcClusLanes = tpcClusSectors;
  if (!disableRootInp) {
    specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
    specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));
    specs.emplace_back(o2::tpc::getPublisherSpec(o2::tpc::PublisherConf{
                                                   "tpc-native-cluster-reader",
                                                   "tpc-native-clusters.root",
                                                   "tpcrec",
                                                   {"clusterbranch", "TPCClusterNative", "Branch with TPC native clusters"},
                                                   {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                                   OutputSpec{"TPC", "CLUSTERNATIVE"},
                                                   OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                                   tpcClusSectors,
                                                   tpcClusLanes},
                                                 useMC));
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    specs.emplace_back(o2::tof::getClusterReaderSpec(useMC));
    specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(useMC));
  }

  specs.emplace_back(o2::tpc::getTPCInterpolationSpec(useMC, tpcClusLanes));

  if (!disableRootOut) {
    specs.emplace_back(o2::tpc::getTPCResidualWriterSpec(useMC));
  }
  return specs;
}

} // namespace tpc
} // namespace o2
