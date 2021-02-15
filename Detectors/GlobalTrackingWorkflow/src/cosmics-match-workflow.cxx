// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "TPCWorkflow/TPCSectorCompletionPolicy.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/PublisherSpec.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "TOFWorkflowUtils/ClusterReaderSpec.h"
#include "TOFWorkflow/TOFMatchedReaderSpec.h"
#include "TOFWorkflowUtils/ClusterReaderSpec.h"
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/CosmicsMatchingSpec.h"
#include "Algorithm/RangeTokenizer.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"onlyDet", VariantType::String, std::string{DetID::NONE}, {"comma-separated list of detectors to use. Overrides skipDet"}},
    {"skipDet", VariantType::String, std::string{DetID::NONE}, {"comma-separate list of detectors to skip"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

// RS FIXME Is this needed ?
// the matcher process requires the TPC sector completion to trigger and data on
// all defined routes
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // the TPC sector completion policy checks when the set of TPC/CLUSTERNATIVE data is complete
  // in addition we require to have input from all other routes
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("itstpc-track-matcher",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  DetID::mask_t dets;
  dets.set(); // by default read all

  if (!configcontext.helpOnCommandLine()) {
    auto mskOnly = DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
    auto mskSkip = DetID::getMask(configcontext.options().get<std::string>("skipDet"));
    if (mskOnly.any()) {
      dets &= mskOnly;
    } else {
      dets ^= mskSkip;
    }
  }

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2match-cosmics-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");

  std::vector<int> tpcClusSectors = o2::RangeTokenizer::tokenize<int>("0-35");
  std::vector<int> tpcClusLanes = tpcClusSectors;

  if (!disableRootInp) {
    // ITS tracks and clusters
    if (dets[DetID::ITS]) {
      specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
      specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(false, true)); // mc not neaded
    }

    // TPC tracks and clusters
    if (dets[DetID::TPC]) {
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
                                                   false));
      specs.emplace_back(o2::tpc::getClusterSharingMapSpec());
    }

    // ITS-TPC matches
    if (dets[DetID::ITS] && dets[DetID::TPC]) {
      specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(true));
    }

    if (dets[DetID::TPC] && dets[DetID::TOF]) {
      // TPC-TOF matches
      specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(true, true, true)); // MC info here is redundant
      // TOF clusters
      specs.emplace_back(o2::tof::getClusterReaderSpec(false));
      // ITS-TPC-TOF matches
      if (dets[DetID::ITS]) {
        specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(true, false, false)); // MC info here is redundant
      }
    }
  }

  specs.emplace_back(o2::globaltracking::getCosmicsMatchingSpec(dets, useMC));

  if (!disableRootOut) {
  }

  return specs;
}
