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
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/CosmicsMatchingSpec.h"
#include "GlobalTrackingWorkflow/TrackCosmicsWriterSpec.h"
#include "Algorithm/RangeTokenizer.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using GID = o2::dataformats::GlobalTrackID;
// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
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
  GID::mask_t alowedSources = GID::getSourcesMask("ITS,TPC,ITS-TPC,TPC-TOF,ITS-TPC-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2match-cosmics-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");

  std::vector<int> tpcClusSectors = o2::RangeTokenizer::tokenize<int>("0-35");
  std::vector<int> tpcClusLanes = tpcClusSectors;

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));

  if (!disableRootInp) {

    if (src[GID::ITS]) {
      specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
    }

    if (src[GID::TPC]) {
      specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));
    }

    if (src[GID::ITSTPC] || src[GID::ITSTPCTOF]) { // ITSTPCTOF does not provide tracks, only matchInfo
      specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    }

    if (src[GID::ITSTPCTOF]) {
      specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(useMC, false, false)); // MC, MatchInfo_glo, no TOF_TPCtracks
      specs.emplace_back(o2::tof::getClusterReaderSpec(false));                  // RSTODO Needed just to set the time of ITSTPC track, consider moving to MatchInfoTOF
    }

    if (src[GID::TPCTOF]) {
      specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(useMC, true, true)); // mc, MatchInfo_TPC, TOF_TPCtracks
    }

    // clusters for refit
    if (GID::includesDet(DetID::ITS, src)) {
      specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(false, true)); // mc not neaded
    }
    if (GID::includesDet(DetID::TPC, src)) {
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
  }

  specs.emplace_back(o2::globaltracking::getCosmicsMatchingSpec(src, useMC));

  if (!disableRootOut) {
    specs.emplace_back(o2::globaltracking::getTrackCosmicsWriterSpec(useMC));
  }

  return specs;
}
