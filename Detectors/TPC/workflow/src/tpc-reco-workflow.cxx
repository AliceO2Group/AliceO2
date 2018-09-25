// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tpc-reco-workflow.cxx
/// @author Matthias Richter
/// @since  2018-03-15
/// @brief  Basic DPL workflow for TPC reconstruction starting from digits

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "TPCWorkflow/RecoWorkflow.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    { "input-type", o2::framework::VariantType::String, "digits", { "digits, clusters, raw" } },
    { "output-type", o2::framework::VariantType::String, "tracks", { "clusters, raw, tracks" } },
    { "disable-mc", o2::framework::VariantType::Bool, false, { "disable sending of MC information" } },
    { "tpc-lanes", o2::framework::VariantType::Int, 1, { "number of parallel lanes up to the tracker" } },
  };
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

/// Defines basic workflow for TPC reconstruction
/// - digit reader
/// - clusterer
/// - cluster converter
/// - cluster raw decoder
/// - CA tracker
///
/// Digit reader and clusterer can be replaced by the cluster reader.
///
/// MC info is always sent by the digit reader and clusterer processes, the
/// cluster converter process creating the raw format can be configured to forward MC.
///
/// This function is required to be implemented to define the workflow specifications
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return std::move(o2::TPC::RecoWorkflow::getWorkflow(not cfgc.options().get<bool>("disable-mc"),    //
                                                      cfgc.options().get<int>("tpc-lanes"),          //
                                                      cfgc.options().get<std::string>("input-type"), //
                                                      cfgc.options().get<std::string>("output-type") //
                                                      ));
}
