// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   emc-reco-workflow.cxx
/// @author Markus Fasel
/// @since  2019-06-07
/// @brief  Basic DPL workflow for EMCAL reconstruction starting from digits (adapted from tpc-reco-workflow.cxx)

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "EMCALWorkflow/RecoWorkflow.h"
#include "Algorithm/RangeTokenizer.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    { "input-type", o2::framework::VariantType::String, "digits", { "digitizer, digits, raw, clusters" } },
    { "output-type", o2::framework::VariantType::String, "digits", { "digits, raw, clusters" } },
    { "disable-mc", o2::framework::VariantType::Bool, false, { "disable sending of MC information" } },
  };
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

/// The workflow executable for the stand alone EMCAL reconstruction workflow
/// The basic workflow for EMCAL reconstruction is defined in RecoWorkflow.cxx
/// and contains the following default processors
/// - digit reader
/// - clusterer
///
/// The default workflow can be customized by specifying input and output types
/// e.g. digits, raw, tracks.
///
/// MC info is processed by default, disabled by using command line option `--disable-mc`
///
/// This function hooks up the the workflow specifications into the DPL driver.
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& cfgc)
{
  //bla
  return o2::emcal::reco_workflow::getWorkflow(not cfgc.options().get<bool>("disable-mc"),    //
                                               cfgc.options().get<std::string>("input-type"), //
                                               cfgc.options().get<std::string>("output-type") //
  );
}
