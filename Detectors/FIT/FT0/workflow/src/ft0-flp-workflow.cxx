// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0Workflow/FT0Workflow.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{
    "tcm-extended-mode", o2::framework::VariantType::Bool, false, {"in case of extended TCM mode (1 header + 1 TCMdata + 8 TCMdataExtended)"}});

  workflowOptions.push_back(ConfigParamSpec{
    "use-process", o2::framework::VariantType::Bool, false, {"enable processor for data taking/dumping"}});
  workflowOptions.push_back(ConfigParamSpec{
    "dump-blocks-process", o2::framework::VariantType::Bool, false, {"enable dumping of event blocks at processor side"}});
  workflowOptions.push_back(ConfigParamSpec{
    "dump-blocks-reader", o2::framework::VariantType::Bool, false, {"enable dumping of event blocks at reader side"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(INFO) << "WorkflowSpec defineDataProcessing";
  auto useProcessor = configcontext.options().get<bool>("use-process");
  auto dumpProcessor = configcontext.options().get<bool>("dump-blocks-process");
  auto dumpReader = configcontext.options().get<bool>("dump-blocks-reader");
  auto isExtendedMode = configcontext.options().get<bool>("tcm-extended-mode");

  LOG(INFO) << "WorkflowSpec FLPWorkflow";
  return std::move(o2::fit::getFT0Workflow(isExtendedMode, useProcessor, dumpProcessor, dumpReader));
}
