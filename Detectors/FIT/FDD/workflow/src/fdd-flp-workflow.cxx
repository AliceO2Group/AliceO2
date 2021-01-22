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
#include "FDDWorkflow/RawWorkflow.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(
    ConfigParamSpec{"use-process",
                    o2::framework::VariantType::Bool,
                    false,
                    {"enable processor for data taking/dumping"}});
  workflowOptions.push_back(
    ConfigParamSpec{"dump-blocks-process",
                    o2::framework::VariantType::Bool,
                    false,
                    {"enable dumping of event blocks at processor side"}});
  workflowOptions.push_back(
    ConfigParamSpec{"dump-blocks-reader",
                    o2::framework::VariantType::Bool,
                    false,
                    {"enable dumping of event blocks at reader side"}});
  workflowOptions.push_back(
    ConfigParamSpec{"disable-root-output",
                    o2::framework::VariantType::Bool,
                    false,
                    {"disable root-files output writers"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(INFO) << "WorkflowSpec defineDataProcessing";
  auto useProcessor = configcontext.options().get<bool>("use-process");
  auto dumpProcessor = configcontext.options().get<bool>("dump-blocks-process");
  auto dumpReader = configcontext.options().get<bool>("dump-blocks-reader");
  auto disableRootOut =
    configcontext.options().get<bool>("disable-root-output");
  LOG(INFO) << "WorkflowSpec FLPWorkflow";
  return std::move(o2::fdd::getFDDRawWorkflow(
    useProcessor, dumpProcessor, dumpReader, disableRootOut));
}
