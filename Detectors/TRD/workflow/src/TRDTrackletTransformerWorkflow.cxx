// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDWorkflow/TRDTrackletTransformerSpec.h"
#include "TRDWorkflowIO/TRDCalibratedTrackletWriterSpec.h"
#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{
    "root-in", VariantType::Int, 1, {"enable (1) or disable (0) input from ROOT file"}});
  workflowOptions.push_back(ConfigParamSpec{
    "root-out", VariantType::Int, 1, {"enable (1) or disable (0) output to ROOT file"}});
  workflowOptions.push_back(
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  int rootIn = configcontext.options().get<int>("root-in");
  int rootOut = configcontext.options().get<int>("root-out");
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  WorkflowSpec spec;

  if (rootIn) {
    spec.emplace_back(o2::trd::getTRDTrackletReaderSpec(false, false));
  }

  spec.emplace_back(o2::trd::getTRDTrackletTransformerSpec());

  if (rootOut) {
    spec.emplace_back(o2::trd::getTRDCalibratedTrackletWriterSpec());
  }

  return spec;
}
