// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "MCTruthSourceSpec.h"
#include "MCTruthWriterSpec.h"
#include "MCTruthReaderSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option to disable MC truth
  workflowOptions.push_back(ConfigParamSpec{"newmctruth", o2::framework::VariantType::Bool, false, {"enable new container"}});
  workflowOptions.push_back(ConfigParamSpec{"consumers", o2::framework::VariantType::Int, 1, {"number of mc consumers"}});
}

#include "Framework/runDataProcessing.h"

/// This function is required to be implemented to define the workflow
/// specifications
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  bool newmctruth = configcontext.options().get<bool>("newmctruth");

  // connect the source
  specs.emplace_back(o2::getMCTruthSourceSpec(newmctruth));
  // connect some consumers
  for (int i = 0; i < configcontext.options().get<int>("consumers"); ++i) {
    specs.emplace_back(o2::getMCTruthWriterSpec(i, i == 0, newmctruth));
  }
  // connect a device reading back the labels
  specs.emplace_back(o2::getMCTruthReaderSpec(newmctruth));
  // connect a device reading back the labels
  specs.emplace_back(o2::getMCTruthWriterSpec(-1, false, newmctruth));

  return specs;
}
