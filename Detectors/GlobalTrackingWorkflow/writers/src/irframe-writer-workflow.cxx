// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"input-spec", o2::framework::VariantType::String, "X:NIL/IRFRAMES/0", {"input spec"}});
  workflowOptions.push_back(ConfigParamSpec{"device-name", o2::framework::VariantType::String, "irframe-writer", {"device name"}});
}

#include "Framework/runDataProcessing.h"
#include "GlobalTrackingWorkflowWriters/IRFrameWriterSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cc)
{
  WorkflowSpec specs;
  if (cc.helpOnCommandLine()) {
    return specs;
  }
  specs.emplace_back(o2::globaltracking::getIRFrameWriterSpec(cc.options().get<std::string>("input-spec"), "o2_irframe.root",
                                                              cc.options().get<std::string>("device-name")));
  return specs;
}
