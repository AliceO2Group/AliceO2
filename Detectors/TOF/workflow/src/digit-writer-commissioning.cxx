// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFWorkflowIO/TOFDigitWriterSplitterSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"ntf", o2::framework::VariantType::Int, 1, {"number of timeframe written for output file"}});
  workflowOptions.push_back(ConfigParamSpec{"write-decoding-errors", o2::framework::VariantType::Bool, false, {"trace errors in digits output when decoding"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  // Update the (declared) parameters if changed from the command line
  auto ntf = cfgc.options().get<int>("ntf");
  auto write_err = cfgc.options().get<bool>("write-decoding-errors");
  wf.emplace_back(o2::framework::getTOFDigitWriterSplitterSpec(ntf, write_err));
  return wf;
}
