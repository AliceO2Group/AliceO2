// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessorSpec.h"
#include "TRDWorkflowIO/TRDCalibReaderSpec.h"
#include "TRDWorkflow/VdAndExBCalibSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"enable-root-input", o2::framework::VariantType::Bool, false, {"enable root-files input readers"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto enableRootInp = configcontext.options().get<bool>("enable-root-input");
  WorkflowSpec specs;
  if (enableRootInp) {
    specs.emplace_back(o2::trd::getTRDCalibReaderSpec());
  }
  specs.emplace_back(getTRDVdAndExBCalibSpec());
  return specs;
}
