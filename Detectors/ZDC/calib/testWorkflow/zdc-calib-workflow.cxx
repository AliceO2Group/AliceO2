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

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb ZDC calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"do-lhc-phase", o2::framework::VariantType::Bool, true, {"do LHC clock phase calibration"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  auto doLHCcalib = configcontext.options().get<bool>("do-lhc-phase");

  LOG(info) << "ZDC Calibration workflow: options";
  LOG(info) << "useCCDB = " << useCCDB;
  return specs;
}
