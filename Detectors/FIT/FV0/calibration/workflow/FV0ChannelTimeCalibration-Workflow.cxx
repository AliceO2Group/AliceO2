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

#include "FV0ChannelTimeCalibrationSpec.h"
#include "FV0Calibration/FV0ChannelTimeOffsetSlotContainer.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{

  // probably some option will be added
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"time-calib-fitting-nbins", VariantType::Int, 2, {""}});
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

using namespace o2::framework;
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  o2::fv0::FV0ChannelTimeOffsetSlotContainer::sGausFitBins = config.options().get<int>("time-calib-fitting-nbins");
  workflow.emplace_back(o2::fv0::getFV0ChannelTimeCalibrationSpec());
  return workflow;
}
