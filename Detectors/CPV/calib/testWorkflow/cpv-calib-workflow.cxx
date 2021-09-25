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
#include "PedestalCalibratorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"pedestals", o2::framework::VariantType::Bool, true, {"do pedestal calibration"}});
  //workflowOptions.push_back(ConfigParamSpec{"gains", o2::framework::VariantType::Bool, false, {"do gain calibration"}});
  //workflowOptions.push_back(ConfigParamSpec{"badmap", o2::framework::VariantType::Bool, false, {"do bad map calibration"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  auto doPedestals = configcontext.options().get<bool>("pedestals");
  //auto doGain = configcontext.options().get<bool>("gains");
  //auto doBadMap = configcontext.options().get<bool>("badmap");
  //if (doPedestals && doGain) {
  //  LOG(FATAL) << "Can not run pedestal and gain calibration simulteneously";
  //  return specs;
  //}
  //if (doGain) {
  //  specs.emplace_back(getCPVGainCalibDeviceSpec());
  //}
  //if (doBadMap) {
  //  specs.emplace_back(getCPVBadMapCalibDeviceSpec());
  //}

  if (doPedestals) {
    specs.emplace_back(getCPVPedestalCalibDeviceSpec());
  }
  return specs;
}
