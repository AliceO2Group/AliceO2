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
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"useDigits", VariantType::Bool, false, {"Use decoded digits"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "MFTCalibration/NoiseCalibratorSpec.h"
#include "MFTCalibration/NoiseCalibrator.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  auto useDigits = cfgc.options().get<bool>("useDigits");

  LOG(INFO) << "MFT calibration workflow options";
  LOG(INFO) << "Use Digits: " << useDigits;

  specs.emplace_back(o2::mft::getNoiseCalibratorSpec(useDigits));

  return specs;
}
