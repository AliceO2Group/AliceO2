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

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "EMCALWorkflow/CellRecalibratorSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;
using namespace o2::emcal;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"input-subspec", VariantType::UInt32, 1U, {"Subspecification for input objects"}},
                                       {"output-subspec", VariantType::UInt32, 0U, {"Subspecification for output objects"}},
                                       {"no-badchannelcalib", VariantType::Bool, false, {"Disable bad channel calibration"}},
                                       {"no-timecalib", VariantType::Bool, false, {"Disable time calibration"}},
                                       {"no-gaincalib", VariantType::Bool, false, {"Disable gain calibration"}},
                                       {"local-badchannelmap", VariantType::String, "", {"path to local file with bad channel map"}},
                                       {"local-timecalib", VariantType::String, "", {"path to local file with time calibration params"}},
                                       {"local-gaincalib", VariantType::String, "", {"path to local file with gain calibration params"}},
                                       {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  // Calibration types
  auto disableBadchannels = cfgc.options().get<bool>("no-badchannelcalib"),
       disableTime = cfgc.options().get<bool>("no-timecalib"),
       disableEnergy = cfgc.options().get<bool>("no-gaincalib");

  // subpsecs for input and output
  auto inputsubspec = cfgc.options().get<uint32_t>("input-subspec"),
       outputsubspec = cfgc.options().get<uint32_t>("output-subspec");

  std::string pathBadChannelMap = cfgc.options().get<std::string>("local-badchannelmap"),
              pathTimeCalib = cfgc.options().get<std::string>("local-timecalib"),
              pathGainCalib = cfgc.options().get<std::string>("local-gaincalib");

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  specs.emplace_back(o2::emcal::getCellRecalibratorSpec(inputsubspec, outputsubspec, !disableBadchannels, !disableTime, !disableEnergy, pathBadChannelMap, pathTimeCalib, pathGainCalib));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);
  return specs;
}