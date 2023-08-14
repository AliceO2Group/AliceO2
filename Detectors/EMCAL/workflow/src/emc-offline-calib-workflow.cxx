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

#include "EMCALWorkflow/OfflineCalibSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{{"makeCellIDTimeEnergy", VariantType::Bool, false, {"list whether or not to make the cell ID, time, energy THnSparse"}},
                                       {"no-rejectCalibTrigg", VariantType::Bool, false, {"if set to true, all events, including calibration triggered events, will be accepted"}},
                                       {"input-subspec", VariantType::UInt32, 0U, {"Subspecification for input objects"}},
                                       {"applyGainCalib", VariantType::Bool, false, {"Apply the gain calibration parameters for the bad channel calibration"}},
                                       {"rejectL0Trigger", VariantType::Bool, false, {"Reject all emcal triggers except the minimum bias trigger"}},
                                       {"ctpconfig-run-independent", VariantType::Bool, false, {"Use CTP config w/o runNumber tag"}},
                                       {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  // Update the (declared) parameters if changed from the command line
  bool makeCellIDTimeEnergy = cfgc.options().get<bool>("makeCellIDTimeEnergy");
  bool rejectCalibTrigg = !cfgc.options().get<bool>("no-rejectCalibTrigg");
  bool doApplyGainCalib = cfgc.options().get<bool>("applyGainCalib");
  bool doRejectL0Trigger = cfgc.options().get<bool>("rejectL0Trigger");
  bool ctpcfgperrun = !cfgc.options().get<bool>("ctpconfig-run-independent");

  // subpsecs for input
  auto inputsubspec = cfgc.options().get<uint32_t>("input-subspec");

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  wf.emplace_back(o2::emcal::getEmcalOfflineCalibSpec(makeCellIDTimeEnergy, rejectCalibTrigg, inputsubspec, doApplyGainCalib, doRejectL0Trigger, ctpcfgperrun));
  return wf;
}
