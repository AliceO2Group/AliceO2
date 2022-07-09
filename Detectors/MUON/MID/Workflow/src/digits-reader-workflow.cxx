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

/// \file   MID/Workflow/src/digits-reader-workflow.cxx
/// \brief  MID digits reader workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 October 2020

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"
#include "MIDWorkflow/DigitReaderSpec.h"
#include "MIDWorkflow/FilteringSpec.h"
#include "MIDWorkflow/ZeroSuppressionSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, false, {"Do not propagate MC info"}},
    {"disable-zero-suppression", VariantType::Bool, true, {"Do not apply zero suppression. Option is disabled since ZS cannot be applied right now (see ROBoradConfigHanler)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool disableZS = cfgc.options().get<bool>("disable-zero-suppression");
  disableZS = true; // Option is disabled
  bool useMC = !cfgc.options().get<bool>("disable-mc");

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  std::string dataDesc = disableZS ? "DATA" : "DATAMC";
  specs.emplace_back(o2::mid::getDigitReaderSpec(useMC, dataDesc.data()));
  if (!disableZS) {
    std::string outDesc = "MFDATA";
    specs.emplace_back(o2::mid::getFilteringSpec(useMC, dataDesc, outDesc));
    specs.emplace_back(o2::mid::getZeroSuppressionSpec(useMC, outDesc));
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return specs;
}
