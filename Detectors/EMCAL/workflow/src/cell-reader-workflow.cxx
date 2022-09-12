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
#include "DataFormatsEMCAL/Cell.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "EMCALWorkflow/PublisherSpec.h"
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
  std::vector<ConfigParamSpec> options{{"disable-mc", VariantType::Bool, false, {"Do not propagate MC labels"}},
                                       {"subspec", VariantType::UInt32, 0U, {"Subspecification for cell output"}},
                                       {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool disableMC = cfgc.options().get<bool>("disable-mc");
  auto subspec = cfgc.options().get<uint32_t>("subspec");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  LOG(info) << "Cell reader: publishing on subspec " << subspec << std::endl;

  WorkflowSpec specs;
  specs.emplace_back(o2::emcal::getPublisherSpec<std::vector<o2::emcal::Cell>>(PublisherConf{
                                                                                 "emcal-cell-reader",
                                                                                 "o2sim",
                                                                                 "emccells.root",
                                                                                 {"cellbranch", "EMCALCell", "Cell branch"},
                                                                                 {"celltriggerbranch", "EMCALCellTRGR", "Trigger record branch"},
                                                                                 {"mcbranch", "EMCALCellMCTruth", "MC label branch"},
                                                                                 o2::framework::OutputSpec{{"cells"}, "EMC", "CELLS", subspec, o2::framework::Lifetime::Timeframe},
                                                                                 o2::framework::OutputSpec{{"triggerrecords"}, "EMC", "CELLSTRGR", subspec, o2::framework::Lifetime::Timeframe},
                                                                                 o2::framework::OutputSpec{{"mclabels"}, "EMC", "CELLSMCTR", subspec, o2::framework::Lifetime::Timeframe}},
                                                                               subspec, !disableMC));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);
  return specs;
}
