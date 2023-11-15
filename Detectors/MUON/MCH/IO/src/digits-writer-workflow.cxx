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

/**
 * o2-digits-writer-workflow writes the digits received, in Root format
 */

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include "MCHIO/DigitWriterSpec.h"

using namespace o2::framework;

void customize(std::vector<CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MCH|mch).*[W,w]riter.*"));
}

/**
 * Add workflow options. Note that customization needs to be declared
 * before including Framework/runDataProcessing.
 */
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(ConfigParamSpec{"input-digits-data-description", VariantType::String, "DIGITS", {"description string for the input digits data"}});
  workflowOptions.emplace_back(ConfigParamSpec{"input-digitrofs-data-description", VariantType::String, "DIGITROFS", {"description string for the input digit rofs data"}});
  workflowOptions.emplace_back(ConfigParamSpec{"enable-mc", VariantType::Bool, false, {" output MC labels if available "}});
  workflowOptions.emplace_back(ConfigParamSpec{"mch-digit-outfile", VariantType::String, "mchdigits.root", {"name of digit root file"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  bool useMC = cc.options().get<bool>("enable-mc");
  auto name = fmt::format("mch-digit-writer-{}-{}",
                          cc.options().get<std::string>("input-digits-data-description"),
                          cc.options().get<std::string>("input-digitrofs-data-description"));

  return WorkflowSpec{o2::mch::getDigitWriterSpec(
    useMC,
    name,
    cc.options().get<std::string>("mch-digit-outfile"),
    cc.options().get<std::string>("input-digits-data-description"),
    cc.options().get<std::string>("input-digitrofs-data-description"))};
}
