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

/// \file tracks-writer-workflow.cxx
/// \brief Implementation of a DPL device to run the MCH-MID track writer
///
/// \author Philippe Pillot, Subatech

#include "TrackWriterSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MUON|muon).*[W,w]riter.*"));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.emplace_back("disable-mc", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"disable MC propagation even if available"});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  return WorkflowSpec{o2::muon::getTrackWriterSpec(useMC)};
}
