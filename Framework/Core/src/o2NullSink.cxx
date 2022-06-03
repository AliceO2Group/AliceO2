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
#include "Framework/ConfigContext.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/ParallelContext.h"
#include "Framework/DataDescriptorQueryBuilder.h"

#include <chrono>
#include <thread>
#include <vector>

/// A DataProcessor which terminates a provided set of
/// inputs.
using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back(o2::framework::ConfigParamSpec{"name", VariantType::String, "null", {"name for the dataprocessor"}});
  options.push_back(o2::framework::ConfigParamSpec{"dataspec", VariantType::String, "", {"inputs for the dataprocessor"}});
};

#include "Framework/runDataProcessing.h"

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  WorkflowSpec workflow;
  // This is an example of how we can parallelize by subSpec.
  // templatedProducer will be instanciated 32 times and the lambda function
  // passed to the parallel statement will be applied to each one of the
  // instances in order to modify it. Parallel will also make sure the name of
  // the instance is amended from "some-producer" to "some-producer-<index>".
  auto name = context.options().get<std::string>("name");
  auto inputsDesc = context.options().get<std::string>("dataspec");
  auto inputs = DataDescriptorQueryBuilder::parse(inputsDesc.c_str());

  workflow.push_back(DataProcessorSpec{
    name,
    inputs,
    {},
    AlgorithmSpec{[](ProcessingContext& ctx) {}}});

  return workflow;
}
