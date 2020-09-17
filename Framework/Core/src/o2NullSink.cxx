// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  options.push_back(o2::framework::ConfigParamSpec{"inputs", VariantType::String, "", {"inputs for the dataprocessor"}});
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
  auto inputsDesc = context.options().get<std::string>("inputs");
  auto inputs = DataDescriptorQueryBuilder::parse(inputsDesc.c_str());

  workflow.push_back(DataProcessorSpec{
    "null",
    inputs,
    {},
    AlgorithmSpec{[](ProcessingContext& ctx) {}}});

  return workflow;
}
