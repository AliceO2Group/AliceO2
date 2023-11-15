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
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"

#include <chrono>
#include <thread>
#include <vector>
#include <fairmq/Device.h>

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(
    ConfigParamSpec{"dataspec", VariantType::String, "", {"DataSpec for the outputs"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"name", VariantType::String, "test-sink", {"Name of the source"}});
}
#include "Framework/runDataProcessing.h"

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{adaptStateful([what, minDelay]() {
    srand(getpid());
    return adaptStateless([what, minDelay](DataAllocator& outputs, RawDeviceService& device) {
      LOG(info) << "Callback invoked";
      outputs.make<int>(OutputRef{what}, 1);
      device.device()->WaitFor(std::chrono::seconds(minDelay));
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& ctx)
{
  // Get the dataspec option and creates OutputSpecs from it
  auto dataspec = ctx.options().get<std::string>("dataspec");
  std::vector<InputSpec> inputs = select(dataspec.c_str());

  return WorkflowSpec{
    {
      .name = ctx.options().get<std::string>("name"),
      .inputs = inputs,
      .algorithm = AlgorithmSpec{adaptStateless(
        [](InputRecord& inputs) {
          LOG(info) << "Received " << inputs.size() << " messages";
        })},
    }};
}
