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
#include "DataSampling/DataSampling.h"
#include "Framework/CompletionPolicyHelpers.h"
#include <vector>
#include <filesystem>

using namespace o2::framework;
using namespace o2::utilities;

void customize(std::vector<CompletionPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
  policies.push_back(CompletionPolicyHelpers::defineByName("dataSink", CompletionPolicy::CompletionOp::Consume));
}

void customize(std::vector<ChannelConfigurationPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"sampling-fraction", VariantType::Double, 1.0, {"sampling fraction"}});
  workflowOptions.push_back(ConfigParamSpec{"payload-size", VariantType::Int, 10000, {"payload size"}});
  workflowOptions.push_back(ConfigParamSpec{"producers", VariantType::Int, 1, {"number of producers"}});
  workflowOptions.push_back(ConfigParamSpec{"dispatchers", VariantType::Int, 1, {"number of dispatchers"}});
  workflowOptions.push_back(ConfigParamSpec{"usleep", VariantType::Int, 0, {"usleep time of producers"}});
  workflowOptions.push_back(ConfigParamSpec{
    "fill", VariantType::Bool, false, {"should fill the messages (prevents memory overcommitting)"}});
}

#include <memory>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/functional/hash.hpp>
#include <boost/property_tree/ptree.hpp>
#include <fairmq/Device.h>
#include <iostream>
#include "Headers/DataHeader.h"
#include "Framework/ControlService.h"
#include "DataSampling/DataSampling.h"
#include "DataSampling/DataSamplingPolicy.h"
#include "Framework/RawDeviceService.h"
#include "Framework/runDataProcessing.h"

using namespace o2::framework;
using namespace o2::utilities;
using namespace boost::property_tree;
using SubSpec = o2::header::DataHeader::SubSpecificationType;

// clang-format off
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  double samplingFraction = config.options().get<double>("sampling-fraction");
  size_t payloadSize = config.options().get<int>("payload-size");
  size_t producers = config.options().get<int>("producers");
  size_t dispatchers = config.options().get<int>("dispatchers");
  size_t usleepTime = config.options().get<int>("usleep");
  bool fill = config.options().get<bool>("fill");

  ptree policy;
  policy.put("id", "benchmark");
  policy.put("active", "true");
  policy.put("query", "TST:TST/RAWDATA");
  ptree samplingConditions;
  ptree conditionRandom;
  conditionRandom.put("condition", "random");
  conditionRandom.put("fraction", std::to_string(samplingFraction));
  conditionRandom.put("seed", "22222");
  samplingConditions.push_back(std::make_pair("", conditionRandom));
  policy.add_child("samplingConditions", samplingConditions);
  policy.put("blocking", "false");
  ptree policies;
  policies.push_back(std::make_pair("", policy));

  WorkflowSpec specs;

  for (size_t p = 0; p < producers; p++) {
    specs.push_back(DataProcessorSpec{
      "dataProducer" + std::to_string(p),
      Inputs{},
      Outputs{
        OutputSpec{ "TST", "RAWDATA", static_cast<SubSpec>(p) }
      },
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback) [=](InitContext& ictx) {
          return (AlgorithmSpec::ProcessCallback) [=](ProcessingContext& pctx) mutable {
            usleep(usleepTime);
            auto data = pctx.outputs().make<char>(Output{ "TST", "RAWDATA", static_cast<SubSpec>(p) }, payloadSize);
            if (fill) {
              memset(data.data(), 0x00, payloadSize);
            }
          };
        }
      }
    });
  }

  DataSampling::GenerateInfrastructure(specs, policies, dispatchers);

  DataProcessorSpec podDataSink{
    "dataSink",
    Inputs{{"test-data", {DataSamplingPolicy::createPolicyDataOrigin(), DataSamplingPolicy::createPolicyDataDescription("benchmark", 0)}}},
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext & ctx){}}};

  specs.push_back(podDataSink);
  return specs;
}
// clang-format on
