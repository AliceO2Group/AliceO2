// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSampling.cxx
/// \brief Implementation of O2 Data Sampling, v1.0
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>

#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/DataSamplingReadoutAdapter.h"

#include "Framework/DataSampling.h"
#include "Framework/WorkflowDispatcher.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "FairLogger.h"

using namespace o2::configuration;
using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

namespace o2
{
namespace framework
{

std::string DataSampling::dispatcherName()
{
  return std::string("Dispatcher"); //_") + getenv("HOSTNAME");
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const std::string& policiesSource, size_t threads)
{
  LOG(DEBUG) << "Generating Data Sampling infrastructure...";
  WorkflowDispatcher dispatcher(dispatcherName(), policiesSource);
  Options options;

  std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(policiesSource);
  auto policiesTree = cfg->getRecursive("dataSamplingPolicies");

  for (auto&& policyConfig : policiesTree) {

    DataSamplingPolicy policy(policyConfig.second);

    // Dispatcher gets connected to any available outputs, which are listed in policies (active or not).
    // policies partially fulfilled are also taken into account - the user might need data from different FLPs,
    // but corresponding to the same event.
    // If there is a strong need, it can be changed to subscribe to every output in the workflow, which would allow to
    // modify input streams during the runtime.

    LOG(DEBUG) << "Checking if the topology can provide any data for policy '" << policy.getName() << "'.";

    bool dataFound = false;
    for (const auto& dataProcessor : workflow) {
      for (const auto& externalOutput : dataProcessor.outputs) {
        InputSpec candidateInputSpec{
          "doesnt-matter", //externalOutput.binding.value,
          externalOutput.origin,
          externalOutput.description,
          externalOutput.subSpec,
          externalOutput.lifetime
        };

        if (policy.match(candidateInputSpec)) {
          Output output = policy.prepareOutput(candidateInputSpec);
          OutputSpec outputSpec{
            output.origin,
            output.description,
            output.subSpec,
            output.lifetime
          };

          dispatcher.registerPath({ candidateInputSpec, outputSpec });
          dataFound = true;
        }
      }
    }
    if (dataFound && !policy.getFairMQOutputChannel().empty()) {
      options.push_back({ "channel-config", VariantType::String, policy.getFairMQOutputChannel().c_str(), { "Out-of-band channel config" } });
    }
  }

  DataProcessorSpec spec;
  spec.name = dispatcher.getName();
  spec.inputs = dispatcher.getInputSpecs();
  spec.outputs = dispatcher.getOutputSpecs();
  spec.algorithm = adaptFromTask<WorkflowDispatcher>(std::move(dispatcher));
  spec.maxInputTimeslices = threads;
  spec.labels = { { "DataSampling" }, { "Dispatcher" } };
  spec.options = options;

  workflow.emplace_back(std::move(spec));
}

void DataSampling::CustomizeInfrastructure(std::vector<CompletionPolicy>& policies)
{
  CompletionPolicy dispatcherConsumesASAP = CompletionPolicyHelpers::defineByName(dispatcherName(), CompletionPolicy::CompletionOp::Consume);
  policies.push_back(dispatcherConsumesASAP);
}

void DataSampling::CustomizeInfrastructure(std::vector<ChannelConfigurationPolicy>& policies)
{
  // todo: add push-pull for channels that require blocking
  // now it cannot be done, since matching is possible only using data processors names
}

} // namespace framework
} // namespace o2
