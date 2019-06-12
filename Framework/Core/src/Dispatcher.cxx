// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Dispatcher.cxx
/// \brief Implementation of Dispatcher for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/Dispatcher.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DataSamplingPolicy.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <fairmq/FairMQDevice.h>

using namespace o2::configuration;

namespace o2
{
namespace framework
{

Dispatcher::Dispatcher(std::string name, const std::string reconfigurationSource)
  : mName(name), mReconfigurationSource(reconfigurationSource)
{
}

Dispatcher::~Dispatcher() = default;

void Dispatcher::init(InitContext& ctx)
{
  LOG(DEBUG) << "Reading Data Sampling Policies...";

  std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(mReconfigurationSource);
  auto policiesTree = cfg->getRecursive("dataSamplingPolicies");
  mPolicies.clear();

  for (auto&& policyConfig : policiesTree) {
    // we don't want the Dispatcher to exit due to one faulty Policy
    try {
      mPolicies.emplace_back(std::make_shared<DataSamplingPolicy>(policyConfig.second));
    } catch (std::exception& ex) {
      LOG(WARN) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "', because: " << ex.what();
    } catch (...) {
      LOG(WARN) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "'";
    }
  }
}

void Dispatcher::run(ProcessingContext& ctx)
{
  for (const auto& input : ctx.inputs()) {
    if (input.header != nullptr && input.spec != nullptr) {
      const auto* inputHeader = header::get<header::DataHeader*>(input.header);
      ConcreteDataMatcher inputMatcher{ inputHeader->dataOrigin, inputHeader->dataDescription, inputHeader->subSpecification };

      for (auto& policy : mPolicies) {
        // todo: consider getting the outputSpec in match to improve performance
        // todo: consider matching (and deciding) in completion policy to save some time

        if (policy->match(inputMatcher) && policy->decide(input)) {

          DataSamplingHeader dsHeader = prepareDataSamplingHeader(*policy.get(), ctx.services().get<const DeviceSpec>());

          if (!policy->getFairMQOutputChannel().empty()) {
            sendFairMQ(ctx.services().get<RawDeviceService>().device(), input, policy->getFairMQOutputChannelName(), std::move(dsHeader));
          } else {
            Output output = policy->prepareOutput(inputMatcher, input.spec->lifetime);
            output.metaHeader = { output.metaHeader, dsHeader };
            send(ctx.outputs(), input, std::move(output));
          }
        }
      }
    }
  }
}

DataSamplingHeader Dispatcher::prepareDataSamplingHeader(const DataSamplingPolicy& policy, const DeviceSpec& spec)
{
  uint64_t sampleTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

  DataSamplingHeader::DeviceIDType id;
  id.runtimeInit(spec.id.substr(0, DataSamplingHeader::deviceIDTypeSize).c_str());

  return {
    sampleTime,
    policy.getTotalAcceptedMessages(),
    policy.getTotalEvaluatedMessages(),
    id
  };
}

void Dispatcher::send(DataAllocator& dataAllocator, const DataRef& inputData, Output&& output) const
{
  const auto* inputHeader = header::get<header::DataHeader*>(inputData.header);
  dataAllocator.snapshot(output, inputData.payload, inputHeader->payloadSize, inputHeader->payloadSerializationMethod);
}

// ideally this should be in a separate proxy device or use Lifetime::External
void Dispatcher::sendFairMQ(FairMQDevice* device, const DataRef& inputData, const std::string& fairMQChannel, DataSamplingHeader&& dsHeader) const
{
  const auto* dh = header::get<header::DataHeader*>(inputData.header);
  assert(dh);
  const auto* dph = header::get<DataProcessingHeader*>(inputData.header);
  assert(dph);

  header::DataHeader dhout{ dh->dataDescription, dh->dataOrigin, dh->subSpecification, dh->payloadSize };
  dhout.payloadSerializationMethod = dh->payloadSerializationMethod;
  DataProcessingHeader dphout{ dph->startTime, dph->duration };
  o2::header::Stack headerStack{ dhout, dphout, dsHeader };

  auto channelAlloc = o2::pmr::getTransportAllocator(device->Transport());
  FairMQMessagePtr msgHeaderStack = o2::pmr::getMessage(std::move(headerStack), channelAlloc);

  char* payloadCopy = new char[dh->payloadSize];
  memcpy(payloadCopy, inputData.payload, dh->payloadSize);
  auto cleanupFcn = [](void* data, void*) { delete[] reinterpret_cast<char*>(data); };
  FairMQMessagePtr msgPayload(device->NewMessage(payloadCopy, dh->payloadSize, cleanupFcn, payloadCopy));

  FairMQParts message;
  message.AddPart(move(msgHeaderStack));
  message.AddPart(move(msgPayload));

  int64_t bytesSent = device->Send(message, fairMQChannel);
}

void Dispatcher::registerPath(const std::pair<InputSpec, OutputSpec>& path)
{
  //todo: take care of inputs inclusive in others, when subSpec matchers are supported
  auto cmp = [a = path.first](const InputSpec b)
  {
    return a.matcher == b.matcher && a.lifetime == b.lifetime;
  };

  if (std::find_if(inputs.begin(), inputs.end(), cmp) == inputs.end()) {
    inputs.push_back(path.first);
    LOG(DEBUG) << "Registering input " << DataSpecUtils::describe(path.first);
  } else {
    LOG(DEBUG) << "Input " << DataSpecUtils::describe(path.first)
               << " already registered";
  }

  outputs.push_back(path.second);
}

const std::string& Dispatcher::getName()
{
  return mName;
}

Inputs Dispatcher::getInputSpecs()
{
  return inputs;
}

Outputs Dispatcher::getOutputSpecs()
{
  return outputs;
}

} // namespace framework
} // namespace o2
