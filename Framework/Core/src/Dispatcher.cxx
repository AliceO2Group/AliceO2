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
#include "Framework/DataSamplingHeader.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"

#include "Framework/Monitoring.h"
#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <fairmq/FairMQDevice.h>

using namespace o2::configuration;
using namespace o2::monitoring;

namespace o2::framework
{

Dispatcher::Dispatcher(std::string name, const std::string reconfigurationSource)
  : mName(name), mReconfigurationSource(reconfigurationSource)
{
  header::DataDescription timerDescription;
  timerDescription.runtimeInit(("TIMER-" + name).substr(0, 16).c_str());
  inputs.emplace_back(InputSpec{"timer-stats", "DS", timerDescription, 0, Lifetime::Timer});
}

Dispatcher::~Dispatcher() = default;

void Dispatcher::init(InitContext& ctx)
{
  LOG(DEBUG) << "Reading Data Sampling Policies...";

  boost::property_tree::ptree policiesTree;

  if (mReconfigurationSource.empty() == false) {
    std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(mReconfigurationSource);
    policiesTree = cfg->getRecursive("dataSamplingPolicies");
    mPolicies.clear();
  } else {
    policiesTree = ctx.options().get<boost::property_tree::ptree>("sampling-config-ptree");
    mPolicies.clear();
  }

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
  for (const auto& input : InputRecordWalker(ctx.inputs())) {

    const auto* inputHeader = header::get<header::DataHeader*>(input.header);
    ConcreteDataMatcher inputMatcher{inputHeader->dataOrigin, inputHeader->dataDescription, inputHeader->subSpecification};

    for (auto& policy : mPolicies) {
      // todo: consider getting the outputSpec in match to improve performance
      // todo: consider matching (and deciding) in completion policy to save some time

      if (policy->match(inputMatcher) && policy->decide(input)) {
        // We copy every header which is not DataHeader or DataProcessingHeader,
        // so that custom data-dependent headers are passed forward,
        // and we add a DataSamplingHeader.
        header::Stack headerStack{
          std::move(extractAdditionalHeaders(input.header)),
          std::move(prepareDataSamplingHeader(*policy.get(), ctx.services().get<const DeviceSpec>()))};

        if (!policy->getFairMQOutputChannel().empty()) {
          sendFairMQ(ctx.services().get<RawDeviceService>().device(), input, policy->getFairMQOutputChannelName(),
                     std::move(headerStack));
        } else {
          Output output = policy->prepareOutput(inputMatcher, input.spec->lifetime);
          output.metaHeader = std::move(header::Stack{std::move(output.metaHeader), std::move(headerStack)});
          send(ctx.outputs(), input, std::move(output));
        }
      }
    }
  }

  if (ctx.inputs().isValid("timer-stats")) {
    reportStats(ctx.services().get<Monitoring>());
  }
}

void Dispatcher::reportStats(Monitoring& monitoring) const
{
  uint64_t dispatcherTotalEvaluatedMessages = 0;
  uint64_t dispatcherTotalAcceptedMessages = 0;

  for (const auto& policy : mPolicies) {
    dispatcherTotalEvaluatedMessages += policy->getTotalEvaluatedMessages();
    dispatcherTotalAcceptedMessages += policy->getTotalAcceptedMessages();
  }

  monitoring.send({dispatcherTotalEvaluatedMessages, "Dispatcher_messages_evaluated"});
  monitoring.send({dispatcherTotalAcceptedMessages, "Dispatcher_messages_passed"});
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
    id};
}

header::Stack Dispatcher::extractAdditionalHeaders(const char* inputHeaderStack) const
{
  header::Stack headerStack;

  const auto* first = header::BaseHeader::get(reinterpret_cast<const byte*>(inputHeaderStack));
  for (const auto* current = first; current != nullptr; current = current->next()) {
    if (current->description != header::DataHeader::sHeaderType &&
        current->description != DataProcessingHeader::sHeaderType) {
      headerStack = std::move(header::Stack{std::move(headerStack), *current});
    }
  }

  return headerStack;
}

void Dispatcher::send(DataAllocator& dataAllocator, const DataRef& inputData, Output&& output) const
{
  const auto* inputHeader = header::get<header::DataHeader*>(inputData.header);
  dataAllocator.snapshot(output, inputData.payload, inputHeader->payloadSize, inputHeader->payloadSerializationMethod);
}

// ideally this should be in a separate proxy device or use Lifetime::External
void Dispatcher::sendFairMQ(FairMQDevice* device, const DataRef& inputData, const std::string& fairMQChannel,
                            header::Stack&& stack) const
{
  const auto* dh = header::get<header::DataHeader*>(inputData.header);
  assert(dh);
  const auto* dph = header::get<DataProcessingHeader*>(inputData.header);
  assert(dph);

  header::DataHeader dhout{dh->dataDescription, dh->dataOrigin, dh->subSpecification, dh->payloadSize};
  dhout.payloadSerializationMethod = dh->payloadSerializationMethod;
  DataProcessingHeader dphout{dph->startTime, dph->duration};
  o2::header::Stack headerStack{dhout, dphout, stack};

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
  auto cmp = [a = path.first](const InputSpec b) {
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

} // namespace o2::framework
