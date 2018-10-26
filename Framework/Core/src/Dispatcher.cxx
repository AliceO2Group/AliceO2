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

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQLogger.h>

using namespace o2::configuration;

namespace o2
{
namespace framework
{

Dispatcher::Dispatcher(std::string name, const std::string reconfigurationSource)
  : mName(name), mReconfigurationSource(reconfigurationSource)
{
}

Dispatcher::~Dispatcher()
{
}

void Dispatcher::init(InitContext& ctx)
{
  LOG(DEBUG) << "Reading Data Sampling Policies...";

  std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(mReconfigurationSource);
  auto policiesTree = cfg->getRecursive("dataSamplingPolicies");
  mPolicies.clear();

  for (auto&& policyConfig : policiesTree) {
    mPolicies.emplace_back(std::make_shared<DataSamplingPolicy>(policyConfig.second));
  }
}

void Dispatcher::run(ProcessingContext& ctx)
{
  for (const auto& input : ctx.inputs()) {
    if (input.header != nullptr && input.spec != nullptr) {

      for (auto& policy : mPolicies) {
        // todo: consider getting the outputSpec in match to improve performance
        // todo: consider matching (and deciding) in completion policy to save some time
        if (policy->match(*input.spec) && policy->decide(input)) {

          if (!policy->getFairMQOutputChannel().empty()) {
            sendFairMQ(ctx.services().get<RawDeviceService>().device(), input, policy->getFairMQOutputChannelName());
          } else {
            send(ctx.outputs(), input, policy->prepareOutput(*input.spec));
          }
        }
      }
    }
  }
}

void Dispatcher::send(DataAllocator& dataAllocator, const DataRef& inputData, const Output& output) const
{
  //todo: support other serialization methods
  const auto* inputHeader = header::get<header::DataHeader*>(inputData.header);
  if (inputHeader->payloadSerializationMethod == header::gSerializationMethodInvalid) {
    LOG(WARNING) << "DataSampling::dispatcherCallback: input of origin'" << inputHeader->dataOrigin.str
                 << "', description '" << inputHeader->dataDescription.str
                 << "' has gSerializationMethodInvalid.";
  } else if (inputHeader->payloadSerializationMethod == header::gSerializationMethodROOT) {
    dataAllocator.adopt(output, DataRefUtils::as<TObject>(inputData).release());
  } else { // POD
    // todo: do it non-copy, when API is available
    auto outputMessage = dataAllocator.newChunk(output, inputHeader->payloadSize);
    memcpy(outputMessage.data, inputData.payload, inputHeader->payloadSize);
  }
}

// ideally this should be in a separate proxy device or use Lifetime::External
void Dispatcher::sendFairMQ(const FairMQDevice* device, const DataRef& inputData, const std::string& fairMQChannel) const
{
  const auto* dh = header::get<header::DataHeader*>(inputData.header);
  assert(dh);
  const auto* dph = header::get<DataProcessingHeader*>(inputData.header);
  assert(dph);

  header::DataHeader dhout{ dh->dataDescription, dh->dataOrigin, dh->subSpecification, dh->payloadSize };
  dhout.payloadSerializationMethod = dh->payloadSerializationMethod;
  DataProcessingHeader dphout{ dph->startTime, dph->duration };
  o2::header::Stack headerStack{ dhout, dphout };

  auto channelAlloc = o2::memory_resource::getTransportAllocator(device->Transport());
  FairMQMessagePtr msgHeaderStack = o2::memory_resource::getMessage(std::move(headerStack), channelAlloc);

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
    return a.origin == b.origin && a.description == b.description && a.subSpec == b.subSpec && a.lifetime == b.lifetime;
  };

  if (std::find_if(inputs.begin(), inputs.end(), cmp) == inputs.end()) {
    inputs.push_back(path.first);
    LOG(DEBUG) << "Registering input " << path.first.origin.str << " " << path.first.description.str << " "
               << path.first.subSpec;
  } else {
    LOG(DEBUG) << "Input " << path.first.origin.str << " " << path.first.description.str << " " << path.first.subSpec
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
