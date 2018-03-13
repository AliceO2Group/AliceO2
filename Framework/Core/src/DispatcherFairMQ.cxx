// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DispatcherFairMQ.cxx
/// \brief Implementation of DispatcherFairMQ for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DispatcherFairMQ.h"

#include "FairLogger.h"
#include "FairMQDevice.h"
#include "FairMQTransportFactory.h"
#include "Framework/SimpleRawDeviceService.h"

namespace o2 {
namespace framework {

DispatcherFairMQ::DispatcherFairMQ(const SubSpecificationType dispatcherSubSpec,
                                   const QcTaskConfiguration& task,
                                   const InfrastructureConfig& cfg) : Dispatcher(dispatcherSubSpec, task, cfg)
{

  // todo: throw an exception when 'name=' not found?
  size_t nameBegin = task.fairMqOutputChannelConfig.find("name=") + sizeof("name=") - 1;
  size_t nameEnd = task.fairMqOutputChannelConfig.find_first_of(',', nameBegin);
  std::string channel = task.fairMqOutputChannelConfig.substr(nameBegin, nameEnd - nameBegin);

  mDataProcessorSpec.algorithm = AlgorithmSpec{
    [fraction = task.fractionOfDataToSample, channel](InitContext& ctx) {
      return initCallback(ctx, channel, fraction);
    }
  };
  mDataProcessorSpec.options.push_back({
      "channel-config", VariantType::String, task.fairMqOutputChannelConfig.c_str(), { "Out-of-band channel config" }
    });
}

DispatcherFairMQ::~DispatcherFairMQ() {}

AlgorithmSpec::ProcessCallback DispatcherFairMQ::initCallback(InitContext& ctx, const std::string& channel,
                                                              double fraction)
{
  auto device = ctx.services().get<RawDeviceService>().device();
  auto gen = Dispatcher::BernoulliGenerator(fraction);

  return [gen, device, channel](o2::framework::ProcessingContext& pCtx) mutable {
    processCallback(pCtx, gen, device, channel);
  };
}

void DispatcherFairMQ::processCallback(ProcessingContext& ctx, Dispatcher::BernoulliGenerator& bernoulliGenerator,
                                       FairMQDevice* device, const std::string& channel)
{
  InputRecord& inputs = ctx.inputs();

  if (bernoulliGenerator.drawLots()) {

    auto cleanupFcn = [](void* data, void* hint) { delete[] reinterpret_cast<char*>(data); };
    for (auto& input : inputs) {
      const auto* header = header::get<header::DataHeader>(input.header);

      char* payloadCopy = new char[header->payloadSize];
      memcpy(payloadCopy, input.payload, header->payloadSize);
      FairMQMessagePtr msgPayload(device->NewMessage(payloadCopy, header->payloadSize, cleanupFcn, payloadCopy));

      int bytesSent = device->Send(msgPayload, channel);
      LOG(DEBUG) << "Payload bytes sent: " << bytesSent;
    }
  }
}

void DispatcherFairMQ::addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                                 const std::string& binding)
{
  InputSpec newInput{
    binding,
    externalOutput.origin,
    externalOutput.description,
    externalOutput.subSpec,
    static_cast<InputSpec::Lifetime>(externalOutput.lifetime),
  };

  mDataProcessorSpec.inputs.push_back(newInput);
}

} // namespace framework
} // namespace o2