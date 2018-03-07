// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DispatcherFlpProto.h"

#include "Framework/SimpleRawDeviceService.h"

DispatcherFlpProto::DispatcherFlpProto(const SubSpecificationType dispatcherSubSpec,
                                       const QcTaskConfiguration& task,
                                       const InfrastructureConfig& cfg) : Dispatcher(dispatcherSubSpec, task, cfg)
{
  //todo: throw an exception when 'name=' not found?
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

DispatcherFlpProto::~DispatcherFlpProto() {}

AlgorithmSpec::ProcessCallback DispatcherFlpProto::initCallback(InitContext& ctx, const std::string& channel,
                                                                double fraction)
{
  auto device = ctx.services().get<RawDeviceService>().device();
  auto gen = Dispatcher::BernoulliGenerator(fraction);
  FlpProtoState state = FlpProtoState::Idle;

  return [gen, device, channel, state](o2::framework::ProcessingContext& pCtx) mutable {
    processCallback(pCtx, gen, device, channel, state);
  };
}

void DispatcherFlpProto::processCallback(ProcessingContext& ctx, BernoulliGenerator& bernoulliGenerator,
                                         FairMQDevice* device, const std::string& channel, FlpProtoState& state)
{
  auto cleanupFcn = [](void* data, void* hint) { delete[] reinterpret_cast<char*>(data); };

  // only one input at a time expected
  assert(ctx.inputs().size() == 1);

  auto input = ctx.inputs().getByPos(0);
  const auto* header = header::get<header::DataHeader>(input.header);

  if (state == FlpProtoState::Idle) {
    // wait until EOM
    if (header->payloadSize == 96 && input.payload[0] == char(0xFF)) {
      // decide in advance whether to take next messages until another EOM
      if (bernoulliGenerator.drawLots()) {
        state = FlpProtoState::ExpectingHeaderOrEOM;
      } else {
        state = FlpProtoState::Idle;
      }
    }
  } else if (state == FlpProtoState::ExpectingHeaderOrEOM) {

    // check what is it
    if (header->payloadSize == 32 && input.payload[0] == char(0xBB)) {
      // it is a header
      char* payloadCopy = new char[header->payloadSize];
      memcpy(payloadCopy, input.payload, header->payloadSize);
      FairMQMessagePtr msgPayload(device->NewMessage(payloadCopy, header->payloadSize, cleanupFcn, payloadCopy));

      device->Send(msgPayload, channel);
      state = FlpProtoState::ExpectingPayload;

    } else if (header->payloadSize == 96 && input.payload[0] == char(0xFF)) {
      // it is an EOM
      char* payloadCopy = new char[header->payloadSize];
      memcpy(payloadCopy, input.payload, header->payloadSize);
      FairMQMessagePtr msgPayload(device->NewMessage(payloadCopy, header->payloadSize, cleanupFcn, payloadCopy));
      device->Send(msgPayload, channel);

      state = bernoulliGenerator.drawLots() ? FlpProtoState::ExpectingHeaderOrEOM : FlpProtoState::Idle;

    } else {
      state = FlpProtoState::Idle;
    }
  } else if (state == FlpProtoState::ExpectingPayload) {

    char* payloadCopy = new char[header->payloadSize];
    memcpy(payloadCopy, input.payload, header->payloadSize);
    FairMQMessagePtr msgPayload(device->NewMessage(payloadCopy, header->payloadSize, cleanupFcn, payloadCopy));
    device->Send(msgPayload, channel);

    // decide in advance whether to take next messages until another EOM
    state = FlpProtoState::ExpectingHeaderOrEOM;
  } else {
    state = FlpProtoState::Idle;
  }
}

void DispatcherFlpProto::addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
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
