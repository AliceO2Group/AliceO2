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
#include "Framework/DataProcessor.h"
#include "Framework/DataSender.h"
#include "Framework/MessageContext.h"
#include "Framework/StringContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/FairMQResizableBuffer.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"

#include <Monitoring/Monitoring.h>
#include <fairmq/Parts.h>
#include <fairmq/Device.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <cstddef>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

namespace o2::framework
{

void DataProcessor::doSend(DataSender& sender, MessageContext& context, ServiceRegistryRef services)
{
  auto& proxy = services.get<FairMQDeviceProxy>();
  std::vector<fair::mq::Parts> outputsPerChannel;
  outputsPerChannel.resize(proxy.getNumOutputChannels());
  auto contextMessages = context.getMessagesForSending();
  for (auto& message : contextMessages) {
    //     monitoringService.send({ message->parts.Size(), "outputs/total" });
    fair::mq::Parts parts = message->finalize();
    assert(message->empty());
    assert(parts.Size() == 2);
    for (auto& part : parts) {
      outputsPerChannel[proxy.getOutputChannelIndex((message->route())).value].AddPart(std::move(part));
    }
  }
  for (int ci = 0; ci < outputsPerChannel.size(); ++ci) {
    auto& parts = outputsPerChannel[ci];
    if (parts.Size() == 0) {
      continue;
    }
    sender.send(parts, {ci});
  }
}

void DataProcessor::doSend(DataSender& sender, StringContext& context, ServiceRegistryRef services)
{
  FairMQDeviceProxy& proxy = services.get<FairMQDeviceProxy>();
  for (auto& messageRef : context) {
    fair::mq::Parts parts;
    fair::mq::MessagePtr payload(sender.create(messageRef.routeIndex));
    auto a = messageRef.payload.get();
    // Rebuild the message using the string as input. For now it involves a copy.
    payload->Rebuild(reinterpret_cast<void*>(const_cast<char*>(strdup(a->data()))), a->size(), nullptr, nullptr);
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    auto* dh = const_cast<DataHeader*>(cdh);
    dh->payloadSize = payload->GetSize();
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    sender.send(parts, proxy.getOutputChannelIndex(messageRef.routeIndex));
  }
}

void DataProcessor::doSend(DataSender& sender, ArrowContext& context, ServiceRegistryRef registry)
{
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;
  auto& monitoring = registry.get<Monitoring>();

  std::regex invalid_metric(" ");
  FairMQDeviceProxy& proxy = registry.get<FairMQDeviceProxy>();
  for (auto& messageRef : context) {
    fair::mq::Parts parts;
    // Depending on how the arrow table is constructed, we finalize
    // the writing here.
    messageRef.finalize(messageRef.buffer);

    std::unique_ptr<fair::mq::Message> payload = messageRef.buffer->Finalise();
    // FIXME: for the moment we simply send empty bodies.
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    auto* dh = const_cast<DataHeader*>(cdh);
    dh->payloadSize = payload->GetSize();
    dh->serialization = o2::header::gSerializationMethodArrow;

    auto origin = std::regex_replace(dh->dataOrigin.as<std::string>(), invalid_metric, "_");
    auto description = std::regex_replace(dh->dataDescription.as<std::string>(), invalid_metric, "_");
    monitoring.send(Metric{(uint64_t)payload->GetSize(),
                           fmt::format("table-bytes-{}-{}-created",
                                       origin,
                                       description)}
                      .addTag(Key::Subsystem, Value::DPL));
    LOGP(info, "Creating {}MB for table {}/{}.", payload->GetSize() / 1000000., dh->dataOrigin, dh->dataDescription);
    context.updateBytesSent(payload->GetSize());
    context.updateMessagesSent(1);
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    sender.send(parts, proxy.getOutputChannelIndex(messageRef.routeIndex));
  }
  static int64_t previousBytesSent = 0;
  auto disposeResources = [bs = context.bytesSent() - previousBytesSent](int taskId,
                                                                         std::array<ComputingQuotaOffer, 16>& offers,
                                                                         ComputingQuotaStats& stats,
                                                                         std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> accountDisposed) {
    ComputingQuotaOffer disposed;
    disposed.sharedMemory = 0;
    int64_t bytesSent = bs;
    for (auto& offer : offers) {
      if (offer.user != taskId) {
        continue;
      }
      int64_t toRemove = std::min((int64_t)bytesSent, offer.sharedMemory);
      offer.sharedMemory -= toRemove;
      bytesSent -= toRemove;
      disposed.sharedMemory += toRemove;
      if (bytesSent <= 0) {
        break;
      }
    }
    return accountDisposed(disposed, stats);
  };
  registry.get<DeviceState>().offerConsumers.emplace_back(disposeResources);
  previousBytesSent = context.bytesSent();
  monitoring.send(Metric{(uint64_t)context.bytesSent(), "arrow-bytes-created"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(uint64_t)context.messagesCreated(), "arrow-messages-created"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.flushBuffer();
}

void DataProcessor::doSend(DataSender& sender, RawBufferContext& context, ServiceRegistryRef registry)
{
  FairMQDeviceProxy& proxy = registry.get<FairMQDeviceProxy>();
  for (auto& messageRef : context) {
    fair::mq::Parts parts;
    fair::mq::MessagePtr payload(sender.create(messageRef.routeIndex));
    auto buffer = messageRef.serializeMsg().str();
    // Rebuild the message using the serialized ostringstream as input. For now it involves a copy.
    size_t size = buffer.length();
    payload->Rebuild(size);
    std::memcpy(payload->GetData(), buffer.c_str(), size);
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    auto* dh = const_cast<DataHeader*>(cdh);
    dh->payloadSize = size;
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    sender.send(parts, proxy.getOutputChannelIndex(messageRef.routeIndex));
  }
}

} // namespace o2::framework
