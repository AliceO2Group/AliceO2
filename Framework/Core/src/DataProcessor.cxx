// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataProcessor.h"
#include "Framework/MessageContext.h"
#include "Framework/StringContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/ServiceRegistry.h"
#include "FairMQResizableBuffer.h"
#include "CommonUtils/BoostSerializer.h"
#include "Headers/DataHeader.h"

#include <Monitoring/Monitoring.h>
#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <cstddef>
#include <unordered_map>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

namespace o2::framework
{

void DataProcessor::doSend(FairMQDevice& device, FairMQParts&& parts, const char* channel, unsigned int index)
{
  device.Send(parts, channel, index);
}

void DataProcessor::doSend(FairMQDevice& device, MessageContext& context, ServiceRegistry&)
{
  std::unordered_map<std::string const*, FairMQParts> outputs;
  auto contextMessages = context.getMessagesForSending();
  for (auto& message : contextMessages) {
    //     monitoringService.send({ message->parts.Size(), "outputs/total" });
    FairMQParts parts = std::move(message->finalize());
    assert(message->empty());
    assert(parts.Size() == 2);
    for (auto& part : parts) {
      outputs[&(message->channel())].AddPart(std::move(part));
    }
  }
  for (auto& [channel, parts] : outputs) {
    device.Send(parts, *channel, 0);
  }
}

void DataProcessor::doSend(FairMQDevice& device, StringContext& context, ServiceRegistry&)
{
  for (auto& messageRef : context) {
    FairMQParts parts;
    FairMQMessagePtr payload(device.NewMessage());
    auto a = messageRef.payload.get();
    // Rebuild the message using the string as input. For now it involves a copy.
    payload->Rebuild(reinterpret_cast<void*>(const_cast<char*>(strdup(a->data()))), a->size(), nullptr, nullptr);
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    DataHeader* dh = const_cast<DataHeader*>(cdh);
    dh->payloadSize = payload->GetSize();
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    device.Send(parts, messageRef.channel, 0);
  }
}

void DataProcessor::doSend(FairMQDevice& device, ArrowContext& context, ServiceRegistry& registry)
{
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;
  auto& monitoring = registry.get<Monitoring>();

  for (auto& messageRef : context) {
    FairMQParts parts;
    // Depending on how the arrow table is constructed, we finalize
    // the writing here.
    messageRef.finalize(messageRef.buffer);

    std::unique_ptr<FairMQMessage> payload = messageRef.buffer->Finalise();
    // FIXME: for the moment we simply send empty bodies.
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    DataHeader* dh = const_cast<DataHeader*>(cdh);
    dh->payloadSize = payload->GetSize();
    dh->serialization = o2::header::gSerializationMethodArrow;
    monitoring.send(Metric{(uint64_t)payload->GetSize(), fmt::format("table-bytes-{}-{}-created", dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>())}.addTag(Key::Subsystem, Value::DPL));
    context.updateBytesSent(payload->GetSize());
    context.updateMessagesSent(1);
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    device.Send(parts, messageRef.channel, 0);
  }
  static int64_t previousBytesSent = 0;
  auto disposeResources = [bs = context.bytesSent() - previousBytesSent](int taskId, std::array<ComputingQuotaOffer, 16>& offers) {
    int64_t bytesSent = bs;
    for (size_t oi = 0; oi < offers.size(); oi++) {
      auto& offer = offers[oi];
      if (offer.user != taskId) {
        continue;
      }
      int64_t toRemove = std::min((int64_t)bytesSent, offer.sharedMemory);
      offer.sharedMemory -= toRemove;
      bytesSent -= toRemove;
      if (bytesSent <= 0) {
        return;
      }
    }
  };
  registry.get<DeviceState>().offerConsumers.push_back(disposeResources);
  previousBytesSent = context.bytesSent();
  monitoring.send(Metric{(uint64_t)context.bytesSent(), "arrow-bytes-created"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(uint64_t)context.messagesCreated(), "arrow-messages-created"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.flushBuffer();
}

void DataProcessor::doSend(FairMQDevice& device, RawBufferContext& context, ServiceRegistry& registry)
{
  for (auto& messageRef : context) {
    FairMQParts parts;
    FairMQMessagePtr payload(device.NewMessage());
    auto buffer = messageRef.serializeMsg().str();
    // Rebuild the message using the serialized ostringstream as input. For now it involves a copy.
    size_t size = buffer.length();
    payload->Rebuild(size);
    std::memcpy(payload->GetData(), buffer.c_str(), size);
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    DataHeader* dh = const_cast<DataHeader*>(cdh);
    dh->payloadSize = size;
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    device.Send(parts, messageRef.channel, 0);
  }
}

} // namespace o2::framework
