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
#include "Framework/RootObjectContext.h"
#include "Framework/MessageContext.h"
#include "Framework/StringContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/TMessageSerializer.h"
#include "FairMQResizableBuffer.h"
#include "CommonUtils/BoostSerializer.h"
#include "Headers/DataHeader.h"
#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <cstddef>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

namespace o2
{
namespace framework
{

void DataProcessor::doSend(FairMQDevice& device, FairMQParts&& parts, const char* channel, unsigned int index)
{
  assert(parts.Size() == 2);
  device.Send(parts, channel, index);
}

void DataProcessor::doSend(FairMQDevice &device, MessageContext &context) {
  for (auto &message : context) {
    //     monitoringService.send({ message->parts.Size(), "outputs/total" });
    FairMQParts parts = std::move(message->finalize());
    assert(message->empty());
    assert(parts.Size() == 2);
    device.Send(parts, message->channel(), 0);
    assert(parts.Size() == 2);
  }
}

void DataProcessor::doSend(FairMQDevice &device, RootObjectContext &context) {
  for (auto &messageRef : context) {
    assert(messageRef.payload.get());
    FairMQParts parts;
    FairMQMessagePtr payload(device.NewMessage());
    auto a = messageRef.payload.get();
    device.Serialize<TMessageSerializer>(*payload, a);
    const DataHeader* cdh = o2::header::get<DataHeader*>(messageRef.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    DataHeader *dh = const_cast<DataHeader *>(cdh);
    dh->payloadSize = payload->GetSize();
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    device.Send(parts, messageRef.channel, 0);
  }
}

void DataProcessor::doSend(FairMQDevice& device, StringContext& context)
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

// FIXME: for the moment we simply send empty messages.
void DataProcessor::doSend(FairMQDevice& device, ArrowContext& context)
{
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
    parts.AddPart(std::move(messageRef.header));
    parts.AddPart(std::move(payload));
    device.Send(parts, messageRef.channel, 0);
  }
}

void DataProcessor::doSend(FairMQDevice& device, RawBufferContext& context)
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

} // namespace framework
} // namespace o2
