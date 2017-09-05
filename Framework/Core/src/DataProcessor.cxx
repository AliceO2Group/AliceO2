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
#include "Framework/TMessageSerializer.h"
#include "Headers/DataHeader.h"
#include <TClonesArray.h>
#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>

using namespace o2::framework;
using DataHeader = o2::Header::DataHeader;

namespace o2 {
namespace framework {

void DataProcessor::doSend(FairMQDevice &device, MessageContext &context) {
  for (auto &message : context) {
 //     metricsService.post("outputs/total", message.parts.Size());
    assert(message.parts.Size() == 2);
    FairMQParts parts = std::move(message.parts);
    assert(message.parts.Size() == 0);
    assert(parts.Size() == 2);
    device.Send(parts, message.channel, 0);
    assert(parts.Size() == 2);
  }
}

void DataProcessor::doSend(FairMQDevice &device, RootObjectContext &context) {
  for (auto &message : context) {
    assert(message.payload.get());
    FairMQParts parts;
    FairMQMessagePtr payload(device.NewMessage());
    TClonesArray *a = reinterpret_cast<TClonesArray*>(message.payload.get());
    device.Serialize<TMessageSerializer>(*payload, a);
    const DataHeader *cdh = o2::Header::get<DataHeader>(message.header->GetData());
    // sigh... See if we can avoid having it const by not
    // exposing it to the user in the first place.
    DataHeader *dh = const_cast<DataHeader *>(cdh);
    dh->payloadSize = payload->GetSize();
    parts.AddPart(std::move(message.header));
    parts.AddPart(std::move(payload));
    device.Send(parts, message.channel, 0);
  }
}

} // namespace framework
} // namespace o2
