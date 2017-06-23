// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataAllocator.h"
#include "Framework/MessageContext.h"
#include "Framework/DataSpecUtils.h"

namespace o2 {
namespace framework {

DataAllocator::DataAllocator(FairMQDevice *device, MessageContext *context, const AllowedOutputsMap &outputs)
: mDevice{device},
  mContext{context},
  mAllowedOutputs{outputs}
{
}

std::string
DataAllocator::matchDataHeader(DataOrigin origin, DataDescription description, SubSpecificationType subSpec) {
  for (auto &output : mAllowedOutputs) {
    if (DataSpecUtils::match(output.second, origin, description, subSpec)) {
      return output.first;
    }
  }
  std::ostringstream str;
  str << "Worker is not authorised to create message with "
      << "origin(" << origin.str << ")"
      << "description(" << description.str << ")"
      << "subSpec(" << subSpec << ")";
  throw std::runtime_error(str.str());
}

DataChunk
DataAllocator::newChunk(DataOrigin origin, DataDescription description, SubSpecificationType subSpec, size_t size) {
  std::string channel = matchDataHeader(origin, description, subSpec);
  FairMQParts parts;
  FairMQMessagePtr headerMessage = mDevice->NewMessageFor(channel, 0, sizeof(Header::DataHeader));
  Header::DataHeader *header = reinterpret_cast<Header::DataHeader*>(headerMessage->GetData());
  header->dataOrigin = origin;
  header->dataDescription = description;
  header->subSpecification = subSpec;
  // FIXME: how do we want to use subchannels? time based parallelism?
  FairMQMessagePtr payloadMessage = mDevice->NewMessageFor(channel, 0, size);
  auto dataPtr = payloadMessage->GetData();
  auto dataSize = payloadMessage->GetSize();
  parts.AddPart(std::move(headerMessage));
  parts.AddPart(std::move(payloadMessage));
  assert(parts.Size() == 2);
  mContext->addPart(std::move(parts), channel, 0);
  assert(parts.Size() == 0);
  return DataChunk{reinterpret_cast<char*>(dataPtr), dataSize};
}

DataChunk
DataAllocator::adoptChunk(DataOrigin origin, DataDescription description, SubSpecificationType subSpec, char *buffer, size_t size, fairmq_free_fn *freefn, void *hint = nullptr) {
  // Find a matching channel, create a new message for it and put it in the
  // queue to be sent at the end of the processing
  std::string channel = matchDataHeader(origin, description, subSpec);
  FairMQParts parts;
  FairMQMessagePtr headerMessage = mDevice->NewMessageFor(channel, 0, sizeof(Header::DataHeader));
  Header::DataHeader *header = reinterpret_cast<Header::DataHeader*>(headerMessage->GetData());
  header->dataOrigin = origin;
  header->dataDescription = description;
  header->subSpecification = subSpec;
  // FIXME: how do we want to use subchannels? time based parallelism?
  FairMQMessagePtr payloadMessage = mDevice->NewMessageFor(channel, 0, buffer, size, freefn, hint);
  auto dataPtr = payloadMessage->GetData();
  LOG(DEBUG) << "New payload at " << payloadMessage->GetData();
  auto dataSize = payloadMessage->GetSize();
  parts.AddPart(std::move(headerMessage));
  parts.AddPart(std::move(payloadMessage));
  mContext->addPart(std::move(parts), channel, 0);
  return DataChunk{reinterpret_cast<char *>(dataPtr), dataSize};
}

}
}
