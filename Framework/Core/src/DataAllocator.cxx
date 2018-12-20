// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataAllocator.h"
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/Stack.h"

#include <fairmq/FairMQDevice.h>

#include <TClonesArray.h>


namespace o2
{
namespace framework
{

using DataHeader = o2::header::DataHeader;
using DataDescription = o2::header::DataDescription;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

DataAllocator::DataAllocator(TimingInfo* timingInfo,
                             ContextRegistry* contextRegistry,
                             const AllowedOutputRoutes& routes)
  : mAllowedOutputRoutes{ routes },
    mTimingInfo{ timingInfo },
    mContextRegistry{ contextRegistry }
{
}

std::string
DataAllocator::matchDataHeader(const Output& spec, size_t timeslice) {
  // FIXME: we should take timeframeId into account as well.
  for (auto &output : mAllowedOutputRoutes) {
    if (DataSpecUtils::match(output.matcher, spec.origin, spec.description, spec.subSpec)
        && ((timeslice % output.maxTimeslices) == output.timeslice)) {
      return output.channel;
    }
  }
  std::ostringstream str;
  str << "Worker is not authorised to create message with "
      << "origin(" << spec.origin.as<std::string>() << ")"
      << "description(" << spec.description.as<std::string>() << ")"
      << "subSpec(" << spec.subSpec << ")";
  throw std::runtime_error(str.str());
}

DataChunk
DataAllocator::newChunk(const Output& spec, size_t size) {
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  auto context = mContextRegistry->get<MessageContext>();

  FairMQMessagePtr headerMessage = headerMessageFromOutput(spec, channel,                        //
                                                           o2::header::gSerializationMethodNone, //
                                                           size                                  //
                                                           );
  FairMQMessagePtr payloadMessage = context->proxy().getDevice()->NewMessageFor(channel, 0, size);
  auto dataPtr = payloadMessage->GetData();
  auto dataSize = payloadMessage->GetSize();

  FairMQParts parts;
  parts.AddPart(std::move(headerMessage));
  parts.AddPart(std::move(payloadMessage));
  assert(parts.Size() == 2);
  context->addPart(std::move(parts), channel);
  assert(parts.Size() == 0);
  return DataChunk{reinterpret_cast<char*>(dataPtr), dataSize};
}

DataChunk
DataAllocator::adoptChunk(const Output& spec, char *buffer, size_t size, fairmq_free_fn *freefn, void *hint = nullptr) {
  // Find a matching channel, create a new message for it and put it in the
  // queue to be sent at the end of the processing
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);

  FairMQMessagePtr headerMessage = headerMessageFromOutput(spec, channel,                        //
                                                           o2::header::gSerializationMethodNone, //
                                                           size                                  //
                                                           );

  FairMQParts parts;

  // FIXME: how do we want to use subchannels? time based parallelism?
  auto context = mContextRegistry->get<MessageContext>();
  FairMQMessagePtr payloadMessage = context->proxy().getDevice()->NewMessageFor(channel, 0, buffer, size, freefn, hint);
  auto dataPtr = payloadMessage->GetData();
  LOG(DEBUG) << "New payload at " << payloadMessage->GetData();
  auto dataSize = payloadMessage->GetSize();
  parts.AddPart(std::move(headerMessage));
  parts.AddPart(std::move(payloadMessage));
  context->addPart(std::move(parts), channel);
  return DataChunk{reinterpret_cast<char *>(dataPtr), dataSize};
}

FairMQMessagePtr DataAllocator::headerMessageFromOutput(Output const& spec,                     //
                                                        std::string const& channel,             //
                                                        o2::header::SerializationMethod method, //
                                                        size_t payloadSize)                     //
{
  DataHeader dh;
  dh.dataOrigin = spec.origin;
  dh.dataDescription = spec.description;
  dh.subSpecification = spec.subSpec;
  dh.payloadSize = payloadSize;
  dh.payloadSerializationMethod = method;

  DataProcessingHeader dph{mTimingInfo->timeslice, 1};
  auto context = mContextRegistry->get<MessageContext>();

  auto channelAlloc = o2::pmr::getTransportAllocator(context->proxy().getTransport(channel, 0));
  return o2::pmr::getMessage(o2::header::Stack{ channelAlloc, dh, dph, spec.metaHeader });
}

void DataAllocator::addPartToContext(FairMQMessagePtr&& payloadMessage, const Output& spec,
                                     o2::header::SerializationMethod serializationMethod)
{
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  // the correct payload size is st later when sending the
  // RootObjectContext, see DataProcessor::doSend
  auto headerMessage = headerMessageFromOutput(spec, channel, serializationMethod, 0);

  FairMQParts parts;

  // FIXME: this is kind of ugly, we know that we can change the content of the
  // header message because we have just created it, but the API declares it const
  const DataHeader* cdh = o2::header::get<DataHeader*>(headerMessage->GetData());
  DataHeader* dh = const_cast<DataHeader*>(cdh);
  dh->payloadSize = payloadMessage->GetSize();
  parts.AddPart(std::move(headerMessage));
  parts.AddPart(std::move(payloadMessage));
  mContextRegistry->get<MessageContext>()->addPart(std::move(parts), channel);
}

void DataAllocator::adopt(const Output& spec, TObject* ptr)
{
  std::unique_ptr<TObject> payload(ptr);
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  // the correct payload size is set later when sending the
  // RootObjectContext, see DataProcessor::doSend
  auto header = headerMessageFromOutput(spec, channel, o2::header::gSerializationMethodROOT, 0);
  mContextRegistry->get<RootObjectContext>()->addObject(std::move(header), std::move(payload), channel);
  assert(payload.get() == nullptr);
}

void DataAllocator::adopt(const Output& spec, std::string* ptr)
{
  std::unique_ptr<std::string> payload(ptr);
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  // the correct payload size is set later when sending the
  // StringContext, see DataProcessor::doSend
  auto header = headerMessageFromOutput(spec, channel, o2::header::gSerializationMethodNone, 0);
  mContextRegistry->get<StringContext>()->addString(std::move(header), std::move(payload), channel);
  assert(payload.get() == nullptr);
}

void DataAllocator::adopt(const Output& spec, TableBuilder* tb)
{
  std::unique_ptr<TableBuilder> payload(tb);
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  auto header = headerMessageFromOutput(spec, channel, o2::header::gSerializationMethodArrow, 0);
  auto context = mContextRegistry->get<ArrowContext>();
  assert(context);
  context->addTable(std::move(header), std::move(payload), channel);
  assert(payload.get() == nullptr);
}

Output DataAllocator::getOutputByBind(OutputRef&& ref)
{
  if (ref.label.empty()) {
    throw std::runtime_error("Invalid (empty) OutputRef provided.");
  }
  for (size_t ri = 0, re = mAllowedOutputRoutes.size(); ri != re; ++ri) {
    if (mAllowedOutputRoutes[ri].matcher.binding.value == ref.label) {
      auto spec = mAllowedOutputRoutes[ri].matcher;
      return Output{ spec.origin, spec.description, ref.subSpec, spec.lifetime, std::move(ref.headerStack) };
    }
  }
  throw std::runtime_error("Unable to find OutputSpec with label " + ref.label);
  assert(false);
}

}
}
