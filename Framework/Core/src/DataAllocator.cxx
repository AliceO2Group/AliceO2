// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CompilerBuiltins.h"
#include "Framework/DataAllocator.h"
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/Stack.h"
#include "FairMQResizableBuffer.h"

#include <fairmq/FairMQDevice.h>

#include <arrow/ipc/writer.h>
#include <arrow/type.h>
#include <arrow/io/memory.h>

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

DataChunk& DataAllocator::newChunk(const Output& spec, size_t size)
{
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  auto context = mContextRegistry->get<MessageContext>();

  FairMQMessagePtr headerMessage = headerMessageFromOutput(spec, channel,                        //
                                                           o2::header::gSerializationMethodNone, //
                                                           size                                  //
                                                           );
  auto& co = context->add<MessageContext::ContainerRefObject<DataChunk>>(std::move(headerMessage), channel, 0, size);
  return co;
}

void DataAllocator::adoptChunk(const Output& spec, char* buffer, size_t size, fairmq_free_fn* freefn, void* hint = nullptr)
{
  // Find a matching channel, create a new message for it and put it in the
  // queue to be sent at the end of the processing
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);

  FairMQMessagePtr headerMessage = headerMessageFromOutput(spec, channel,                        //
                                                           o2::header::gSerializationMethodNone, //
                                                           size                                  //
                                                           );

  // FIXME: how do we want to use subchannels? time based parallelism?
  auto context = mContextRegistry->get<MessageContext>();
  context->add<MessageContext::TrivialObject>(std::move(headerMessage), channel, 0, buffer, size, freefn, hint);
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

  // FIXME: this is kind of ugly, we know that we can change the content of the
  // header message because we have just created it, but the API declares it const
  const DataHeader* cdh = o2::header::get<DataHeader*>(headerMessage->GetData());
  DataHeader* dh = const_cast<DataHeader*>(cdh);
  dh->payloadSize = payloadMessage->GetSize();
  auto context = mContextRegistry->get<MessageContext>();
  context->add<MessageContext::TrivialObject>(std::move(headerMessage), std::move(payloadMessage), channel);
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
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  auto header = headerMessageFromOutput(spec, channel, o2::header::gSerializationMethodArrow, 0);
  auto context = mContextRegistry->get<ArrowContext>();

  auto creator = [device = context->proxy().getDevice()](size_t s) -> std::unique_ptr<FairMQMessage> { return device->NewMessage(s); };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  /// To finalise this we write the table to the buffer.
  /// FIXME: most likely not a great idea. We should probably write to the buffer
  ///        directly in the TableBuilder, incrementally.
  std::shared_ptr<TableBuilder> p(tb);
  auto finalizer = [payload = p](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    auto table = payload->finalize();

    auto stream = std::make_shared<arrow::io::BufferOutputStream>(b);
    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    auto outBatch = arrow::ipc::RecordBatchStreamWriter::Open(stream.get(), table->schema(), &writer);
    auto outStatus = writer->WriteTable(*table);
    if (outStatus.ok() == false) {
      throw std::runtime_error("Unable to Write table");
    }
  };

  assert(context);
  context->addBuffer(std::move(header), buffer, std::move(finalizer), channel);
}

void DataAllocator::snapshot(const Output& spec, const char* payload, size_t payloadSize,
                             o2::header::SerializationMethod serializationMethod)
{
  auto proxy = mContextRegistry->get<MessageContext>()->proxy();
  FairMQMessagePtr payloadMessage(proxy.createMessage(payloadSize));
  memcpy(payloadMessage->GetData(), payload, payloadSize);

  addPartToContext(std::move(payloadMessage), spec, serializationMethod);
}

void DataAllocator::create(const Output& spec,
                           std::shared_ptr<arrow::ipc::RecordBatchWriter>* writer,
                           std::shared_ptr<arrow::Schema> schema)
{
  std::string channel = matchDataHeader(spec, mTimingInfo->timeslice);
  auto header = headerMessageFromOutput(spec, channel, o2::header::gSerializationMethodArrow, 0);
  auto context = mContextRegistry->get<ArrowContext>();

  auto creator = [device = context->proxy().getDevice()](size_t s) -> std::unique_ptr<FairMQMessage> { return device->NewMessage(s); };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);
  auto stream = std::make_shared<arrow::io::BufferOutputStream>(buffer);
  auto outBatch = arrow::ipc::RecordBatchStreamWriter::Open(stream.get(), schema, writer);

  auto finalizer = [stream](std::shared_ptr<FairMQResizableBuffer>) -> void {
    auto s = stream->Close();
    if (s.ok() == false) {
      throw std::runtime_error("Error while closing stream");
    }
  };

  assert(context);
  context->addBuffer(std::move(header), buffer, std::move(finalizer), channel);
}

Output DataAllocator::getOutputByBind(OutputRef&& ref)
{
  if (ref.label.empty()) {
    throw std::runtime_error("Invalid (empty) OutputRef provided.");
  }
  for (size_t ri = 0, re = mAllowedOutputRoutes.size(); ri != re; ++ri) {
    if (mAllowedOutputRoutes[ri].matcher.binding.value == ref.label) {
      auto spec = mAllowedOutputRoutes[ri].matcher;
      auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      return Output{ dataType.origin, dataType.description, ref.subSpec, spec.lifetime, std::move(ref.headerStack) };
    }
  }
  throw std::runtime_error("Unable to find OutputSpec with label " + ref.label);
  O2_BUILTIN_UNREACHABLE();
}

}
}
