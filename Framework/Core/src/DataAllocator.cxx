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
#include "Framework/CompilerBuiltins.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/DataAllocator.h"
#include "Framework/MessageContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/Stack.h"
#include "FairMQResizableBuffer.h"

#include <fairmq/Device.h>

#include <arrow/ipc/writer.h>
#include <arrow/type.h>
#include <arrow/io/memory.h>
#include <arrow/util/config.h>

#include <TClonesArray.h>

namespace o2::framework
{

using DataHeader = o2::header::DataHeader;
using DataDescription = o2::header::DataDescription;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

DataAllocator::DataAllocator(ServiceRegistry* contextRegistry,
                             const AllowedOutputRoutes& routes)
  : mAllowedOutputRoutes{routes},
    mRegistry{contextRegistry}
{
}

RouteIndex DataAllocator::matchDataHeader(const Output& spec, size_t timeslice)
{
  // FIXME: we should take timeframeId into account as well.
  for (auto ri = 0; ri < mAllowedOutputRoutes.size(); ++ri) {
    auto& route = mAllowedOutputRoutes[ri];
    if (DataSpecUtils::match(route.matcher, spec.origin, spec.description, spec.subSpec) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
      return RouteIndex{ri};
    }
  }
  throw runtime_error_f(
    "Worker is not authorised to create message with "
    "origin(%s) description(%s) subSpec(%d)",
    spec.origin.as<std::string>().c_str(),
    spec.description.as<std::string>().c_str(),
    spec.subSpec);
}

DataChunk& DataAllocator::newChunk(const Output& spec, size_t size)
{
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto& context = mRegistry->get<MessageContext>();

  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,                     //
                                                               o2::header::gSerializationMethodNone, //
                                                               size                                  //
  );
  auto& co = context.add<MessageContext::ContainerRefObject<DataChunk>>(std::move(headerMessage), routeIndex, 0, size);
  return co;
}

void DataAllocator::adoptChunk(const Output& spec, char* buffer, size_t size, fair::mq::FreeFn* freefn, void* hint = nullptr)
{
  // Find a matching channel, create a new message for it and put it in the
  // queue to be sent at the end of the processing
  RouteIndex routeIndex = matchDataHeader(spec, mRegistry->get<TimingInfo>().timeslice);

  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,                     //
                                                               o2::header::gSerializationMethodNone, //
                                                               size                                  //
  );

  // FIXME: how do we want to use subchannels? time based parallelism?
  auto& context = mRegistry->get<MessageContext>();
  context.add<MessageContext::TrivialObject>(std::move(headerMessage), routeIndex, 0, buffer, size, freefn, hint);
}

fair::mq::MessagePtr DataAllocator::headerMessageFromOutput(Output const& spec,                     //
                                                            RouteIndex routeIndex,                  //
                                                            o2::header::SerializationMethod method, //
                                                            size_t payloadSize)                     //
{
  auto& timingInfo = mRegistry->get<TimingInfo>();
  DataHeader dh;
  dh.dataOrigin = spec.origin;
  dh.dataDescription = spec.description;
  dh.subSpecification = spec.subSpec;
  dh.payloadSize = payloadSize;
  dh.payloadSerializationMethod = method;
  dh.tfCounter = timingInfo.tfCounter;
  dh.firstTForbit = timingInfo.firstTForbit;
  dh.runNumber = timingInfo.runNumber;

  DataProcessingHeader dph{timingInfo.timeslice, 1, timingInfo.creation};
  auto& context = mRegistry->get<MessageContext>();
  auto& proxy = mRegistry->get<FairMQDeviceProxy>();
  auto* transport = proxy.getOutputTransport(routeIndex);

  auto channelAlloc = o2::pmr::getTransportAllocator(transport);
  return o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph, spec.metaHeader});
}

void DataAllocator::addPartToContext(fair::mq::MessagePtr&& payloadMessage, const Output& spec,
                                     o2::header::SerializationMethod serializationMethod)
{
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto headerMessage = headerMessageFromOutput(spec, routeIndex, serializationMethod, 0);

  // FIXME: this is kind of ugly, we know that we can change the content of the
  // header message because we have just created it, but the API declares it const
  const DataHeader* cdh = o2::header::get<DataHeader*>(headerMessage->GetData());
  auto* dh = const_cast<DataHeader*>(cdh);
  dh->payloadSize = payloadMessage->GetSize();
  auto& context = mRegistry->get<MessageContext>();
  // make_scoped creates the context object inside of a scope handler, since it goes out of
  // scope immediately, the created object is scheduled and can be directly sent if the context
  // is configured with the dispatcher callback
  context.make_scoped<MessageContext::TrivialObject>(std::move(headerMessage), std::move(payloadMessage), routeIndex);
}

void DataAllocator::adopt(const Output& spec, std::string* ptr)
{
  std::unique_ptr<std::string> payload(ptr);
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  // the correct payload size is set later when sending the
  // StringContext, see DataProcessor::doSend
  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodNone, 0);
  mRegistry->get<StringContext>().addString(std::move(header), std::move(payload), routeIndex);
  assert(payload.get() == nullptr);
}

void doWriteTable(std::shared_ptr<FairMQResizableBuffer> b, arrow::Table* table)
{
  auto mock = std::make_shared<arrow::io::MockOutputStream>();
  int64_t expectedSize = 0;
  auto mockWriter = arrow::ipc::MakeStreamWriter(mock.get(), table->schema());
  arrow::Status outStatus;
  if (O2_BUILTIN_LIKELY(table->num_rows() != 0)) {
    outStatus = mockWriter.ValueOrDie()->WriteTable(*table);
  } else {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.resize(table->columns().size());
    for (size_t ci = 0; ci < table->columns().size(); ci++) {
      columns[ci] = table->column(ci)->chunk(0);
    }
    auto batch = arrow::RecordBatch::Make(table->schema(), 0, columns);
    outStatus = mockWriter.ValueOrDie()->WriteRecordBatch(*batch);
  }

  expectedSize = mock->Tell().ValueOrDie();
  auto reserve = b->Reserve(expectedSize);
  if (reserve.ok() == false) {
    throw std::runtime_error("Unable to reserve memory for table");
  }

  auto stream = std::make_shared<FairMQOutputStream>(b);
  auto outBatch = arrow::ipc::MakeStreamWriter(stream.get(), table->schema());
  if (outBatch.ok() == false) {
    throw ::std::runtime_error("Unable to create batch writer");
  }

  if (O2_BUILTIN_UNLIKELY(table->num_rows() == 0)) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.resize(table->columns().size());
    for (size_t ci = 0; ci < table->columns().size(); ci++) {
      columns[ci] = table->column(ci)->chunk(0);
    }
    auto batch = arrow::RecordBatch::Make(table->schema(), 0, columns);
    outStatus = outBatch.ValueOrDie()->WriteRecordBatch(*batch);
  } else {
    outStatus = outBatch.ValueOrDie()->WriteTable(*table);
  }

  if (outStatus.ok() == false) {
    throw std::runtime_error("Unable to Write table");
  }
}

void DataAllocator::adopt(const Output& spec, TableBuilder* tb)
{
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodArrow, 0);
  auto& context = mRegistry->get<ArrowContext>();
  auto* transport = context.proxy().getOutputTransport(routeIndex);
  assert(transport != nullptr);

  auto creator = [transport](size_t s) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(s);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  /// To finalise this we write the table to the buffer.
  /// FIXME: most likely not a great idea. We should probably write to the buffer
  ///        directly in the TableBuilder, incrementally.
  std::shared_ptr<TableBuilder> p(tb);
  auto finalizer = [payload = p](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    auto table = payload->finalize();
    doWriteTable(b, table.get());
  };

  context.addBuffer(std::move(header), buffer, std::move(finalizer), routeIndex);
}

void DataAllocator::adopt(const Output& spec, TreeToTable* t2t)
{
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);

  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodArrow, 0);
  auto& context = mRegistry->get<ArrowContext>();

  auto creator = [transport = context.proxy().getOutputTransport(routeIndex)](size_t s) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(s);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  /// To finalise this we write the table to the buffer.
  /// FIXME: most likely not a great idea. We should probably write to the buffer
  ///        directly in the TableBuilder, incrementally.
  auto finalizer = [payload = t2t](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    auto table = payload->finalize();
    doWriteTable(b, table.get());
    delete payload;
  };

  context.addBuffer(std::move(header), buffer, std::move(finalizer), routeIndex);
}

void DataAllocator::adopt(const Output& spec, std::shared_ptr<arrow::Table> ptr)
{
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodArrow, 0);
  auto& context = mRegistry->get<ArrowContext>();

  auto creator = [transport = context.proxy().getOutputTransport(routeIndex)](size_t s) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(s);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  auto writer = [table = ptr](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    doWriteTable(b, table.get());
  };

  context.addBuffer(std::move(header), buffer, std::move(writer), routeIndex);
}

void DataAllocator::snapshot(const Output& spec, const char* payload, size_t payloadSize,
                             o2::header::SerializationMethod serializationMethod)
{
  auto& proxy = mRegistry->get<FairMQDeviceProxy>();
  auto& timingInfo = mRegistry->get<TimingInfo>();

  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  fair::mq::MessagePtr payloadMessage(proxy.createOutputMessage(routeIndex, payloadSize));
  memcpy(payloadMessage->GetData(), payload, payloadSize);

  addPartToContext(std::move(payloadMessage), spec, serializationMethod);
}

Output DataAllocator::getOutputByBind(OutputRef&& ref)
{
  if (ref.label.empty()) {
    throw runtime_error("Invalid (empty) OutputRef provided.");
  }
  for (auto ri = 0ul, re = mAllowedOutputRoutes.size(); ri != re; ++ri) {
    if (mAllowedOutputRoutes[ri].matcher.binding.value == ref.label) {
      auto spec = mAllowedOutputRoutes[ri].matcher;
      auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      return Output{dataType.origin, dataType.description, ref.subSpec, spec.lifetime, std::move(ref.headerStack)};
    }
  }
  throw runtime_error_f("Unable to find OutputSpec with label %s", ref.label.c_str());
  O2_BUILTIN_UNREACHABLE();
}

bool DataAllocator::isAllowed(Output const& query)
{
  for (auto const& route : mAllowedOutputRoutes) {
    if (DataSpecUtils::match(route.matcher, query.origin, query.description, query.subSpec)) {
      return true;
    }
  }
  return false;
}

void DataAllocator::adoptFromCache(const Output& spec, CacheId id, header::SerializationMethod method)
{
  // Find a matching channel, extract the message for it form the container
  // and put it in the queue to be sent at the end of the processing
  auto& timingInfo = mRegistry->get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);

  auto& context = mRegistry->get<MessageContext>();
  fair::mq::MessagePtr payloadMessage = context.cloneFromCache(id.value);

  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,         //
                                                               method,                   //
                                                               payloadMessage->GetSize() //
  );

  context.add<MessageContext::TrivialObject>(std::move(headerMessage), std::move(payloadMessage), routeIndex);
}

} // namespace o2::framework
