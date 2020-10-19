// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CommonMessageBackends.h"
#include "Framework/MessageContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/StringContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/DataProcessor.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"

#include <Monitoring/Monitoring.h>
#include <Headers/DataHeader.h>

#include <options/FairMQProgOptions.h>

namespace o2::framework
{

struct EndOfStreamContext;
struct ProcessingContext;

namespace
{
template <typename T>
struct CommonMessageBackendsHelpers {
  static ServiceInit createCallback()
  {
    return [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) {
      auto& device = services.get<RawDeviceService>();
      return ServiceHandle{TypeIdHelpers::uniqueId<T>(), new T(FairMQDeviceProxy{device.device()})};
    };
  }

  static ServiceProcessingCallback sendCallback()
  {
    return [](ProcessingContext& ctx, void* service) {
      ZoneScopedN("send message callback");
      T* context = reinterpret_cast<T*>(service);
      auto& device = ctx.services().get<RawDeviceService>();
      DataProcessor::doSend(*device.device(), *context, ctx.services());
    };
  }

  static ServiceProcessingCallback clearContext()
  {
    return [](ProcessingContext&, void* service) {
      T* context = reinterpret_cast<T*>(service);
      context->clear();
    };
  }

  static ServiceEOSCallback clearContextEOS()
  {
    return [](EndOfStreamContext&, void* service) {
      T* context = reinterpret_cast<T*>(service);
      context->clear();
    };
  }

  static ServiceEOSCallback sendCallbackEOS()
  {
    return [](EndOfStreamContext& ctx, void* service) {
      T* context = reinterpret_cast<T*>(service);
      auto& device = ctx.services().get<RawDeviceService>();
      DataProcessor::doSend(*device.device(), *context, ctx.services());
    };
  }
};
} // namespace

o2::framework::ServiceSpec CommonMessageBackends::arrowBackendSpec()
{
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  return ServiceSpec{"arrow-backend",
                     CommonMessageBackendsHelpers<ArrowContext>::createCallback(),
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<ArrowContext>::clearContext(),
                     CommonMessageBackendsHelpers<ArrowContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<ArrowContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<ArrowContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     [](ProcessingContext& ctx, void* service) {
                       using DataHeader = o2::header::DataHeader;
                       ArrowContext* arrow = reinterpret_cast<ArrowContext*>(service);
                       for (auto& input : ctx.inputs()) {
                         if (input.header == nullptr) {
                           continue;
                         }
                         auto dh = o2::header::get<DataHeader*>(input.header);
                         if (dh->serialization != o2::header::gSerializationMethodArrow) {
                           continue;
                         }
                         auto dph = o2::header::get<DataProcessingHeader*>(input.header);
                         for (auto const& forward : ctx.services().get<DeviceSpec const>().forwards) {
                           if (DataSpecUtils::match(forward.matcher, dh->dataOrigin, dh->dataDescription, dh->subSpecification) == true && (dph->startTime % forward.maxTimeslices) == forward.timeslice) {
                             continue;
                           }
                         }
                         arrow->updateBytesDestroyed(dh->payloadSize);
                         arrow->updateMessagesDestroyed(1);
                       }
                       ctx.services().get<Monitoring>().send(Metric{(uint64_t)arrow->bytesDestroyed(), "arrow-bytes-destroyed"}.addTag(Key::Subsystem, Value::DPL));
                       ctx.services().get<Monitoring>().send(Metric{(uint64_t)arrow->messagesDestroyed(), "arrow-messages-destroyed"}.addTag(Key::Subsystem, Value::DPL));
                     },
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::fairMQBackendSpec()
{
  return ServiceSpec{"fairmq-backend",
                     [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
                       auto& device = services.get<RawDeviceService>();
                       auto context = new MessageContext(FairMQDeviceProxy{device.device()});
                       auto& spec = services.get<DeviceSpec const>();

                       auto dispatcher = [&device](FairMQParts&& parts, std::string const& channel, unsigned int index) {
                         DataProcessor::doSend(*device.device(), std::move(parts), channel.c_str(), index);
                       };

                       auto matcher = [policy = spec.dispatchPolicy](o2::header::DataHeader const& header) {
                         if (policy.triggerMatcher == nullptr) {
                           return true;
                         }
                         return policy.triggerMatcher(Output{header});
                       };

                       if (spec.dispatchPolicy.action == DispatchPolicy::DispatchOp::WhenReady) {
                         context->init(DispatchControl{dispatcher, matcher});
                       }
                       return ServiceHandle{TypeIdHelpers::uniqueId<MessageContext>(), context};
                     },
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<MessageContext>::clearContext(),
                     CommonMessageBackendsHelpers<MessageContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<MessageContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<MessageContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::stringBackendSpec()
{
  return ServiceSpec{"string-backend",
                     CommonMessageBackendsHelpers<StringContext>::createCallback(),
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<StringContext>::clearContext(),
                     CommonMessageBackendsHelpers<StringContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<StringContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<StringContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::rawBufferBackendSpec()
{
  return ServiceSpec{"raw-backend",
                     CommonMessageBackendsHelpers<RawBufferContext>::createCallback(),
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<RawBufferContext>::clearContext(),
                     CommonMessageBackendsHelpers<RawBufferContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<RawBufferContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<RawBufferContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

} // namespace o2::framework
