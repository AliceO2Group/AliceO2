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
#include "Framework/CommonMessageBackends.h"
#include "Framework/MessageContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/StringContext.h"
#include "Framework/DataProcessor.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceInfo.h"

#include "CommonMessageBackendsHelpers.h"

#include <Monitoring/Monitoring.h>
#include <Headers/DataHeader.h>

#include <fairmq/ProgOptions.h>
#include <fairmq/Device.h>

#include <uv.h>
#include <boost/program_options/variables_map.hpp>
#include <csignal>

// This is to allow C++20 aggregate initialisation
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

class EndOfStreamContext;
class ProcessingContext;

o2::framework::ServiceSpec CommonMessageBackends::fairMQDeviceProxy()
{
  return ServiceSpec{
    .name = "fairmq-device-proxy",
    .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto* proxy = new FairMQDeviceProxy();
      return ServiceHandle{.hash = TypeIdHelpers::uniqueId<FairMQDeviceProxy>(), .instance = proxy, .kind = ServiceKind::Serial};
    },
    .start = [](ServiceRegistryRef services, void* instance) {
      auto* proxy = static_cast<FairMQDeviceProxy*>(instance);
      auto& outputs = services.get<DeviceSpec const>().outputs;
      auto& inputs = services.get<DeviceSpec const>().inputs;
      auto& forwards = services.get<DeviceSpec const>().forwards;
      auto* device = services.get<RawDeviceService>().device();
      /// Notice that we do it here (and not in the init) because
      /// some of the channels are added only later on to the party,
      /// (e.g. by ECS) and Init might not be late enough to
      /// account for them.
      proxy->bind(outputs, inputs, forwards, *device); },
  };
}

o2::framework::ServiceSpec CommonMessageBackends::fairMQBackendSpec()
{
  return ServiceSpec{
    .name = "fairmq-backend",
    .uniqueId = CommonServices::simpleServiceId<MessageContext>(),
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      auto& proxy = services.get<FairMQDeviceProxy>();
      auto context = new MessageContext(proxy);
      auto& spec = services.get<DeviceSpec const>();
      auto& dataSender = services.get<DataSender>();

      auto dispatcher = [&dataSender](fair::mq::Parts&& parts, ChannelIndex channelIndex, unsigned int) {
        dataSender.send(parts, channelIndex);
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
      return ServiceHandle{.hash = TypeIdHelpers::uniqueId<MessageContext>(), .instance = context, .kind = ServiceKind::Stream};
    },
    .configure = CommonServices::noConfiguration(),
    .preProcessing = CommonMessageBackendsHelpers<MessageContext>::clearContext(),
    .postProcessing = CommonMessageBackendsHelpers<MessageContext>::sendCallback(),
    .preEOS = CommonMessageBackendsHelpers<MessageContext>::clearContextEOS(),
    .postEOS = CommonMessageBackendsHelpers<MessageContext>::sendCallbackEOS(),
    .kind = ServiceKind::Stream};
}

o2::framework::ServiceSpec CommonMessageBackends::stringBackendSpec()
{
  return ServiceSpec{
    .name = "string-backend",
    .uniqueId = CommonServices::simpleServiceId<StringContext>(),
    .init = CommonMessageBackendsHelpers<StringContext>::createCallback(),
    .configure = CommonServices::noConfiguration(),
    .preProcessing = CommonMessageBackendsHelpers<StringContext>::clearContext(),
    .postProcessing = CommonMessageBackendsHelpers<StringContext>::sendCallback(),
    .preEOS = CommonMessageBackendsHelpers<StringContext>::clearContextEOS(),
    .postEOS = CommonMessageBackendsHelpers<StringContext>::sendCallbackEOS(),
    .kind = ServiceKind::Stream};
}

} // namespace o2::framework
