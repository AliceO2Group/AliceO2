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
#include "Framework/RawBufferContext.h"
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

#include <options/FairMQProgOptions.h>
#include <fairmq/Device.h>

#include <uv.h>
#include <boost/program_options/variables_map.hpp>
#include <csignal>

// This is to allow C++20 aggregate initialisation
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

struct EndOfStreamContext;
struct ProcessingContext;

o2::framework::ServiceSpec CommonMessageBackends::fairMQBackendSpec()
{
  return ServiceSpec{
    .name = "fairmq-backend",
    .init = [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      auto& device = services.get<RawDeviceService>();
      auto context = new MessageContext(FairMQDeviceProxy{device.device()});
      auto& spec = services.get<DeviceSpec const>();

      auto dispatcher = [&device](FairMQParts&& parts, std::string const& channel, unsigned int index) {
        device.device()->Send(parts, channel, index);
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
    .configure = CommonServices::noConfiguration(),
    .preProcessing = CommonMessageBackendsHelpers<MessageContext>::clearContext(),
    .postProcessing = CommonMessageBackendsHelpers<MessageContext>::sendCallback(),
    .preEOS = CommonMessageBackendsHelpers<MessageContext>::clearContextEOS(),
    .postEOS = CommonMessageBackendsHelpers<MessageContext>::sendCallbackEOS(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::stringBackendSpec()
{
  return ServiceSpec{
    .name = "string-backend",
    .init = CommonMessageBackendsHelpers<StringContext>::createCallback(),
    .configure = CommonServices::noConfiguration(),
    .preProcessing = CommonMessageBackendsHelpers<StringContext>::clearContext(),
    .postProcessing = CommonMessageBackendsHelpers<StringContext>::sendCallback(),
    .preEOS = CommonMessageBackendsHelpers<StringContext>::clearContextEOS(),
    .postEOS = CommonMessageBackendsHelpers<StringContext>::sendCallbackEOS(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::rawBufferBackendSpec()
{
  return ServiceSpec{
    .name = "raw-backend",
    .init = CommonMessageBackendsHelpers<RawBufferContext>::createCallback(),
    .configure = CommonServices::noConfiguration(),
    .preProcessing = CommonMessageBackendsHelpers<RawBufferContext>::clearContext(),
    .postProcessing = CommonMessageBackendsHelpers<RawBufferContext>::sendCallback(),
    .preEOS = CommonMessageBackendsHelpers<RawBufferContext>::clearContextEOS(),
    .postEOS = CommonMessageBackendsHelpers<RawBufferContext>::sendCallbackEOS(),
    .kind = ServiceKind::Serial};
}

} // namespace o2::framework

#pragma GCC diagnostic pop
