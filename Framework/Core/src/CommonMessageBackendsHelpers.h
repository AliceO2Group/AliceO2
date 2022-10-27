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
#ifndef O2_FRAMEWORK_COMMONMESSAGEBACKENDSHELPERS_H_
#define O2_FRAMEWORK_COMMONMESSAGEBACKENDSHELPERS_H_

#include "Framework/RawDeviceService.h"
#include "Framework/DataProcessor.h"
#include "Framework/DataSender.h"
#include "Framework/ProcessingContext.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/Tracing.h"

#include <fairmq/ProgOptions.h>

namespace o2::framework
{

template <typename T>
struct CommonMessageBackendsHelpers {
  static ServiceInit createCallback()
  {
    return [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) {
      auto& proxy = services.get<FairMQDeviceProxy>();
      return ServiceHandle{TypeIdHelpers::uniqueId<T>(), new T(proxy)};
    };
  }

  static ServiceProcessingCallback sendCallback()
  {
    return [](ProcessingContext& ctx, void* service) {
      ZoneScopedN("send message callback");
      T* context = reinterpret_cast<T*>(service);
      DataProcessor::doSend(ctx.services().get<DataSender>(), *context, ctx.services());
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
      DataProcessor::doSend(ctx.services().get<DataSender>(), *context, ctx.services());
    };
  }
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_COMMONMESSAGEBACKENDSHELPERS_H_
