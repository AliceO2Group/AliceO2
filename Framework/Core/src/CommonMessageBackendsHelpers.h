// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_COMMONMESSAGEBACKENDSHELPERS_H_
#define O2_FRAMEWORK_COMMONMESSAGEBACKENDSHELPERS_H_

#include "Framework/RawDeviceService.h"
#include "Framework/DataProcessor.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/Tracing.h"

#include <options/FairMQProgOptions.h>

namespace o2::framework
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
} // namespace o2::framework

#endif // O2_FRAMEWORK_COMMONMESSAGEBACKENDSHELPERS_H_
