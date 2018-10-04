// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/LifetimeHelpers.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/InputRoute.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"
#include "Framework/DataProcessingHeader.h"
#include <fairmq/FairMQDevice.h>

using namespace o2::header;
using namespace fair;

namespace o2
{
namespace framework
{

ExpirationHandler::Checker LifetimeHelpers::expireNever()
{
  return [](int64_t) -> bool { return false; };
}

ExpirationHandler::Checker LifetimeHelpers::expireAlways()
{
  return [](int64_t) -> bool { return true; };
}

namespace
{
auto getCurrentTime()
{
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}
}

ExpirationHandler::Checker LifetimeHelpers::expireTimed(std::chrono::milliseconds period)
{
  auto start = getCurrentTime();
  auto last = std::make_shared<decltype(start)>(start);
  return [last, period](int64_t) -> bool {
    auto current = getCurrentTime();
    auto delta = current - *last;
    if (delta > period.count()) {
      *last = current;
      return true;
    }
    return false;
  };
}

/// Does nothing. Use this for cases where you do not want to do anything
/// when records expire. This is the default behavior for data (which never
/// expires via this mechanism).
ExpirationHandler::Handler LifetimeHelpers::doNothing()
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void { return; };
}

/// Fetch an object from CCDB if the record is expired. The actual
/// name of the object is given by:
///
/// "<namespace>/<InputRoute.origin>/<InputRoute.description>"
///
/// FIXME: actually implement the fetching
/// FIXME: provide a way to customize the namespace from the ProcessingContext
ExpirationHandler::Handler LifetimeHelpers::fetchFromCCDBCache(std::string const& prefix)
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void {
    throw std::runtime_error("fetchFromCCDBCache: Not yet implemented");
    return;
  };
}

/// Create an entry in the registry for histograms on the first
/// FIXME: actually implement this
/// FIXME: provide a way to customise the histogram from the configuration.
ExpirationHandler::Handler LifetimeHelpers::fetchFromQARegistry()
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void {
    throw std::runtime_error("fetchFromQARegistry: Not yet implemented");
    return;
  };
}

/// Create an entry in the registry for histograms on the first
/// FIXME: actually implement this
/// FIXME: provide a way to customise the histogram from the configuration.
ExpirationHandler::Handler LifetimeHelpers::fetchFromObjectRegistry()
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void {
    throw std::runtime_error("fetchFromObjectRegistry: Not yet implemented");
    return;
  };
}

/// Enumerate entries on every invokation.
ExpirationHandler::Handler LifetimeHelpers::enumerate(InputRoute const& route)
{
  auto counter = std::make_shared<int64_t>(0);
  auto f = [route, counter](ServiceRegistry& services, PartRef& ref, uint64_t timestamp) -> void {
    // We should invoke the handler only once.
    assert(!ref.header);
    assert(!ref.payload);
    auto& rawDeviceService = services.get<RawDeviceService>();

    DataHeader dh;
    dh.dataOrigin = route.matcher.origin;
    dh.dataDescription = route.matcher.description;
    dh.subSpecification = route.matcher.subSpec;
    dh.payloadSize = 8;
    dh.payloadSerializationMethod = gSerializationMethodNone;

    DataProcessingHeader dph{ timestamp, 1 };

    auto&& transport = rawDeviceService.device()->GetChannel(route.sourceChannel, 0).Transport();
    auto channelAlloc = o2::memory_resource::getTransportAllocator(transport);
    auto header = o2::memory_resource::getMessage(o2::header::Stack{ channelAlloc, dh, dph });
    ref.header = std::move(header);

    auto payload = rawDeviceService.device()->NewMessage(*counter);
    ref.payload = std::move(payload);
    (*counter)++;
  };
  return f;
}

} // namespace framework
} // namespace o2
