#ifndef O2_DATAINSPECTORSERVICE_H
#define O2_DATAINSPECTORSERVICE_H

#include <fairlogger/Logger.h>
#include "DISocket.hpp"
#include "Framework/RoutingIndices.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/ServiceSpec.h"
#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

namespace o2::framework
{
class DataInspectorProxyService {
 public:
  DataInspectorProxyService(const std::string& deviceName, const std::string& address, int port);
  ~DataInspectorProxyService();

  static std::unique_ptr<DataInspectorProxyService> create(DeviceSpec const& spec, const std::string& address, int port);

  void receive();
  void send(DIMessage&& message);
  bool isInspected() { return _isInspected; }

 private:
  void handleMessage(DIMessage& msg);

  const std::string deviceName;
  bool _isInspected = false;
  DISocket socket;
};
}

#endif //O2_DATAINSPECTORSERVICE_H
