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

/**
 * Service used for communication with Proxy of DataInspector.
 */
class DataInspectorProxyService {
 public:
  DataInspectorProxyService(ServiceRegistry& serviceRegistry,
                            DeviceSpec const& spec,
                            const std::string& address,
                            int port,
                            const std::string& runId);
  ~DataInspectorProxyService();

  static std::unique_ptr<DataInspectorProxyService> create(ServiceRegistry& serviceRegistry,
                                                           DeviceSpec const& spec,
                                                           const std::string& address,
                                                           int port,
                                                           const std::string& runId);

  void receive();
  void send(DIMessage&& message);
  bool isInspected() { return _isInspected; }

 private:
  void handleMessage(DIMessage& msg);

  const std::string deviceName;
  const std::string runId;

  bool _isInspected = false;
  DISocket socket;

  ServiceRegistry& serviceRegistry;
};
}

#endif //O2_DATAINSPECTORSERVICE_H
