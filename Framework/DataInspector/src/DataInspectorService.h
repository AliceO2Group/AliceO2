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
#ifndef O2_DATAINSPECTORSERVICE_H
#define O2_DATAINSPECTORSERVICE_H

#include <fairlogger/Logger.h>
#include "DISocket.h"
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
class DataInspectorProxyService
{
 public:
  DataInspectorProxyService(ServiceRegistryRef serviceRegistry,
                            DeviceSpec const& spec,
                            const std::string& address,
                            int port,
                            const std::string& runId);
  ~DataInspectorProxyService();

  void receive();
  void send(DIMessage&& message);
  bool isInspected() { return _isInspected; }

 private:
  void handleMessage(const DIMessage& msg);
  void terminate();

  const std::string deviceName;
  const std::string runId;

  bool _isInspected = false;
  DISocket socket;

  ServiceRegistryRef serviceRegistry;
};

struct DIServicePlugin : public ServicePlugin {
  auto create() -> ServiceSpec* final;
};
} // namespace o2::framework

#endif // O2_DATAINSPECTORSERVICE_H
