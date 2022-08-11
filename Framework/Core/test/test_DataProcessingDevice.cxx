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

#define BOOST_TEST_MODULE Test Framework DataProcessingDevice
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/DataProcessingDevice.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/ServiceRegistryHelpers.h"
#include "Framework/CommonServices.h"
#include "Framework/DeviceState.h"
#include <fairmq/ProgOptions.h>

// Make sure DataProcessingDevice::handleData can handle 0 parts just fine
BOOST_AUTO_TEST_CASE(HandleDataZeroParts)
{
  using namespace o2::framework;
  o2::framework::DataProcessorContext context;
  context.registry = new o2::framework::ServiceRegistry;
  DeviceState deviceState;
  std::vector<o2::framework::ServiceSpec> services{
    CommonServices::dataProcessingStats(),
  };
  fair::mq::ProgOptions options;
  for (auto& service : services) {
    context.registry->declareService(service, deviceState, options);
  }
  o2::framework::InputChannelInfo channelInfo;
  o2::framework::DataProcessingDevice::handleData(context, channelInfo);
}
