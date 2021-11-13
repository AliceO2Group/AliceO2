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

#include "Framework/DataSender.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"

#include <fairmq/Device.h>

namespace o2::framework
{

DataSender::DataSender(ServiceRegistry& registry,
                       SendingPolicy const& policy)
  : mContext{registry.get<RawDeviceService>().device()},
    mRegistry{registry},
    mPolicy{policy}
{
}

std::unique_ptr<FairMQMessage> DataSender::create()
{
  FairMQDevice* device = (FairMQDevice*)mContext;
  return device->NewMessage();
}

void DataSender::send(FairMQParts& parts, std::string const& channel)
{
  FairMQDevice* device = (FairMQDevice*)mContext;
  mPolicy.send(*device, parts, channel);
}

} // namespace o2::framework
