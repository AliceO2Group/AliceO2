// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/FairMQDeviceProxy.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQMessage.h>

namespace o2
{
namespace framework
{
FairMQTransportFactory* FairMQDeviceProxy::getTransport()
{
  return mDevice->Transport();
}

FairMQTransportFactory* FairMQDeviceProxy::getTransport(const std::string& channel, const int index)
{
  return mDevice->GetChannel(channel, index).Transport();
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createMessage() const
{
  return mDevice->Transport()->CreateMessage();
}

std::unique_ptr<FairMQMessage> FairMQDeviceProxy::createMessage(const size_t size) const
{
  return mDevice->Transport()->CreateMessage(size);
}

} // namespace framework
} // namespace o2
