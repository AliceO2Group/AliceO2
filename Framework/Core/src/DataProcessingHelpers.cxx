// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DataProcessingHelpers.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ChannelInfo.h"
#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include <FairMQDevice.h>

namespace o2::framework
{
void DataProcessingHelpers::sendEndOfStream(FairMQDevice& device, OutputChannelSpec const& channel)
{
  FairMQParts parts;
  FairMQMessagePtr payload(device.NewMessage());
  SourceInfoHeader sih;
  sih.state = InputChannelState::Completed;
  auto channelAlloc = o2::pmr::getTransportAllocator(device.GetChannel(channel.name, 0).Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, sih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));
  device.Send(parts, channel.name, 0);
}
} // namespace o2::framework
