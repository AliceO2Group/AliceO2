// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/MessageContext.h"
#include "fairmq/FairMQDevice.h"

namespace o2
{
namespace framework
{

FairMQMessagePtr MessageContext::createMessage(const std::string& channel, int index, size_t size)
{
  return proxy().getDevice()->NewMessageFor(channel, 0, size);
}

FairMQMessagePtr MessageContext::createMessage(const std::string& channel, int index, void* data, size_t size, fairmq_free_fn* ffn, void* hint)
{
  return proxy().getDevice()->NewMessageFor(channel, 0, data, size, ffn, hint);
}

} // namespace framework
} // namespace o2
