// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Output.h"
#include "Framework/MessageContext.h"
#include "fairmq/FairMQDevice.h"

namespace o2
{
namespace framework
{

FairMQMessagePtr MessageContext::createMessage(const std::string& channel, int index, size_t size)
{
  return proxy().getDevice()->NewMessageFor(channel, 0, size, fair::mq::Alignment{64});
}

FairMQMessagePtr MessageContext::createMessage(const std::string& channel, int index, void* data, size_t size, fairmq_free_fn* ffn, void* hint)
{
  return proxy().getDevice()->NewMessageFor(channel, 0, data, size, ffn, hint);
}

o2::header::DataHeader* MessageContext::findMessageHeader(const Output& spec)
{
  for (auto it = mMessages.rbegin(); it != mMessages.rend(); ++it) {
    const auto* hd = (*it)->header();
    if (hd->dataOrigin == spec.origin && hd->dataDescription == spec.description && hd->subSpecification == spec.subSpec) {
      return const_cast<o2::header::DataHeader*>(hd); // o2::header::get returns const pointer, but the caller may need non-const
    }
  }
  for (auto it = mScheduledMessages.rbegin(); it != mScheduledMessages.rend(); ++it) {
    const auto* hd = (*it)->header();
    if (hd->dataOrigin == spec.origin && hd->dataDescription == spec.description && hd->subSpecification == spec.subSpec) {
      return const_cast<o2::header::DataHeader*>(hd); // o2::header::get returns const pointer, but the caller may need non-const
    }
  }
  return nullptr;
}

} // namespace framework
} // namespace o2
