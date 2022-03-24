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

#include "Framework/Output.h"
#include "Framework/MessageContext.h"
#include "fairmq/FairMQDevice.h"

namespace o2::framework
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

o2::framework::DataProcessingHeader* MessageContext::findMessageDataProcessingHeader(const Output& spec)
{
  for (auto it = mMessages.rbegin(); it != mMessages.rend(); ++it) {
    const auto* hd = (*it)->header();
    if (hd->dataOrigin == spec.origin && hd->dataDescription == spec.description && hd->subSpecification == spec.subSpec) {
      return const_cast<o2::framework::DataProcessingHeader*>((*it)->dataProcessingHeader());
    }
  }
  for (auto it = mScheduledMessages.rbegin(); it != mScheduledMessages.rend(); ++it) {
    const auto* hd = (*it)->header();
    if (hd->dataOrigin == spec.origin && hd->dataDescription == spec.description && hd->subSpecification == spec.subSpec) {
      return const_cast<o2::framework::DataProcessingHeader*>((*it)->dataProcessingHeader()); // o2::header::get returns const pointer, but the caller may need non-const
    }
  }
  return nullptr;
}

o2::header::Stack* MessageContext::findMessageHeaderStack(const Output& spec)
{
  std::pair<o2::header::DataHeader*, o2::framework::DataProcessingHeader*> h{};
  for (auto it = mMessages.rbegin(); it != mMessages.rend(); ++it) {
    const auto* hd = (*it)->header();
    if (hd->dataOrigin == spec.origin && hd->dataDescription == spec.description && hd->subSpecification == spec.subSpec) {
      return const_cast<o2::header::Stack*>((*it)->headerStack());
    }
  }
  for (auto it = mScheduledMessages.rbegin(); it != mScheduledMessages.rend(); ++it) {
    const auto* hd = (*it)->header();
    if (hd->dataOrigin == spec.origin && hd->dataDescription == spec.description && hd->subSpecification == spec.subSpec) {
      return const_cast<o2::header::Stack*>((*it)->headerStack());
    }
  }
  return nullptr;
}

int MessageContext::countDeviceOutputs(bool excludeDPLOrigin)
{
  int noutputs = 0;
  constexpr o2::header::DataOrigin DataOriginDPL{"DPL"};
  for (auto it = mMessages.rbegin(); it != mMessages.rend(); ++it) {
    if (!excludeDPLOrigin || (*it)->header()->dataOrigin != DataOriginDPL) {
      noutputs++;
    }
  }
  for (auto it = mScheduledMessages.rbegin(); it != mScheduledMessages.rend(); ++it) {
    if (!excludeDPLOrigin || (*it)->header()->dataOrigin != DataOriginDPL) {
      noutputs++;
    }
  }
  return noutputs;
}

int64_t MessageContext::addToCache(std::unique_ptr<FairMQMessage>& toCache)
{
  auto&& cached = toCache->GetTransport()->CreateMessage();
  cached->Copy(*toCache);
  // The pointer is immutable!
  int64_t cacheId = (int64_t)toCache->GetData();
  mMessageCache.insert({cacheId, std::move(cached)});
  return cacheId;
}

std::unique_ptr<FairMQMessage> MessageContext::cloneFromCache(int64_t id) const
{
  auto& inCache = mMessageCache.at(id);
  auto&& cloned = inCache->GetTransport()->CreateMessage();
  cloned->Copy(*inCache);
  return std::move(cloned);
}

void MessageContext::pruneFromCache(int64_t id)
{
  mMessageCache.erase(id);
}

} // namespace o2::framework
