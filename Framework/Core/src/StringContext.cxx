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

#include "Framework/StringContext.h"
#include <fairmq/Message.h>
#include <cassert>

namespace o2::framework
{

void StringContext::addString(std::unique_ptr<fair::mq::Message> header,
                              std::unique_ptr<std::string> s,
                              RouteIndex routeIndex)
{
  mMessages.push_back(std::move(MessageRef{std::move(header),
                                           std::move(s),
                                           routeIndex}));
}

void StringContext::clear()
{
  // On send we move the header, but the payload remains
  // there because what's really sent is the copy of the string
  // payload will be cleared by the mMessages.clear()
  for (auto& m : mMessages) {
    assert(m.header.get() == nullptr);
    assert(m.payload.get() != nullptr);
  }
  mMessages.clear();
}

} // namespace o2::framework
