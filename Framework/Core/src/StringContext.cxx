// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/StringContext.h"
#include <FairMQMessage.h>
#include <cassert>

namespace o2::framework
{

void StringContext::addString(std::unique_ptr<FairMQMessage> header,
                              std::unique_ptr<std::string> s,
                              const std::string& channel)
{
  mMessages.push_back(std::move(MessageRef{std::move(header),
                                           std::move(s),
                                           channel}));
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
