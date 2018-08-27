// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ChannelSpecHelpers.h"
#include <ostream>

namespace o2
{
namespace framework
{

char const* ChannelSpecHelpers::typeAsString(enum ChannelType type)
{
  switch (type) {
    case ChannelType::Pub:
      return "pub";
    case ChannelType::Sub:
      return "sub";
    case ChannelType::Push:
      return "push";
    case ChannelType::Pull:
      return "pull";
  }
}

char const* ChannelSpecHelpers::methodAsString(enum ChannelMethod method)
{
  switch (method) {
    case ChannelMethod::Bind:
      return "bind";
    case ChannelMethod::Connect:
      return "connect";
  }
}

char const* ChannelSpecHelpers::methodAsUrl(enum ChannelMethod method)
{
  return (method == ChannelMethod::Bind ? "tcp://*:%d" : "tcp://127.0.0.1:%d");
}

/// Stream operators so that we can use ChannelType with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelType const& type)
{
  s << ChannelSpecHelpers::typeAsString(type);
  return s;
}

/// Stream operators so that we can use ChannelString with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelMethod const& method)
{
  s << ChannelSpecHelpers::methodAsString(method);
  return s;
}

} // namespace framework
} // namespace o2
