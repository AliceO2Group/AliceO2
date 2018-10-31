// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CHANNELMATCHING_H
#define FRAMEWORK_CHANNELMATCHING_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include <vector>
#include <string>

namespace o2
{
namespace framework
{

struct LogicalChannelRange {
  LogicalChannelRange(const OutputSpec& spec)
  {
    name = std::string("out_") +
           spec.origin.as<std::string>() + "_" +
           spec.description.as<std::string>() + "_" +
           std::to_string(spec.subSpec);
  }

  std::string name;
  bool operator<(LogicalChannelRange const& other) const
  {
    return this->name < other.name;
  }
};

struct DomainId {
  std::string value;
};

struct LogicalChannelDomain {
  LogicalChannelDomain(const InputSpec& spec)
  {
    name.value = std::string("out_") + spec.origin.as<std::string>() + "_" + spec.description.as<std::string>() + "_" + std::to_string(spec.subSpec);
  }
  DomainId name;
  bool operator<(LogicalChannelDomain const& other) const
  {
    return this->name.value < other.name.value;
  }
};

struct PhysicalChannelRange {
  PhysicalChannelRange(const OutputSpec& spec, int count)
  {
    char buffer[16];
    auto channel = LogicalChannelRange(spec);
    id = channel.name + (snprintf(buffer, 16, "_%d", count), buffer);
  }

  std::string id;
  bool operator<(PhysicalChannelRange const& other) const
  {
    return this->id < other.id;
  }
};

struct PhysicalChannelDomain {
  PhysicalChannelDomain(const InputSpec& spec, int count)
  {
    char buffer[16];
    auto channel = LogicalChannelDomain(spec);
    id.value = channel.name.value + (snprintf(buffer, 16, "_%d", count), buffer);
  }
  DomainId id;
  bool operator<(PhysicalChannelDomain const& other) const
  {
    return this->id.value < other.id.value;
  }
};

/// @return true if the doma
/// FIXME: for the moment we require a full match, however matcher could really be
///        a *-expression or even a regular expression.
inline bool intersect(const LogicalChannelDomain& targetDomain, const LogicalChannelRange& sourceRange)
{
  return targetDomain.name.value == sourceRange.name;
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CHANNELMATCHING_H
