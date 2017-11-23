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

namespace o2 {
namespace framework {

struct LogicalChannel {
  std::string name;
  bool operator<(LogicalChannel const&other) const {
    return this->name < other.name;
  }
};

struct PhysicalChannel {
  std::string id;
  bool operator<(PhysicalChannel const&other) const {
    return this->id < other.id;
  }
};

inline LogicalChannel outputSpec2LogicalChannel(const OutputSpec &spec) {
  auto name = std::string("out_") +
              spec.origin.str + "_" +
              spec.description.str + "_" +
              std::to_string(spec.subSpec);
  return LogicalChannel{name};
}

inline PhysicalChannel outputSpec2PhysicalChannel(const OutputSpec &spec, int count) {
  char buffer[16];
  auto channel = outputSpec2LogicalChannel(spec);
  return PhysicalChannel{channel.name + (snprintf(buffer, 16, "_%d", count), buffer)};
}

inline LogicalChannel inputSpec2LogicalChannelMatcher(const InputSpec &spec) {
  auto name = std::string("out_") + spec.origin.str + "_" + spec.description.str + "_" + std::to_string(spec.subSpec);
  return LogicalChannel{name};
}

inline PhysicalChannel inputSpec2PhysicalChannelMatcher(const InputSpec&spec, int count) {
  char buffer[16];
  auto channel = inputSpec2LogicalChannelMatcher(spec);
  return PhysicalChannel{channel.name + (snprintf(buffer, 16, "_%d", count), buffer)};
}

/// @return true if a given DataSpec can use the provided channel.
/// FIXME: for the moment we require a full match, however matcher could really be
///        a *-expression or even a regular expression.
inline bool matchDataSpec2Channel(const InputSpec &spec, const LogicalChannel &channel) {
  auto matcher = inputSpec2LogicalChannelMatcher(spec);
  return matcher.name == channel.name;
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CHANNELMATCHING_H
