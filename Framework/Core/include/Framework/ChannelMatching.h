// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CHANNELMATCHING_H_
#define O2_FRAMEWORK_CHANNELMATCHING_H_

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"

#include <string>

namespace o2::framework
{

struct LogicalChannelRange {
  LogicalChannelRange(OutputSpec const& spec);

  std::string name;
  bool operator<(LogicalChannelRange const& other) const;
};

struct DomainId {
  std::string value;
};

struct LogicalChannelDomain {
  LogicalChannelDomain(InputSpec const& spec);
  DomainId name;
  bool operator<(LogicalChannelDomain const& other) const;
};

struct PhysicalChannelRange {
  PhysicalChannelRange(OutputSpec const& spec, int count);

  std::string id;
  bool operator<(PhysicalChannelRange const& other) const;
};

struct PhysicalChannelDomain {
  PhysicalChannelDomain(InputSpec const& spec, int count);

  DomainId id;
  bool operator<(PhysicalChannelDomain const& other) const;
};

} // namespace o2::framework
#endif // FRAMEWORK_CHANNELMATCHING_H
